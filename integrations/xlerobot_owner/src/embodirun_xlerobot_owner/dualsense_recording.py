"""AGX-local demo capture: passive camera/state reader and exact control log.

Disk and image work never runs in the controller loop. Observations and control
events keep their own AGX timestamps; no nearest-frame action is fabricated.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path

from .recording import EpisodeRecorder
from .robot import RemoteRobot


def input_record(sample):
    return {
        "raw_axes": list(sample.raw_axes),
        "sticks": list(sample.sticks),
        "triggers": list(sample.triggers),
        "buttons": sorted(sample.buttons),
        "received_monotonic_s": sample.received_at,
        "sequence": sample.sequence,
        "report_format": sample.report_format,
    }


class DemoRecording:
    def __init__(self, output, task, robot_url, token, metadata, *, fps=10, max_seconds=600, observer=None):
        self.observer = observer or RemoteRobot(robot_url, token, timeout=1, scope="base")
        self.recorder = EpisodeRecorder(Path(output), fps=fps)
        self.path = self.recorder.start(
            task,
            {
                **metadata,
                "operator_mode": "dualsense",
                "collection_mode": "navigation_reference",
                "trainable": False,
                "control_events": "control.jsonl",
                "alignment": "AGX timestamps; observations and commands are separate, not frame-aligned actions",
            },
        )
        self.fps, self.max_seconds = fps, max_seconds
        self.events = queue.Queue(maxsize=256)
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.frames = 0
        self.result = None
        self._thread = threading.Thread(target=self._capture, name="dualsense-demo", daemon=True)
        self._thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("示范录制失败，停止控制: " + self.error)

    def wait_ready(self, timeout=5):
        if not self.ready.wait(timeout):
            raise RuntimeError("示范录制尚未收到三路新鲜相机画面")
        self.check()

    def emit(self, kind, **values):
        self.check()
        line = (
            json.dumps(
                {"kind": kind, "timestamp_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(), **values},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            self.events.put_nowait(line)
        except queue.Full as exc:
            raise RuntimeError("录制写盘积压，停止控制") from exc

    def _drain(self, stream):
        for _ in range(256):
            try:
                line = self.events.get_nowait()
            except queue.Empty:
                break
            stream.write(line)
        stream.flush()

    def _capture(self):
        began = time.monotonic()
        next_frame, last_source, last_fresh = began, None, began
        stream = None
        try:
            stream = (self.path / "control.jsonl").open("x", encoding="utf-8")
            self.observer.connect()
            while not self.stop_requested.is_set():
                self._drain(stream)
                now = time.monotonic()
                if now - began >= self.max_seconds:
                    raise RuntimeError("本段录制达到时长上限")
                if now < next_frame:
                    self.stop_requested.wait(min(0.02, next_frame - now))
                    continue
                observation, images = self.observer.read()
                if (
                    observation.get("metadata", {}).get("source") != "physical"
                    or set(images) != {"front", "left_wrist", "right_wrist"}
                    or observation.get("errors")
                ):
                    raise RuntimeError("相机/机器人观测不完整: " + str(observation.get("errors")))
                source = observation.get("source_timestamp_ns")
                if type(source) is not int or (last_source is not None and source < last_source):
                    raise RuntimeError("观测时间戳缺失或倒退")
                if source == last_source:
                    if time.monotonic() - last_fresh > 1:
                        raise RuntimeError("观测停更超过1秒")
                    next_frame = time.monotonic() + 1 / self.fps
                    continue
                if shutil.disk_usage(self.path).free < 512 * 1024 * 1024:
                    raise RuntimeError("录制磁盘剩余不足512MB")
                last_source = source
                last_fresh = time.monotonic()
                self.recorder.append(observation=observation, action=None, images=images)
                self.frames += 1
                self.ready.set()
                next_frame = time.monotonic() + 1 / self.fps
        except Exception as exc:  # noqa: BLE001 - the controller polls and fails closed
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            try:
                if stream is not None:
                    self._drain(stream)
                    stream.close()
                self.result = self.recorder.finish(
                    success=True if self.stop_requested.is_set() and self.error is None else None,
                    reason="operator_completed_reference_capture" if self.error is None else "disk_error",
                    interrupted=self.error is not None,
                )
            except Exception as exc:  # noqa: BLE001 - retain partial capture, never claim completion
                self.error = f"{self.error or ''}; finalization: {type(exc).__name__}: {exc}"

    def close(self):
        self.stop_requested.set()
        # Control is already released; allow USB storage to finish the episode scan.
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("录制仍在收尾，保留目录，不报告完成")
        self.check()
        return self.result


def export_videos(episode):
    """Make convenient constant-rate MP4 previews; original timestamps stay in JSONL."""
    import cv2

    episode = Path(episode)
    metadata = json.loads((episode / "metadata.json").read_text())
    writers, counts = {}, {}
    try:
        with (episode / "frames.jsonl").open() as frames:
            for line in frames:
                frame = json.loads(line)
                for role, relative in frame["images"].items():
                    picture = cv2.imread(str(episode / relative))
                    if picture is None:
                        raise RuntimeError(f"cannot decode saved image {relative}")
                    if role not in writers:
                        output = episode / f"{role}.mp4"
                        if output.exists():
                            raise FileExistsError(output)
                        height, width = picture.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output), cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], (width, height)
                        )
                        if not writer.isOpened():
                            raise RuntimeError("MP4 encoder unavailable; original JPEGs are retained")
                        writers[role], counts[role] = writer, 0
                    writers[role].write(picture)
                    counts[role] += 1
    finally:
        for writer in writers.values():
            writer.release()
    return {
        "episode": str(episode),
        "preview_frames": counts,
        "note": "constant-rate previews; use frames.jsonl for original timestamps",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export saved demo JPEGs to three MP4 previews")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    for episode in sorted(args.run_directory.glob("episode-*")):
        print(json.dumps(export_videos(episode), ensure_ascii=False), flush=True)
