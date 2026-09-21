"""Record follower episodes locally, without putting camera bytes on the network.

One episode is a directory holding a ``frames.jsonl`` control log plus one folder
per camera role. Each control row carries the received target, the measured
position, the target actually sent and the newest frame path per camera, so a
reviewer can tell an operator error apart from a tracking error.

Cameras are read by this process, on the follower host, at ``camera_fps``. Only
the resulting JPEG paths enter the log, which keeps the wired link free for the
control stream.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_LOG = logging.getLogger(__name__)

DEFAULT_ROLES = ("main", "wrist")
JPEG_QUALITY = 90
STOP_JOIN_TIMEOUT_S = 3.0


def role_names(cameras: list[int], roles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return one directory name per camera, defaulting to main/wrist."""

    if roles:
        return tuple(roles)
    return tuple(
        DEFAULT_ROLES[slot] if slot < len(DEFAULT_ROLES) else f"camera_{index}" for slot, index in enumerate(cameras)
    )


class EpisodeRecorder:
    """Write one episode at a time; ``start``/``stop`` are safe to repeat."""

    def __init__(
        self,
        root: str | Path,
        cameras: list[int],
        camera_fps: float,
        roles: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.root = Path(root)
        self.cameras = list(cameras)
        self.roles = role_names(self.cameras, roles)
        self.camera_fps = camera_fps
        self._lock = threading.Lock()
        self._active = False
        self._episode: Path | None = None
        self._frames: TextIO | None = None
        self._latest: dict[int, str | None] = dict.fromkeys(self.cameras)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def episode(self) -> Path | None:
        with self._lock:
            return self._episode

    def start(self) -> str:
        """Open a new episode and start the camera workers."""

        if any(thread.is_alive() for thread in self._threads) and not self.active:
            raise RuntimeError("previous camera threads have not stopped")
        with self._lock:
            if self._active and self._episode is not None:
                return str(self._episode)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self._episode = self.root / f"episode-{stamp}"
            self._episode.mkdir(parents=True, exist_ok=False)
            for role in self.roles:
                (self._episode / role).mkdir()
            self._frames = (self._episode / "frames.jsonl").open("w", encoding="utf-8")
            self._frames.write(json.dumps({"event": "start", "time": time.time()}) + "\n")
            self._frames.flush()
            self._latest = dict.fromkeys(self.cameras)
            self._stop.clear()
            self._active = True
            self._threads = []
            for slot, index in enumerate(self.cameras):
                thread = threading.Thread(
                    target=self._camera_loop,
                    args=(slot, index),
                    daemon=True,
                    name=f"camera-{index}",
                )
                thread.start()
                self._threads.append(thread)
            return str(self._episode)

    def stop(self) -> str | None:
        """Close the current episode and wait for the camera workers."""

        with self._lock:
            if not self._active:
                return None
            episode = str(self._episode)
            self._active = False
            self._stop.set()
            frames = self._frames
            self._frames = None
            if frames is not None:
                frames.write(json.dumps({"event": "stop", "time": time.time()}) + "\n")
                frames.close()
            self._episode = None
        for thread in self._threads:
            thread.join(timeout=STOP_JOIN_TIMEOUT_S)
        return episode

    def record(
        self,
        sequence: int,
        action: dict[str, float],
        present: dict[str, float],
        sent: dict[str, float],
    ) -> None:
        """Append one control row, or do nothing when no episode is open."""

        with self._lock:
            if not self._active or self._frames is None:
                return
            row = {
                "time": time.time(),
                "sequence": sequence,
                "action": action,
                "present": present,
                "sent": sent,
                "cameras": dict(self._latest),
            }
            self._frames.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._frames.flush()

    def _camera_loop(self, slot: int, index: int) -> None:
        try:
            import cv2
        except ImportError:
            _LOG.warning("camera %d not recorded: opencv is not installed", index)
            return
        camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not camera.isOpened():
            print(f"CAMERA_ERROR index={index} open_failed", flush=True)
            return
        # Ask for on-camera compression before size and rate are negotiated. A
        # device without MJPEG keeps its own format, which is why the actual
        # FOURCC is reported rather than assumed.
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
        code = int(camera.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * n)) & 255) for n in range(4))
        print(
            f"CAMERA_FORMAT index={index} role={self.roles[slot]} format={fourcc} fps={camera.get(cv2.CAP_PROP_FPS)}",
            flush=True,
        )
        interval = 1.0 / max(self.camera_fps, 1.0)
        next_frame = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame = camera.read()
                if ok:
                    with self._lock:
                        episode, active = self._episode, self._active
                    if not active or episode is None:
                        break
                    role = self.roles[slot]
                    filename = f"{time.time_ns()}.jpg"
                    path = episode / role / filename
                    if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                        with self._lock:
                            self._latest[index] = f"{role}/{filename}"
                next_frame += interval
                time.sleep(max(0.0, next_frame - time.monotonic()))
        finally:
            camera.release()

    def mark(self, episode: str | Path, decision: str, **extra: Any) -> Path:
        """Write ``decision.json`` so a kept episode is told apart from a discard."""

        payload = {"decision": decision, "time": time.time(), **extra}
        path = Path(episode) / "decision.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
