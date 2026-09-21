"""Camera-only observation publisher for experiments without motor access.

The supplied state is a fixed fixture, not a robot measurement. This module
never constructs a robot adapter and exposes no action or robot-control API.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import signal
import threading
import time
from contextlib import nullcontext, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def video_device(path: str) -> str:
    """Resolve aliases and reject every device outside V4L2 before opening."""
    resolved = Path(path).resolve(strict=True)
    if not re.fullmatch(r"/dev/video[0-9]+", str(resolved)):
        raise ValueError("camera-only capture accepts /dev/video* devices only")
    if not (Path("/sys/class/video4linux") / resolved.name).exists():
        raise ValueError("device is not registered with video4linux")
    return str(resolved)


def observation_packet(frames, *, index, state, instruction, started, finished, binary=False):
    """Keep each response self-contained to avoid mixed camera generations."""
    return {
        "schema": "embodirun.camera-only.v1",
        "index": index,
        "state": list(state),
        "state_source": "fixed-fixture-no-robot-read",
        "instruction": instruction,
        "capture_started_monotonic_s": started,
        "capture_finished_monotonic_s": finished,
        "capture_unix_s": time.time(),
        "images": {
            frame.name: {
                "mime_type": frame.mime_type,
                **(
                    {
                        "width": frame.width,
                        "height": frame.height,
                        "pixel_format": frame.pixel_format,
                        "row_stride_bytes": frame.width * 3,
                    }
                    if frame.mime_type == "application/x-embodirun-raw-image"
                    else {}
                ),
                **({"data": frame.data} if binary else {"base64": base64.b64encode(frame.data).decode("ascii")}),
                "sha256": hashlib.sha256(frame.data).hexdigest(),
            }
            for frame in frames
        },
    }


class ObservationStore:
    """Bounded recorded observations plus the most recent camera frame."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.recorded = []
        self.status = {"state": "starting", "motor_access": False}

    def get(self, path):
        with self.lock:
            if path == "/health":
                return self.status.copy()
            if path == "/latest":
                if self.latest is None:
                    raise LookupError("camera has not produced a frame")
                if self.status["state"] != "running":
                    raise LookupError("camera is not running")
                return self.latest
            if re.fullmatch(r"/frame/[0-9]+", path):
                if not self.recorded:
                    raise LookupError("recording is not ready")
                return self.recorded[int(path.rsplit("/", 1)[1]) % len(self.recorded)]
        raise KeyError(path)


def make_handler(store, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0] not in allowed_hosts:
                self.send_error(403)
                return
            try:
                value = store.get(urlsplit(self.path).path)
            except KeyError:
                self.send_error(404)
                return
            except LookupError:
                self.send_error(503)
                return
            if "images" in value:
                value = {
                    **value,
                    "images": {
                        name: (
                            {
                                **{k: v for k, v in frame.items() if k != "data"},
                                "base64": base64.b64encode(frame["data"]).decode("ascii"),
                            }
                            if "data" in frame
                            else frame
                        )
                        for name, frame in value["images"].items()
                    },
                }
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return Handler


def run(args):
    # Resolve all devices before constructing even the camera source.
    devices = [(name, video_device(path)) for name, path in args.camera]
    raw = getattr(args, "frame_format", "jpeg") == "raw-bgr8"
    backend = getattr(args, "capture_backend", "opencv-v4l2")
    input_format = getattr(args, "input_format", None)
    output_width = getattr(args, "output_width", None)
    output_height = getattr(args, "output_height", None)
    if backend not in {"opencv-v4l2", "gstreamer-cpu", "gstreamer-jetson"}:
        raise ValueError("unknown camera capture backend")
    if backend.startswith("gstreamer") and not raw:
        raise ValueError("GStreamer capture requires --frame-format raw-bgr8")
    if (output_width is None) != (output_height is None):
        raise ValueError("set both output dimensions or neither")
    if backend == "opencv-v4l2" and output_width is not None:
        raise ValueError("capture-side output resizing requires a GStreamer backend")
    output_width = args.width if output_width is None else output_width
    output_height = args.height if output_height is None else output_height
    if any(type(n) is not int or not 0 < n <= 16384 for n in (output_width, output_height)):
        raise ValueError("output dimensions must be integers in 1..16384")
    args.output.mkdir(parents=True, exist_ok=False)
    store = ObservationStore()
    shared = None
    if getattr(args, "shm_path", None) is not None:
        from .camera_shm import SharedCameraStore

        shared = SharedCameraStore(
            args.shm_path,
            create=True,
            record_capacity=args.record_count,
            slot_bytes=max(
                2 * 1024**2,
                output_width * output_height * 3 * len(devices) + 65536 if raw else 0,
            ),
        )
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    server = None
    try:
        server = ThreadingHTTPServer((args.bind, args.port), make_handler(store, set(args.allow_host)))
        server.daemon_threads = True
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
    except BaseException:
        if server is not None:
            server.server_close()
        if shared is not None:
            shared.close()
        raise
    source = None
    started = time.monotonic()
    frames = recorded = 0
    total_capture_s = 0.0
    next_record = started
    last_report = started
    metadata = {
        "pid": os.getpid(),
        "devices": dict(devices),
        "width": args.width,
        "height": args.height,
        "fps_target": args.fps,
        "record_count_target": args.record_count,
        "record_hz": args.record_hz,
        "bind": args.bind,
        "port": args.port,
        "state": args.state,
        "motor_access": False,
        "frame_format": "raw-bgr8" if raw else "jpeg",
        "acquisition_backend": backend,
        "input_format_requested": input_format,
        "output_width": output_width,
        "output_height": output_height,
        "camera_driver_decode_may_remain": backend == "opencv-v4l2",
        "stage_measurement": getattr(args, "measure_stages", False),
        "shm_allocated_bytes": shared.allocated_bytes if shared is not None else 0,
        "shm_path": str(args.shm_path) if shared is not None else None,
    }
    try:
        (args.output / "process.json").write_text(json.dumps(metadata, indent=2) + "\n")
        setup_started = time.monotonic()
        if backend == "opencv-v4l2":
            from embodirun.robots.sensors.cameras.v4l2.camera import (
                V4L2CameraConfig,
                V4L2CameraSource,
            )

            source = V4L2CameraSource(
                tuple(
                    V4L2CameraConfig(
                        name,
                        device,
                        args.width,
                        args.height,
                        args.fps,
                        input_format=input_format,
                    )
                    for name, device in devices
                )
            )
        else:
            from embodirun.robots.sensors.cameras.gstreamer import (
                GStreamerCameraConfig,
                GStreamerCameraSource,
            )

            source = GStreamerCameraSource(
                tuple(
                    GStreamerCameraConfig(
                        name,
                        device,
                        args.width,
                        args.height,
                        args.fps,
                        input_format=input_format or "mjpeg",
                        conversion=backend.removeprefix("gstreamer-"),
                        output_width=output_width,
                        output_height=output_height,
                    )
                    for name, device in devices
                ),
                timeout_s=getattr(args, "capture_timeout_s", 2),
                measure_stages=getattr(args, "measure_stages", False),
            )
        metadata.update(getattr(source, "metadata", {}))
        metadata["source_setup_s"] = time.monotonic() - setup_started
        (args.output / "process.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (
            (args.output / "observations.jsonl").open("x") as manifest,
            (args.output / "capture.jsonl").open("x") as metrics,
            (
                (args.output / "capture-stages.jsonl").open("x")
                if getattr(args, "measure_stages", False)
                else nullcontext()
            ) as stages,
        ):
            while not stopped.is_set() and time.monotonic() - started < args.duration_s:
                before = time.monotonic()
                before_cpu = time.process_time()
                captured = source.capture_raw() if raw else source.capture()
                after = time.monotonic()
                after_cpu = time.process_time()
                packet = observation_packet(
                    captured,
                    index=frames,
                    state=args.state,
                    instruction=args.instruction,
                    started=before,
                    finished=after,
                    binary=shared is not None,
                )
                acquisition = getattr(source, "last_metrics", {})
                if acquisition:
                    packet["acquisition"] = {"backend": backend, "frames": acquisition}
                packet_finished = time.monotonic()
                frames += 1
                total_capture_s += after - before
                with store.lock:
                    store.latest = packet
                should_record = recorded < args.record_count and after >= next_record
                if should_record:
                    paths = {}
                    for frame in captured:
                        extension = frame.pixel_format if raw else "jpg"
                        name = f"{recorded:05d}-{frame.name}.{extension}"
                        (args.output / name).write_bytes(frame.data)
                        paths[frame.name] = name
                    row = {k: v for k, v in packet.items() if k != "images"}
                    row.update(images=paths, source="live-camera-recording")
                    row["image_metadata"] = {
                        name: {k: v for k, v in image.items() if k not in {"data", "base64"}}
                        for name, image in packet["images"].items()
                    }
                    manifest.write(json.dumps(row, allow_nan=False) + "\n")
                    manifest.flush()
                    with store.lock:
                        store.recorded.append(packet)
                    recorded += 1
                    next_record = after + 1 / args.record_hz
                record_finished = time.monotonic()
                elapsed = after - started
                status = {
                    **metadata,
                    "state": "running",
                    "frames": frames,
                    "recorded": recorded,
                    "elapsed_s": elapsed,
                    "effective_fps": frames / elapsed,
                    "capture_mean_ms": total_capture_s / frames * 1000,
                    "last_capture_ms": (after - before) * 1000,
                    "process_cpu_s": time.process_time(),
                    "loadavg": list(os.getloadavg()),
                    "latest_monotonic_s": after,
                    "last_acquisition": acquisition,
                }
                with store.lock:
                    store.status = status
                publish_started = time.monotonic()
                if shared is not None:
                    shared.publish(packet, status, record=should_record)
                published = time.monotonic()
                if stages is not None:
                    stages.write(
                        json.dumps(
                            {
                                "index": packet["index"],
                                "backend": backend,
                                "capture_s": after - before,
                                "process_cpu_during_capture_s": after_cpu - before_cpu,
                                "packet_hash_encode_s": packet_finished - after,
                                "record_io_s": record_finished - packet_finished,
                                "shm_publish_s": published - publish_started if shared is not None else None,
                                "work_before_stage_log_s": published - before,
                                "process_cpu_before_stage_log_s": time.process_time() - before_cpu,
                                "image_payload_bytes": sum(len(frame.data) for frame in captured),
                                "recorded": should_record,
                                "acquisition": acquisition,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                if after - last_report >= 1:
                    metrics.write(json.dumps(status, allow_nan=False) + "\n")
                    metrics.flush()
                    if stages is not None:
                        stages.flush()
                    last_report = after
                stopped.wait(max(0, 1 / args.fps - (time.monotonic() - before)))
    except BaseException as error:
        with store.lock:
            store.status = dict(
                store.status,
                failure={"type": type(error).__name__, "message": str(error)},
            )
        raise
    finally:
        if source is not None:
            source.close()
        if shared is not None:
            shared.close()
        with store.lock:
            store.status = dict(store.status, state="stopped")
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
        (args.output / "final.json").write_text(json.dumps(store.status, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", nargs=2, metavar=("NAME", "DEVICE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shm-path",
        type=Path,
        help="Optional new file under /dev/shm for local binary camera snapshots",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--capture-backend",
        choices=["opencv-v4l2", "gstreamer-cpu", "gstreamer-jetson"],
        default="opencv-v4l2",
    )
    parser.add_argument(
        "--input-format",
        choices=["mjpeg", "yuy2"],
        help="Explicit driver format for either backend; default is OpenCV auto / GStreamer MJPEG",
    )
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--capture-timeout-s", type=float, default=2)
    parser.add_argument("--measure-stages", action="store_true")
    parser.add_argument(
        "--frame-format",
        choices=["jpeg", "raw-bgr8"],
        default="jpeg",
        help="raw-bgr8 skips application JPEG encoding; use a same-host SHM consumer",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--record-count", type=int, default=120)
    parser.add_argument("--record-hz", type=float, default=2)
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29480)
    parser.add_argument("--allow-host", action="append", default=["127.0.0.1"])
    parser.add_argument("--state", type=float, nargs=6, default=[0, 0, 0, 0, 0, 50])
    parser.add_argument("--instruction", default="pick up the object")
    args = parser.parse_args()
    names = [name for name, _ in args.camera]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", n) for n in names):
        parser.error("camera names must be unique lowercase identifiers")
    if any(
        not math.isfinite(x) or x <= 0
        for x in (
            args.width,
            args.height,
            args.fps,
            args.record_hz,
            args.duration_s,
            args.capture_timeout_s,
        )
    ):
        parser.error("dimensions, rates and duration must be finite and positive")
    if args.record_count <= 0 or not all(math.isfinite(x) for x in args.state):
        parser.error("record count must be positive and fixture state finite")
    run(args)


if __name__ == "__main__":
    main()
