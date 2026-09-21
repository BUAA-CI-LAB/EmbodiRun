"""Command-line entrypoint for the dog-side Go2 camera service."""

from __future__ import annotations

import json
import signal
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence  # noqa: UP035

from .arguments import parse_args
from .depth_store import DepthStore
from .frame_store import FrameStore
from .realsense import RealSenseWorker
from .server import CameraHTTPServer
from .settings import CameraServiceConfig
from .v4l2 import V4L2YUYVSource, V4L2Z16Source
from .workers import CameraWorker, DepthWorker


class CaptureWorker(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def join(self, timeout: Optional[float] = None) -> None: ...  # noqa: UP045


@dataclass(frozen=True)
class CameraServiceRuntime:
    server: CameraHTTPServer
    rgb_worker: CaptureWorker
    depth_worker: Optional[CaptureWorker]  # noqa: UP045
    device_name: str


def build_runtime(config: CameraServiceConfig) -> CameraServiceRuntime:
    """Construct stores, capture workers, and the HTTP server without starting them."""

    config.validate()
    frame_store = FrameStore(device=config.initial_device)
    depth_store: Optional[DepthStore] = None  # noqa: UP045
    depth_worker: Optional[CaptureWorker] = None  # noqa: UP045

    if config.backend == "realsense":
        assert config.depth_scale is not None
        depth_store = DepthStore(
            config.width,
            config.height,
            device=config.initial_device,
            backend="realsense",
        )
        rgb_worker: CaptureWorker = RealSenseWorker(
            frame_store=frame_store,
            depth_store=depth_store,
            width=config.width,
            height=config.height,
            fps=config.fps,
            expected_depth_scale=config.depth_scale,
            jpeg_quality=config.jpeg_quality,
            jpeg_fps=config.jpeg_fps,
            frame_timeout=config.frame_timeout,
            retry_delay=config.retry_delay,
            serial=config.serial,
            max_depth_m=config.max_depth_m,
        )
    else:

        def rgb_source_factory() -> V4L2YUYVSource:
            return V4L2YUYVSource(
                device=config.device,
                width=config.width,
                height=config.height,
                warmup_frames=config.warmup_frames,
                frame_timeout=config.frame_timeout,
                format_timeout=config.format_timeout,
            )

        rgb_worker = CameraWorker(
            store=frame_store,
            source_factory=rgb_source_factory,
            width=config.width,
            height=config.height,
            jpeg_quality=config.jpeg_quality,
            jpeg_fps=config.jpeg_fps,
            retry_delay=config.retry_delay,
        )
        if config.depth_calibrated:
            assert config.depth_scale is not None
            depth_store = DepthStore(
                config.depth_width,
                config.depth_height,
                device=config.depth_device,
                backend="v4l2",
            )
            depth_store.set_registration(False, False, config.depth_scale)

            def depth_source_factory() -> V4L2Z16Source:
                return V4L2Z16Source(
                    device=config.depth_device,
                    width=config.depth_width,
                    height=config.depth_height,
                    warmup_frames=config.depth_warmup_frames,
                    frame_timeout=config.frame_timeout,
                    format_timeout=config.format_timeout,
                )

            depth_worker = DepthWorker(
                store=depth_store,
                source_factory=depth_source_factory,
                width=config.depth_width,
                height=config.depth_height,
                depth_scale=config.depth_scale,
                sample_fps=config.depth_fps,
                retry_delay=config.retry_delay,
                max_depth_m=config.max_depth_m,
            )

    server = CameraHTTPServer(
        (config.host, config.port),
        store=frame_store,
        max_frame_age=config.max_frame_age,
        depth_store=depth_store,
        max_depth_age=config.max_depth_age,
        api_token=config.token,
        calibration_confirmed=config.depth_calibrated,
        rgb_depth_alignment_claimed=config.rgb_depth_alignment_claimed,
        backend=config.backend,
        depth_scale_m=config.depth_scale,
        depth_device=(config.initial_device if config.backend == "realsense" else config.depth_device),
        access_log=config.access_log,
    )
    return CameraServiceRuntime(
        server=server,
        rgb_worker=rgb_worker,
        depth_worker=depth_worker,
        device_name=config.initial_device,
    )


def _startup_payload(
    config: CameraServiceConfig,
    runtime: CameraServiceRuntime,
) -> Dict[str, object]:  # noqa: UP006
    host, port = runtime.server.server_address[:2]
    base_url = f"http://{host}:{port}"
    return {
        "backend": config.backend,
        "camera_device": runtime.device_name,
        "profile": config.profile,
        "authenticated": config.token is not None,
        "health_url": f"{base_url}/health",
        "frame_url": f"{base_url}/frame.jpg",
        "depth_url": f"{base_url}/depth.json",
        "depth_png_url": f"{base_url}/depth.png",
        "observation_url": f"{base_url}/observation.json",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045
    """Run the service until SIGINT/SIGTERM; hardware is opened by workers."""

    config = parse_args(argv)
    runtime = build_runtime(config)

    def request_exit(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, request_exit)
        signal.signal(signal.SIGINT, request_exit)

    rgb_started = False
    depth_started = False
    try:
        runtime.rgb_worker.start()
        rgb_started = True
        if runtime.depth_worker is not None:
            runtime.depth_worker.start()
            depth_started = True
        print(
            json.dumps(
                _startup_payload(config, runtime),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        runtime.server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.server.server_close()
        if rgb_started:
            runtime.rgb_worker.stop()
        if runtime.depth_worker is not None and depth_started:
            runtime.depth_worker.stop()
        if rgb_started:
            runtime.rgb_worker.join(timeout=3.0)
        if runtime.depth_worker is not None and depth_started:
            runtime.depth_worker.join(timeout=3.0)
    return 0


__all__ = ["CameraServiceRuntime", "build_runtime", "main"]
