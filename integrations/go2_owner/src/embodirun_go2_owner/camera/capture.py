"""Own camera capture workers and snapshots independently of HTTP serving."""

from __future__ import annotations

from typing import Optional

from .depth_store import DepthStore
from .frame_store import FrameStore
from .observation import CameraObservationService
from .realsense import RealSenseWorker
from .settings import CameraServiceConfig
from .v4l2 import V4L2YUYVSource, V4L2Z16Source
from .workers import CameraWorker, DepthWorker


class CameraCapture:
    """Camera resources with explicit startup/cleanup, without a listening socket."""

    def __init__(self, config, frame_store, depth_store, rgb_worker, depth_worker):
        self.frame_store, self.depth_store = frame_store, depth_store
        self.max_frame_age, self.max_depth_age = config.max_frame_age, config.max_depth_age
        self.calibration_confirmed = config.depth_calibrated
        self.rgb_depth_alignment_claimed = config.rgb_depth_alignment_claimed
        self.backend = config.backend
        self.registration_verified = False
        self.depth_scale_m = config.depth_scale
        self.depth_device = config.initial_device if config.backend == "realsense" else config.depth_device
        self.observations = CameraObservationService(self)
        self.workers = [rgb_worker] + ([depth_worker] if depth_worker is not None else [])
        self._started = []

    def start(self) -> None:
        """Start capture only after the owner has bound its HTTP socket."""
        for worker in self.workers:
            worker.start()
            self._started.append(worker)

    def close(self) -> None:
        """Stop all capture workers; report any worker that cannot be joined."""
        errors = []
        for worker in self._started:
            try:
                worker.stop()
            except Exception as error:
                errors.append(str(error))
        for worker in self._started:
            try:
                worker.join(timeout=3.0)
                if worker.is_alive():
                    errors.append("camera worker did not terminate")
            except Exception as error:
                errors.append(str(error))
        self._started.clear()
        if errors:
            raise RuntimeError("; ".join(errors))


def build_capture(config: CameraServiceConfig) -> CameraCapture:
    """Construct capture workers without importing SDKs or opening devices."""

    config.validate()
    frame_store = FrameStore(device=config.initial_device)
    depth_store: Optional[DepthStore] = None  # noqa: UP045
    depth_worker: Optional[object] = None  # noqa: UP045

    if config.backend == "realsense":
        assert config.depth_scale is not None
        depth_store = DepthStore(
            config.width,
            config.height,
            device=config.initial_device,
            backend="realsense",
        )
        rgb_worker = RealSenseWorker(
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

    return CameraCapture(config, frame_store, depth_store, rgb_worker, depth_worker)
