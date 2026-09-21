"""Retrying workers for split diagnostic V4L2 streams."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional  # noqa: UP035

from .depth_store import DepthStore
from .encoding import analyze_z16_depth, encode_yuyv_as_jpeg
from .frame_store import FrameStore
from .types import DepthAnalysis
from .v4l2 import V4L2YUYVSource, V4L2Z16Source


class CameraWorker:
    """Capture and encode RGB frames from one persistent source."""

    def __init__(
        self,
        store: FrameStore,
        source_factory: Callable[[], V4L2YUYVSource],
        width: int,
        height: int,
        jpeg_quality: int,
        jpeg_fps: float,
        retry_delay: float,
        encoder: Callable[[bytes, int, int, int], bytes] = encode_yuyv_as_jpeg,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.source_factory = source_factory
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.jpeg_fps = jpeg_fps
        self.retry_delay = retry_delay
        self.encoder = encoder
        self.clock = clock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None  # noqa: UP045
        self._source_lock = threading.Lock()
        self._source: Optional[V4L2YUYVSource] = None  # noqa: UP045

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("camera worker has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="go2-camera-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._source_lock:
            source = self._source
        if source is not None:
            source.close()

    def join(self, timeout: Optional[float] = None) -> None:  # noqa: UP045
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        frame_interval = 1.0 / self.jpeg_fps
        while not self._stop.is_set():
            source = self.source_factory()
            with self._source_lock:
                self._source = source
            try:
                source.open()
                if source.actual_format is not None:
                    self.store.set_source_profile(source.device, source.actual_format)
                self.store.set_capture_running(True)
                self.store.set_error(None)
                next_encode_at = 0.0
                while not self._stop.is_set():
                    raw = source.read()
                    now = self.clock()
                    if now < next_encode_at:
                        continue
                    jpeg = self.encoder(raw, self.width, self.height, self.jpeg_quality)
                    self.store.update(jpeg)
                    next_encode_at = now + frame_interval
            except Exception as error:  # noqa: BLE001 - camera loop must recover
                if not self._stop.is_set():
                    self.store.set_error(f"{type(error).__name__}: {error}")
            finally:
                self.store.set_capture_running(False)
                source.close()
                with self._source_lock:
                    if self._source is source:
                        self._source = None

            self._stop.wait(self.retry_delay)
        self.store.set_capture_running(False)


class DepthWorker:
    """Capture diagnostic Z16 frames; invalid samples fail closed."""

    def __init__(
        self,
        store: DepthStore,
        source_factory: Callable[[], V4L2Z16Source],
        width: int,
        height: int,
        depth_scale: float,
        sample_fps: float,
        retry_delay: float,
        roi_width_ratio: float = 0.5,
        roi_height_ratio: float = 0.5,
        min_valid_fraction: float = 0.1,
        max_depth_m: float = 10.0,
        analyzer: Callable[..., DepthAnalysis] = analyze_z16_depth,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.source_factory = source_factory
        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        self.sample_fps = sample_fps
        self.retry_delay = retry_delay
        self.roi_width_ratio = roi_width_ratio
        self.roi_height_ratio = roi_height_ratio
        self.min_valid_fraction = min_valid_fraction
        self.max_depth_m = max_depth_m
        self.analyzer = analyzer
        self.clock = clock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None  # noqa: UP045
        self._source_lock = threading.Lock()
        self._source: Optional[V4L2Z16Source] = None  # noqa: UP045

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("depth worker has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="go2-depth-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._source_lock:
            source = self._source
        if source is not None:
            source.close()

    def join(self, timeout: Optional[float] = None) -> None:  # noqa: UP045
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        sample_interval = 1.0 / self.sample_fps
        while not self._stop.is_set():
            source = self.source_factory()
            with self._source_lock:
                self._source = source
            try:
                source.open()
                if source.actual_format is not None:
                    self.store.set_source_profile(source.device, source.actual_format)
                self.store.set_capture_running(True)
                self.store.set_error(None)
                next_sample_at = 0.0
                while not self._stop.is_set():
                    raw = source.read()
                    now = self.clock()
                    if now < next_sample_at:
                        continue
                    analysis = self.analyzer(
                        raw,
                        self.width,
                        self.height,
                        self.depth_scale,
                        self.roi_width_ratio,
                        self.roi_height_ratio,
                        self.min_valid_fraction,
                        self.max_depth_m,
                    )
                    error = None
                    if analysis.center_distance_m is None or analysis.minimum_distance_m is None:
                        error = "insufficient valid depth pixels in center ROI"
                    self.store.update(
                        analysis.center_distance_m,
                        analysis.minimum_distance_m,
                        analysis.valid_fraction,
                        error=error,
                    )
                    next_sample_at = now + sample_interval
            except Exception as error:  # noqa: BLE001 - camera loop must recover
                if not self._stop.is_set():
                    self.store.set_error(f"{type(error).__name__}: {error}")
            finally:
                self.store.set_capture_running(False)
                source.close()
                with self._source_lock:
                    if self._source is source:
                        self._source = None

            self._stop.wait(self.retry_delay)
        self.store.set_capture_running(False)


__all__ = ["CameraWorker", "DepthWorker"]
