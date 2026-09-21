"""Aligned single-pipeline RealSense RGB-D backend."""

from __future__ import annotations

import importlib
import threading
import time
from typing import Any, Callable, Optional  # noqa: UP035

from .depth_store import DepthStore
from .encoding import analyze_z16_depth, encode_bgr_as_jpeg, encode_z16_as_png
from .frame_store import FrameStore
from .realsense_pipeline import configured_pipeline
from .realsense_validation import (
    realsense_device_name,
    validated_depth_scale,
    validated_frame_bytes,
    validated_source_identity,
    validated_video_profile,
)
from .types import CameraStreamError, DepthAnalysis


class RealSenseWorker:
    """Capture an atomic RGB/depth pair aligned to the color stream.

    A single ``pyrealsense2.pipeline`` produces both streams. Depth is passed
    through ``rs.align(rs.stream.color)`` before either slot is published. The
    worker validates negotiated profiles, stride, frame size, depth scale, and
    monotonically increasing source identity on every published pair.
    """

    def __init__(
        self,
        frame_store: FrameStore,
        depth_store: DepthStore,
        width: int,
        height: int,
        fps: int,
        expected_depth_scale: float,
        jpeg_quality: int,
        jpeg_fps: float,
        frame_timeout: float,
        retry_delay: float,
        serial: Optional[str] = None,  # noqa: UP045
        roi_width_ratio: float = 0.5,
        roi_height_ratio: float = 0.5,
        min_valid_fraction: float = 0.1,
        max_depth_m: float = 10.0,
        encoder: Callable[[bytes, int, int, int], bytes] = encode_bgr_as_jpeg,
        depth_encoder: Callable[[bytes, int, int], bytes] = encode_z16_as_png,
        analyzer: Callable[..., DepthAnalysis] = analyze_z16_depth,
        rs_module: Any = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.frame_store = frame_store
        self.depth_store = depth_store
        self.width = width
        self.height = height
        self.fps = fps
        self.expected_depth_scale = expected_depth_scale
        self.jpeg_quality = jpeg_quality
        self.jpeg_fps = jpeg_fps
        self.frame_timeout = frame_timeout
        self.retry_delay = retry_delay
        self.serial = serial
        self.roi_width_ratio = roi_width_ratio
        self.roi_height_ratio = roi_height_ratio
        self.min_valid_fraction = min_valid_fraction
        self.max_depth_m = max_depth_m
        self.encoder = encoder
        self.depth_encoder = depth_encoder
        self.analyzer = analyzer
        self._rs_module = rs_module
        self.clock = clock
        self.wall_clock = wall_clock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None  # noqa: UP045
        self._pipeline_lock = threading.Lock()
        self._pipeline: Any = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RealSense worker has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="go2-realsense-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._pipeline_lock:
            pipeline = self._pipeline
        self._stop_pipeline(pipeline)

    def join(self, timeout: Optional[float] = None) -> None:  # noqa: UP045
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @staticmethod
    def _stop_pipeline(pipeline: Any) -> None:
        if pipeline is None:
            return
        try:
            pipeline.stop()
        except (OSError, RuntimeError, ValueError):
            return

    def _load_rs(self) -> Any:
        if self._rs_module is not None:
            return self._rs_module
        try:
            self._rs_module = importlib.import_module("pyrealsense2")
        except ImportError as error:
            raise CameraStreamError("pyrealsense2 is required for the realsense backend") from error
        return self._rs_module

    def _run_once(self, rs: Any) -> None:
        pipeline, config = configured_pipeline(
            rs,
            serial=self.serial,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        with self._pipeline_lock:
            self._pipeline = pipeline

        actual_depth_scale: Optional[float] = None  # noqa: UP045
        try:
            if self._stop.is_set():
                return
            pipeline_profile = pipeline.start(config)
            color_profile = validated_video_profile(
                pipeline_profile,
                rs.stream.color,
                rs.format.bgr8,
                "BGR8",
                3,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )
            depth_profile = validated_video_profile(
                pipeline_profile,
                rs.stream.depth,
                rs.format.z16,
                "Z16",
                2,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )
            actual_depth_scale = validated_depth_scale(
                pipeline_profile,
                self.expected_depth_scale,
            )

            align = rs.align(rs.stream.color)
            device_name = realsense_device_name(rs, pipeline_profile, self.serial)
            self.frame_store.set_capture_running(True)
            self.depth_store.set_capture_running(True)
            self.frame_store.set_error(None)
            self.depth_store.set_error(None)
            frame_interval = 1.0 / self.jpeg_fps
            next_publish_at = 0.0
            pair_sequence = max(
                int(self.frame_store.health(float("inf"))["frame_sequence"]),
                int(self.depth_store.status(float("inf"))["frame_sequence"]),
            )
            latest_frame = self.frame_store.latest()
            last_captured_at_unix = None if latest_frame is None else latest_frame.captured_at_unix
            last_source_frame_number: Optional[int] = None  # noqa: UP045
            last_source_timestamp_ms: Optional[float] = None  # noqa: UP045

            while not self._stop.is_set():
                try:
                    frames = pipeline.wait_for_frames(max(1, int(self.frame_timeout * 1000)))
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                except Exception as error:
                    raise CameraStreamError(f"RealSense frame watchdog/alignment failed: {error}") from error

                now = self.clock()
                if now < next_publish_at:
                    continue
                source_frame_number, source_timestamp_ms = validated_source_identity(
                    color_frame,
                    last_source_frame_number,
                    last_source_timestamp_ms,
                )

                color_raw = validated_frame_bytes(
                    color_frame,
                    color_profile,
                    "color",
                )
                depth_raw = validated_frame_bytes(
                    depth_frame,
                    depth_profile,
                    "aligned depth",
                )
                jpeg = self.encoder(
                    color_raw,
                    self.width,
                    self.height,
                    self.jpeg_quality,
                )
                depth_png = self.depth_encoder(depth_raw, self.width, self.height)
                analysis = self.analyzer(
                    depth_raw,
                    self.width,
                    self.height,
                    actual_depth_scale,
                    self.roi_width_ratio,
                    self.roi_height_ratio,
                    self.min_valid_fraction,
                    self.max_depth_m,
                )

                # Profiles only become visible after an aligned frameset has
                # also passed the per-frame stride and size checks.
                self.frame_store.set_source_profile(device_name, color_profile)
                self.depth_store.set_source_profile(device_name, depth_profile)
                self.depth_store.set_registration(
                    verified=True,
                    aligned=True,
                    depth_scale_m=actual_depth_scale,
                )
                pair_sequence += 1
                captured_at_monotonic = self.clock()
                captured_at_unix = self.wall_clock()
                if last_captured_at_unix is not None and captured_at_unix <= last_captured_at_unix:
                    captured_at_unix = last_captured_at_unix + 0.000001
                invalid_error = None
                if analysis.center_distance_m is None or analysis.minimum_distance_m is None:
                    invalid_error = "insufficient valid aligned depth pixels"

                # Depth-first/RGB-second creates at most a transient unavailable
                # window. Readers require identical immutable sequence slots.
                self.depth_store.update(
                    analysis.center_distance_m,
                    analysis.minimum_distance_m,
                    analysis.valid_fraction,
                    error=invalid_error,
                    sequence=pair_sequence,
                    captured_at_unix=captured_at_unix,
                    captured_at_monotonic=captured_at_monotonic,
                    source_timestamp_ms=source_timestamp_ms,
                    source_frame_number=source_frame_number,
                    depth_png=depth_png,
                )
                self.frame_store.update(
                    jpeg,
                    sequence=pair_sequence,
                    captured_at_unix=captured_at_unix,
                    captured_at_monotonic=captured_at_monotonic,
                    source_timestamp_ms=source_timestamp_ms,
                    source_frame_number=source_frame_number,
                )
                last_source_frame_number = source_frame_number
                last_source_timestamp_ms = source_timestamp_ms
                last_captured_at_unix = captured_at_unix
                next_publish_at = now + frame_interval
        finally:
            self.frame_store.set_capture_running(False)
            self.depth_store.set_capture_running(False)
            self.depth_store.set_registration(
                verified=False,
                aligned=False,
                depth_scale_m=actual_depth_scale,
            )
            self._stop_pipeline(pipeline)
            with self._pipeline_lock:
                if self._pipeline is pipeline:
                    self._pipeline = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_once(self._load_rs())
            except Exception as error:  # noqa: BLE001 - hardware loop must retry
                if not self._stop.is_set():
                    message = f"{type(error).__name__}: {error}"
                    self.frame_store.set_error(message)
                    self.depth_store.set_error(message)
            self._stop.wait(self.retry_delay)
        self.frame_store.set_capture_running(False)
        self.depth_store.set_capture_running(False)


__all__ = ["RealSenseWorker"]
