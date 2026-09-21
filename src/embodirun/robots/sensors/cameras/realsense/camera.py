"""Persistent color-only RealSense camera source."""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..camera import CameraFrame


class RealSenseCameraError(RuntimeError):
    """A configured RealSense camera could not produce a color frame."""


@dataclass(frozen=True, slots=True)
class RealSenseCameraConfig:
    name: str
    serial: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    jpeg_quality: int = 90
    frame_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        for name in ("name", "serial"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"camera {name} must not be empty")
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if (
            isinstance(self.frame_timeout_s, bool)
            or not isinstance(self.frame_timeout_s, (int, float))
            or not math.isfinite(self.frame_timeout_s)
            or self.frame_timeout_s <= 0
        ):
            raise ValueError("frame_timeout_s must be a positive number")


class RealSenseCameraSource:
    """Open each RealSense pipeline once and reuse it for every capture."""

    def __init__(
        self,
        cameras: Sequence[RealSenseCameraConfig],
        *,
        rs_module: Any | None = None,
        jpeg_encoder: Callable[[bytes, int, int, int], bytes] | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._cameras = tuple(cameras)
        if not self._cameras:
            raise RealSenseCameraError("at least one RealSense camera is required")
        if rs_module is None:
            try:
                rs_module = importlib.import_module("pyrealsense2")
            except ImportError as error:
                raise RealSenseCameraError(
                    "RealSense capture requires pyrealsense2 in the robot environment"
                ) from error
        self._rs = rs_module
        self._jpeg_encoder = jpeg_encoder or _encode_rgb_as_jpeg
        # Freshness is evaluated in the owner-local monotonic domain; this
        # timestamp must not be subtracted against another node's clock.
        self._clock_ns = clock_ns or time.monotonic_ns
        self._pipelines: list[Any] = []
        self._pipeline_profiles: list[Any | None] = []
        try:
            for camera in self._cameras:
                pipeline = rs_module.pipeline()
                config = rs_module.config()
                config.enable_device(camera.serial)
                config.enable_stream(
                    rs_module.stream.color,
                    camera.width,
                    camera.height,
                    rs_module.format.rgb8,
                    camera.fps,
                )
                self._pipelines.append(pipeline)
                self._pipeline_profiles.append(pipeline.start(config))
        except Exception as error:
            self.close()
            if isinstance(error, RealSenseCameraError):
                raise
            raise RealSenseCameraError(f"could not start RealSense color pipeline: {error}") from error

    def capture(self) -> tuple[CameraFrame, ...]:
        images: list[CameraFrame] = []
        for index, (camera, pipeline) in enumerate(zip(self._cameras, self._pipelines)):
            try:
                frames = pipeline.wait_for_frames(max(1, int(camera.frame_timeout_s * 1000)))
                color = frames.get_color_frame()
            except RuntimeError as error:
                raise RealSenseCameraError(f"camera {camera.name!r} frame capture failed: {error}") from error
            if not color:
                raise RealSenseCameraError(f"camera {camera.name!r} returned no color frame")
            # RealSense's Python API does not provide a portable exposure
            # timestamp for this generic contract.  Capture the host read
            # boundary before copying/encoding the RGB payload instead of
            # relabelling the later encoded time as capture time.
            captured_timestamp_ns = self._now_ns()
            encoded = self._jpeg_encoder(
                bytes(color.get_data()),
                camera.width,
                camera.height,
                camera.jpeg_quality,
            )
            images.append(
                CameraFrame(
                    camera.name,
                    "image/jpeg",
                    encoded,
                    captured_timestamp_ns=captured_timestamp_ns,
                    received_timestamp_ns=self._now_ns(),
                    clock_domain="host_monotonic_ns",
                    profile=self._actual_profile(self._pipeline_profiles[index], color),
                )
            )
        return tuple(images)

    def _now_ns(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RealSenseCameraError("camera clock must return a non-negative integer nanosecond value")
        return value

    def _actual_profile(self, pipeline_profile: Any, color: Any) -> dict[str, object] | None:
        """Return only values reported by the SDK's active color profile."""

        stream_profile = None
        getter = getattr(pipeline_profile, "get_stream", None)
        if callable(getter):
            try:
                stream = getattr(self._rs, "stream", None)
                color_stream = getattr(stream, "color", None)
                stream_profile = getter(color_stream) if color_stream is not None else None
            except Exception:
                stream_profile = None
        if stream_profile is None:
            color_get_profile = getattr(color, "get_profile", None)
            if callable(color_get_profile):
                try:
                    stream_profile = color_get_profile()
                except Exception:
                    stream_profile = None
        video_profile = stream_profile
        as_video = getattr(stream_profile, "as_video_stream_profile", None)
        if callable(as_video):
            try:
                video_profile = as_video()
            except Exception:
                video_profile = stream_profile
        values: dict[str, object] = {}
        for name in ("width", "height", "fps"):
            method = getattr(video_profile, name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)) or float(value) <= 0:
                continue
            values[name] = int(value) if name != "fps" else float(value)
        return values or None

    def close(self) -> None:
        for pipeline in reversed(self._pipelines):
            with contextlib.suppress(Exception):
                pipeline.stop()
        self._pipelines.clear()
        self._pipeline_profiles.clear()


def _encode_rgb_as_jpeg(raw: bytes, width: int, height: int, quality: int) -> bytes:
    expected = width * height * 3
    if len(raw) != expected:
        raise RealSenseCameraError(f"unexpected RGB8 frame size: got {len(raw)}, expected {expected}")
    try:
        from PIL import Image
    except ImportError as error:
        raise RealSenseCameraError("RealSense JPEG encoding requires Pillow in the robot environment") from error
    image = Image.frombytes("RGB", (width, height), raw)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=quality)
    return encoded.getvalue()


__all__ = [
    "RealSenseCameraConfig",
    "RealSenseCameraError",
    "RealSenseCameraSource",
]
