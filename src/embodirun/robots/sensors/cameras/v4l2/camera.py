"""V4L2 camera capture through OpenCV."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..camera import CameraFrame


def _camera_buffer_size() -> int:
    """Return the V4L2 driver buffer count to request.

    A single buffer makes the driver drop every other frame on the UVC cameras
    used for SO-101: measured on a Pi 4B front camera, ``CAP_PROP_BUFFERSIZE=1``
    caps delivery at 10.00 fps while 2 (or more) reaches 20.00 fps. Two buffers
    keep the configured rate and cost only one frame of freshness.

    Override with ``EMBODIRUN_CAMERA_BUFFERSIZE``; values below 1 fall back to
    the default.
    """

    raw = os.environ.get("EMBODIRUN_CAMERA_BUFFERSIZE", "2")
    try:
        value = int(raw)
    except ValueError:
        return 2
    return value if value >= 1 else 2


class CameraError(RuntimeError):
    """A configured camera could not produce a valid frame."""


@dataclass(frozen=True, slots=True)
class V4L2CameraConfig:
    name: str
    device: str
    width: int
    height: int
    fps: float
    input_format: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera name must not be empty")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("camera device must not be empty")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("camera fps must be a finite positive number")
        if self.input_format not in {None, "mjpeg", "yuy2"}:
            raise ValueError("camera input_format must be mjpeg or yuy2 when set")


class V4L2CameraSource:
    """Capture synchronized-enough JPEG frames from V4L2 cameras."""

    def __init__(
        self,
        cameras: Sequence[V4L2CameraConfig],
        *,
        cv2_module: Any | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise CameraError("V4L2 capture requires OpenCV in the robot environment") from error
        self._cv2 = cv2_module
        # Age/freshness uses the owner-local monotonic domain.  This value is
        # not comparable with another node's monotonic clock and is labelled
        # accordingly in the frame metadata.
        self._clock_ns = clock_ns or time.monotonic_ns
        self._cameras = tuple(cameras)
        if not self._cameras:
            raise CameraError("at least one V4L2 camera is required")
        self._captures: list[Any] = []
        self.metadata: dict[str, Any] = {"negotiated_cameras": {}}
        try:
            for camera in self._cameras:
                capture = cv2_module.VideoCapture(camera.device, cv2_module.CAP_V4L2)
                self._captures.append(capture)
                if not capture.isOpened():
                    raise CameraError(f"camera {camera.name!r} cannot open {camera.device!r}")
                if camera.input_format is not None:
                    fourcc = "MJPG" if camera.input_format == "mjpeg" else "YUYV"
                    if not capture.set(
                        cv2_module.CAP_PROP_FOURCC,
                        cv2_module.VideoWriter_fourcc(*fourcc),
                    ):
                        raise CameraError(f"camera {camera.name!r} rejected input format {fourcc}")
                capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, camera.width)
                capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, camera.height)
                capture.set(cv2_module.CAP_PROP_FPS, camera.fps)
                if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
                    capture.set(cv2_module.CAP_PROP_BUFFERSIZE, _camera_buffer_size())
            for _ in range(3):
                self._capture_frames()
            for camera, capture in zip(self._cameras, self._captures):
                if camera.input_format is None:
                    continue
                actual = int(capture.get(cv2_module.CAP_PROP_FOURCC))
                fourcc = "".join(chr((actual >> (8 * n)) & 255) for n in range(4))
                width = capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)
                fps = capture.get(cv2_module.CAP_PROP_FPS)
                expected = "MJPG" if camera.input_format == "mjpeg" else "YUYV"
                if (
                    fourcc != expected
                    or (width, height) != (camera.width, camera.height)
                    or not math.isfinite(fps)
                    or abs(fps - camera.fps) > 0.05
                ):
                    raise CameraError(
                        f"camera {camera.name!r} negotiated {fourcc} "
                        f"{width}x{height}@{fps}, expected {expected} "
                        f"{camera.width}x{camera.height}@{camera.fps}"
                    )
                self.metadata["negotiated_cameras"][camera.name] = {
                    "fourcc": fourcc,
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
        except BaseException:
            self.close()
            raise

    def capture(self) -> tuple[CameraFrame, ...]:
        frames = self._capture_frames_with_timestamps()
        images: list[CameraFrame] = []
        for camera, (frame, captured_timestamp_ns) in zip(self._cameras, frames):
            encoded, jpeg = self._cv2.imencode(
                ".jpg",
                frame,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            if not encoded:
                raise CameraError(f"camera {camera.name!r} frame could not be JPEG encoded")
            images.append(
                CameraFrame(
                    name=camera.name,
                    mime_type="image/jpeg",
                    data=jpeg.tobytes(),
                    captured_timestamp_ns=captured_timestamp_ns,
                    # This is deliberately sampled after encoding.  It is a
                    # receive/availability time, not the physical capture
                    # time and never substitutes for the latter.
                    received_timestamp_ns=self._now_ns(),
                    clock_domain="host_monotonic_ns",
                    profile=self._actual_profile(camera, self._captures[len(images)]),
                )
            )
        return tuple(images)

    def _capture_frames(self) -> tuple[Any, ...]:
        return tuple(frame for frame, _timestamp in self._capture_frames_with_timestamps())

    def _capture_frames_with_timestamps(self) -> tuple[tuple[Any, int], ...]:
        for camera, capture in zip(self._cameras, self._captures):
            if not capture.grab():
                raise CameraError(f"camera {camera.name!r} failed to grab a frame")
        frames: list[tuple[Any, int]] = []
        for camera, capture in zip(self._cameras, self._captures):
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise CameraError(f"camera {camera.name!r} failed to retrieve a frame")
            height, width = frame.shape[:2]
            if (width, height) != (camera.width, camera.height):
                raise CameraError(
                    f"camera {camera.name!r} returned {width}x{height}; expected {camera.width}x{camera.height}"
                )
            # OpenCV exposes no portable exposure timestamp.  Record the
            # host read boundary before any JPEG encoding or subscriber work.
            frames.append((frame, self._now_ns()))
        return tuple(frames)

    def _now_ns(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CameraError("camera clock must return a non-negative integer nanosecond value")
        return value

    def _actual_profile(
        self,
        camera: V4L2CameraConfig,
        capture: Any,
    ) -> dict[str, object] | None:
        """Read driver-reported values when the OpenCV backend exposes them.

        Requested configuration is not repeated as an ``actual`` profile.
        In particular, a backend without ``get(CAP_PROP_FPS)`` must not be
        represented as successfully running at the requested FPS.
        """

        getter = getattr(capture, "get", None)
        if not callable(getter):
            return None
        values: dict[str, object] = {}
        properties = (
            ("width", "CAP_PROP_FRAME_WIDTH", camera.width),
            ("height", "CAP_PROP_FRAME_HEIGHT", camera.height),
            ("fps", "CAP_PROP_FPS", camera.fps),
        )
        for name, property_name, _requested in properties:
            property_id = getattr(self._cv2, property_name, None)
            if property_id is None:
                continue
            try:
                value = getter(property_id)
            except Exception:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)) or float(value) <= 0:
                continue
            values[name] = int(value) if name != "fps" else float(value)
        return values or None

    def close(self) -> None:
        for capture in self._captures:
            capture.release()
        self._captures.clear()


__all__ = ["CameraError", "V4L2CameraConfig", "V4L2CameraSource"]
