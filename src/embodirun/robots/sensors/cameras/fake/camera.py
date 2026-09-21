"""Static PNG camera source for dependency-free local examples."""

from __future__ import annotations

import struct
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..camera import CameraFrame


class FakeCameraError(RuntimeError):
    """A fake camera configuration is invalid."""


@dataclass(frozen=True, slots=True)
class FakeCameraConfig:
    name: str
    width: int = 64
    height: int = 48
    color: tuple[int, int, int] = (32, 96, 160)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera name must not be empty")
        for field_name in ("width", "height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"camera {field_name} must be a positive integer")
        if len(self.color) != 3:
            raise ValueError("camera color must contain three channels")
        if any(
            isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
            for channel in self.color
        ):
            raise ValueError("camera color channels must be integers in [0, 255]")


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _static_png(config: FakeCameraConfig) -> bytes:
    """Encode one solid RGB image using only the Python standard library."""

    row = bytes(config.color) * config.width
    raw = b"".join(b"\x00" + row for _ in range(config.height))
    header = struct.pack(">IIBBBBB", config.width, config.height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, level=1))
        + _chunk(b"IEND", b"")
    )


class FakeCameraSource:
    """Return a deterministic PNG while recording host read timestamps."""

    def __init__(
        self,
        cameras: Sequence[FakeCameraConfig],
        *,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._cameras = tuple(cameras)
        if not self._cameras:
            raise FakeCameraError("at least one fake camera is required")
        self._clock_ns = clock_ns or time.monotonic_ns
        self._frames = tuple(_static_png(camera) for camera in self._cameras)
        self._closed = False

    def _now_ns(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FakeCameraError("fake camera clock must return a non-negative integer nanosecond value")
        return value

    def capture(self) -> tuple[CameraFrame, ...]:
        if self._closed:
            raise FakeCameraError("fake camera source is closed")
        frames: list[CameraFrame] = []
        for camera, data in zip(self._cameras, self._frames):
            captured_timestamp_ns = self._now_ns()
            frames.append(
                CameraFrame(
                    name=camera.name,
                    mime_type="image/png",
                    data=data,
                    captured_timestamp_ns=captured_timestamp_ns,
                    received_timestamp_ns=self._now_ns(),
                    clock_domain="host_monotonic_ns",
                    profile={
                        "simulated": True,
                        "width": camera.width,
                        "height": camera.height,
                    },
                )
            )
        return tuple(frames)

    def close(self) -> None:
        self._closed = True


__all__ = ["FakeCameraConfig", "FakeCameraError", "FakeCameraSource"]
