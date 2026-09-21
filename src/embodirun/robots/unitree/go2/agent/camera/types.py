"""Value types shared by the Go2 camera-agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional  # noqa: UP035


class CameraStreamError(RuntimeError):
    """Raised when a camera backend cannot produce a trustworthy frame."""


@dataclass(frozen=True)
class VideoProfile:
    """Validated active video profile.

    ``bytes_per_line`` and ``size_image`` are kept alongside the advertised
    dimensions because accepting a driver-coerced stride can silently mix
    frame boundaries when a raw V4L2 pipe is used.
    """

    width: int
    height: int
    pixel_format: str
    bytes_per_line: int
    size_image: int
    fps: Optional[int] = None  # noqa: UP045

    def as_dict(self) -> Dict[str, object]:  # noqa: UP006
        return {
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "bytes_per_line": self.bytes_per_line,
            "size_image": self.size_image,
            "fps": self.fps,
            "validated": True,
        }


@dataclass(frozen=True)
class FrameSnapshot:
    jpeg: bytes
    sequence: int
    captured_at_unix: float
    captured_at_monotonic: float
    source_timestamp_ms: Optional[float] = None  # noqa: UP045
    source_frame_number: Optional[int] = None  # noqa: UP045


@dataclass(frozen=True)
class DepthSnapshot:
    center_distance_m: Optional[float]  # noqa: UP045
    minimum_distance_m: Optional[float]  # noqa: UP045
    valid_fraction: float
    sequence: int
    captured_at_unix: float
    captured_at_monotonic: float
    depth_png: Optional[bytes] = None  # noqa: UP045
    source_timestamp_ms: Optional[float] = None  # noqa: UP045
    source_frame_number: Optional[int] = None  # noqa: UP045


@dataclass(frozen=True)
class DepthAnalysis:
    center_distance_m: Optional[float]  # noqa: UP045
    minimum_distance_m: Optional[float]  # noqa: UP045
    valid_fraction: float


__all__ = [
    "CameraStreamError",
    "DepthAnalysis",
    "DepthSnapshot",
    "FrameSnapshot",
    "VideoProfile",
]
