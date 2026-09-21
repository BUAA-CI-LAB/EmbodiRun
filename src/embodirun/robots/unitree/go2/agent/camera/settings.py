"""Validated settings for the Go2 camera service."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Optional


def is_loopback_bind(host: str) -> bool:
    candidate = host.strip().lower()
    if candidate == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def validate_bind_token(host: str, token: Optional[str]) -> None:  # noqa: UP045
    """Require a strong bearer token when exposing the service off-host."""

    if token is not None:
        try:
            token.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("camera token must contain only ASCII characters") from error
        if len(token) < 32:
            raise ValueError("camera token must contain at least 32 characters")
        if token.strip() != token or any(character.isspace() for character in token):
            raise ValueError("camera token must not contain whitespace")
    if not is_loopback_bind(host) and token is None:
        raise ValueError("camera token is required when binding beyond localhost")


@dataclass(frozen=True)
class CameraServiceConfig:
    backend: str
    host: str
    port: int
    token: Optional[str]  # noqa: UP045
    serial: Optional[str]  # noqa: UP045
    width: int
    height: int
    fps: int
    device: str
    warmup_frames: int
    jpeg_quality: int
    jpeg_fps: float
    retry_delay: float
    frame_timeout: float
    format_timeout: float
    max_frame_age: float
    depth_device: str
    depth_width: int
    depth_height: int
    depth_warmup_frames: int
    depth_scale: Optional[float]  # noqa: UP045
    depth_calibrated: bool
    rgb_depth_alignment_claimed: bool
    depth_fps: float
    max_depth_age: float
    max_depth_m: float
    access_log: bool

    @property
    def profile(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"

    @property
    def initial_device(self) -> str:
        if self.backend != "realsense":
            return self.device
        return "realsense" if not self.serial else f"realsense:{self.serial}"

    def validate(self) -> None:
        if self.backend not in {"v4l2", "realsense"}:
            raise ValueError("backend must be v4l2 or realsense")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        validate_bind_token(self.host, self.token)
        if self.width % 2:
            raise ValueError("width must be even for packed YUYV diagnostics")
        if self.warmup_frames < 0 or self.depth_warmup_frames < 0:
            raise ValueError("warmup frame counts must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg quality must be between 1 and 100")
        if self.depth_calibrated and self.depth_scale is None:
            raise ValueError("depth calibration requires an explicit depth scale")
        if self.depth_scale is not None and not self.depth_calibrated:
            raise ValueError("depth scale requires explicit depth calibration")
        if self.rgb_depth_alignment_claimed and not self.depth_calibrated:
            raise ValueError("RGB-depth alignment claim requires depth calibration")
        if self.backend == "realsense" and not self.depth_calibrated:
            raise ValueError("realsense backend requires explicit depth calibration")
        if self.backend == "realsense" and (self.width != self.depth_width or self.height != self.depth_height):
            raise ValueError("realsense backend requires matching RGB/depth dimensions")


__all__ = ["CameraServiceConfig", "is_loopback_bind", "validate_bind_token"]
