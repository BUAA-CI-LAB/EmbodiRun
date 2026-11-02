"""Validated settings for the Go2 camera service."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class CameraServiceConfig:
    backend: str = "v4l2"
    serial: Optional[str] = None  # noqa: UP045
    width: int = 640
    height: int = 480
    fps: int = 15
    device: str = "/dev/video4"
    warmup_frames: int = 10
    jpeg_quality: int = 85
    jpeg_fps: float = 5.0
    retry_delay: float = 2.0
    frame_timeout: float = 2.0
    format_timeout: float = 5.0
    max_frame_age: float = 2.0
    depth_device: str = "/dev/video0"
    depth_width: int = 640
    depth_height: int = 480
    depth_warmup_frames: int = 10
    depth_scale: Optional[float] = None  # noqa: UP045
    depth_calibrated: bool = False
    rgb_depth_alignment_claimed: bool = False
    depth_fps: float = 10.0
    max_depth_age: float = 0.5
    max_depth_m: float = 10.0

    @classmethod
    def from_mapping(cls, value: dict) -> CameraServiceConfig:
        """Read component settings without a second server or environment parser."""
        if not isinstance(value, dict):
            raise ValueError("camera must be an object or null")
        options = dict(value)
        profile = options.pop("profile", "640x480@15")
        match = re.fullmatch(r"(\d+)x(\d+)@(\d+)", profile) if isinstance(profile, str) else None
        if match is None:
            raise ValueError("camera.profile must have the form WIDTHxHEIGHT@FPS")
        width, height, fps = map(int, match.groups())
        for key, number in (
            ("width", width),
            ("height", height),
            ("fps", fps),
            ("depth_width", width),
            ("depth_height", height),
        ):
            options.setdefault(key, number)
        if set(options) - {field.name for field in fields(cls)}:
            raise ValueError("unknown camera option; configure the shared endpoint in server")
        config = cls(**options)
        config.validate()
        return config

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
        for name in ("width", "height", "fps", "depth_width", "depth_height", "jpeg_quality"):
            number = getattr(self, name)
            if type(number) is not int or number <= 0:
                raise ValueError(f"camera.{name} must be a positive integer")
        for name in ("warmup_frames", "depth_warmup_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"camera.{name} must be a non-negative integer")
        for name in (
            "jpeg_fps",
            "retry_delay",
            "frame_timeout",
            "format_timeout",
            "max_frame_age",
            "depth_fps",
            "max_depth_age",
            "max_depth_m",
            "depth_scale",
        ):
            number = getattr(self, name)
            if name == "depth_scale" and number is None:
                continue
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"camera.{name} must be finite and positive")
        for name in ("depth_calibrated", "rgb_depth_alignment_claimed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"camera.{name} must be boolean")
        for name in ("device", "depth_device"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"camera.{name} must be a device path")
        if self.serial is not None and (not isinstance(self.serial, str) or not self.serial):
            raise ValueError("camera.serial must be a non-empty string")
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


__all__ = ["CameraServiceConfig"]
