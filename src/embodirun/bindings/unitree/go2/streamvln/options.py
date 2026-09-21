"""Validated options for starting Unitree Go2 edge services."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Literal, cast

ServiceName = Literal["control", "camera"]
ServiceSelection = Literal["all", "control", "camera"]


@dataclass(frozen=True)
class ControlStartOptions:
    """Remote control-service arguments."""

    mode: Literal["dry-run", "live"] = "live"
    interface: str = "eno2"
    state_topic: str = "rt/lf/sportmodestate"
    cyclonedds_lib_dir: str | None = None
    bind: str = "127.0.0.1"
    port: int = 8080
    api_token: str | None = field(default=None, repr=False)
    operator_ready: bool = False

    def __post_init__(self) -> None:
        _validate_port(self.port, "control")
        _validate_token_for_bind(self.bind, self.api_token, "control API")
        if self.mode == "live" and not self.cyclonedds_lib_dir:
            raise ValueError("live control requires --cyclonedds-lib-dir")


@dataclass(frozen=True)
class CameraStartOptions:
    """Remote camera-service arguments."""

    backend: Literal["realsense", "v4l2"] = "realsense"
    realsense_serial: str | None = None
    depth_scale: float | None = None
    bind: str = "127.0.0.1"
    port: int = 8765
    camera_token: str | None = field(default=None, repr=False)
    width: int = 640
    height: int = 360
    camera_fps: int = 15
    jpeg_fps: float = 5.0
    device: str = "/dev/video4"

    def __post_init__(self) -> None:
        _validate_port(self.port, "camera")
        _validate_token_for_bind(self.bind, self.camera_token, "camera API")
        if self.width <= 0 or self.height <= 0 or self.width % 2:
            raise ValueError("camera dimensions must be positive and width must be even")
        if self.camera_fps <= 0 or self.jpeg_fps <= 0:
            raise ValueError("camera frame rates must be positive")
        if self.backend == "realsense" and (self.depth_scale is None or self.depth_scale <= 0):
            raise ValueError("RealSense camera requires an explicit positive --depth-scale")


@dataclass(frozen=True)
class StartOptions:
    """Complete start request for one or both edge services."""

    python: str
    services: ServiceSelection = "all"
    control: ControlStartOptions | None = None
    camera: CameraStartOptions | None = None

    def __post_init__(self) -> None:
        if not self.python.strip():
            raise ValueError("remote Python path cannot be empty")
        if self.services in {"all", "control"} and self.control is None:
            raise ValueError("control options are required for the selected services")
        if self.services in {"all", "camera"} and self.camera is None:
            raise ValueError("camera options are required for the selected services")


def selected_services(selection: ServiceSelection) -> list[ServiceName]:
    if selection == "all":
        return ["control", "camera"]
    return [cast(ServiceName, selection)]


def _validate_port(port: int, label: str) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"{label} port must be between 1 and 65535")


def _validate_token_for_bind(bind: str, token: str | None, label: str) -> None:
    try:
        loopback = bind.strip().lower() == "localhost" or ipaddress.ip_address(bind).is_loopback
    except ValueError:
        loopback = False
    if not loopback and token is None:
        raise ValueError(f"{label} token is required when binding beyond localhost")
    if token is not None:
        if len(token) < 32:
            raise ValueError(f"{label} token must contain at least 32 characters")
        if not token.isascii() or token.strip() != token or any(char.isspace() for char in token):
            raise ValueError(f"{label} token must be ASCII without whitespace")


__all__ = [
    "CameraStartOptions",
    "ControlStartOptions",
    "ServiceName",
    "ServiceSelection",
    "StartOptions",
]
