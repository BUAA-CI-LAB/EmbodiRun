"""Configuration and hard admission bounds for the Go2 control agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ...profile import MAX_LINEAR_SPEED_MPS, MAX_YAW_RATE_RPS

MAX_MOVE_DISTANCE_M = 1.0
MAX_TURN_ANGLE_RAD = math.pi / 2
MAX_MOTION_DURATION_S = 10.0
STREAM_HEARTBEAT_TIMEOUT_S = 2.0
MAX_TILT_RAD = 0.5
STATE_MAX_AGE_S = 1.0
CONTROL_PERIOD_S = 0.05
MAX_REQUEST_BYTES = 4096
MIN_EXTERNAL_TOKEN_CHARS = 32
WORKER_STOP_JOIN_TIMEOUT_S = 2.5
SDK_OWNER_INIT_TIMEOUT_S = 8.0
SDK_OWNER_CALL_GRACE_S = 1.0
SDK_OWNER_CLOSE_TIMEOUT_S = 4.0


@dataclass(frozen=True)
class ControlServerConfig:
    """Fully resolved process configuration.

    Deployment-specific values are supplied by the CLI/environment.  This
    object intentionally contains no SSH credentials: it runs on the robot
    after deployment and never opens an SSH connection itself.
    """

    mode: Literal["dry-run", "live"]
    host: str
    port: int
    interface: str | None
    state_topic: str
    cyclonedds_lib_dir: str | None
    token: str | None
    operator_ready: bool
    rpc_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"dry-run", "live"}:
            raise ValueError("mode must be 'dry-run' or 'live'")
        if not self.host:
            raise ValueError("host must not be empty")
        if isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not self.state_topic:
            raise ValueError("state_topic must not be empty")
        if not math.isfinite(self.rpc_timeout_s) or self.rpc_timeout_s <= 0.0:
            raise ValueError("rpc_timeout_s must be finite and positive")
        if self.mode == "live" and not self.interface:
            raise ValueError("live mode requires a DDS network interface")
        if self.mode == "live" and not self.cyclonedds_lib_dir:
            raise ValueError("live mode requires a CycloneDDS library directory")


__all__ = [
    "CONTROL_PERIOD_S",
    "MAX_LINEAR_SPEED_MPS",
    "MAX_MOTION_DURATION_S",
    "MAX_MOVE_DISTANCE_M",
    "MAX_REQUEST_BYTES",
    "MAX_TILT_RAD",
    "MAX_TURN_ANGLE_RAD",
    "MAX_YAW_RATE_RPS",
    "MIN_EXTERNAL_TOKEN_CHARS",
    "SDK_OWNER_CALL_GRACE_S",
    "SDK_OWNER_CLOSE_TIMEOUT_S",
    "SDK_OWNER_INIT_TIMEOUT_S",
    "STATE_MAX_AGE_S",
    "STREAM_HEARTBEAT_TIMEOUT_S",
    "WORKER_STOP_JOIN_TIMEOUT_S",
    "ControlServerConfig",
]
