"""Host-side configuration for a Unitree Go2 control adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .navigation.motion import PlanarVelocityLimits
from .profile import MAX_LINEAR_SPEED_MPS, MAX_YAW_RATE_RPS


def _default_velocity_limits() -> PlanarVelocityLimits:
    return PlanarVelocityLimits(
        max_abs_vx_mps=MAX_LINEAR_SPEED_MPS,
        max_abs_vy_mps=MAX_LINEAR_SPEED_MPS,
        max_abs_yaw_rate_rps=MAX_YAW_RATE_RPS,
    )


@dataclass(frozen=True, slots=True)
class Go2Config:
    """Connection and motion bounds for one Go2 control agent."""

    control_url: str
    robot_id: str = "go2"
    api_token: str | None = field(default=None, repr=False)
    timeout_s: float = 1.0
    expected_transport: str | None = "unitree-sdk2"
    velocity_lease_duration_s: float = 10.0
    velocity_limits: PlanarVelocityLimits = field(default_factory=_default_velocity_limits)

    @classmethod
    def from_mapping(
        cls,
        robot_id: str,
        value: Mapping[str, Any],
    ) -> Go2Config:
        """Build one Go2 configuration from its deployment YAML options."""

        options = dict(value)
        allowed = {
            "control_url",
            "api_token",
            "timeout_s",
            "expected_transport",
            "velocity_lease_duration_s",
            "velocity_limits",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown Go2 configuration fields: {', '.join(unknown)}")
        limits = _velocity_limits(options.get("velocity_limits"))
        return cls(
            control_url=options.get("control_url"),
            robot_id=robot_id,
            api_token=options.get("api_token"),
            timeout_s=options.get("timeout_s", 1.0),
            expected_transport=options.get("expected_transport", "unitree-sdk2"),
            velocity_lease_duration_s=options.get(
                "velocity_lease_duration_s",
                10.0,
            ),
            velocity_limits=limits,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.control_url, str):
            raise TypeError("control_url must be a string")
        parsed_url = urlsplit(self.control_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("control_url must be an absolute HTTP(S) URL")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        for name in ("timeout_s", "velocity_lease_duration_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.api_token is not None and (not isinstance(self.api_token, str) or not self.api_token.strip()):
            raise ValueError("api_token must not be empty when provided")
        if self.expected_transport is not None and (
            not isinstance(self.expected_transport, str) or not self.expected_transport.strip()
        ):
            raise ValueError("expected_transport must not be empty when provided")
        if not isinstance(self.velocity_limits, PlanarVelocityLimits):
            raise TypeError("velocity_limits must be PlanarVelocityLimits")


def _velocity_limits(value: object) -> PlanarVelocityLimits:
    if value is None:
        return _default_velocity_limits()
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("velocity_limits must be a mapping")
    options = dict(value)
    allowed = {
        "max_abs_vx_mps",
        "max_abs_vy_mps",
        "max_abs_yaw_rate_rps",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown velocity limit fields: {', '.join(unknown)}")
    defaults = _default_velocity_limits()
    return PlanarVelocityLimits(
        max_abs_vx_mps=options.get(
            "max_abs_vx_mps",
            defaults.max_abs_vx_mps,
        ),
        max_abs_vy_mps=options.get(
            "max_abs_vy_mps",
            defaults.max_abs_vy_mps,
        ),
        max_abs_yaw_rate_rps=options.get(
            "max_abs_yaw_rate_rps",
            defaults.max_abs_yaw_rate_rps,
        ),
    )


__all__ = ["Go2Config"]
