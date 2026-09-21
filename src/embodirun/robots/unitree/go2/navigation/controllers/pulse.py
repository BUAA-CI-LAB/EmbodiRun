"""Convert one waypoint into one bounded open-loop base pulse."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..motion import (
    DEFAULT_PLANAR_VELOCITY_LIMITS,
    PlanarVelocityCommand,
    PlanarVelocityLimits,
)
from ..plan import Waypoint


@dataclass(frozen=True, slots=True)
class VelocityPulseConfig:
    linear_speed_mps: float = 0.30
    yaw_rate_rps: float = 0.60
    min_duration_s: float = 0.05
    max_duration_s: float = 1.50
    limits: PlanarVelocityLimits = DEFAULT_PLANAR_VELOCITY_LIMITS

    def __post_init__(self) -> None:
        if not isinstance(self.limits, PlanarVelocityLimits):
            raise TypeError("limits must be PlanarVelocityLimits")
        for name in (
            "linear_speed_mps",
            "yaw_rate_rps",
            "min_duration_s",
            "max_duration_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.min_duration_s > self.max_duration_s:
            raise ValueError("min_duration_s must not exceed max_duration_s")
        if self.linear_speed_mps > min(
            self.limits.max_abs_vx_mps,
            self.limits.max_abs_vy_mps,
        ):
            raise ValueError("linear_speed_mps exceeds planar velocity limits")
        if self.yaw_rate_rps > self.limits.max_abs_yaw_rate_rps:
            raise ValueError("yaw_rate_rps exceeds planar velocity limits")


DEFAULT_VELOCITY_PULSE_CONFIG = VelocityPulseConfig()


def waypoint_to_velocity_pulse(
    waypoint: Waypoint,
    config: VelocityPulseConfig = DEFAULT_VELOCITY_PULSE_CONFIG,
) -> tuple[PlanarVelocityCommand, float]:
    if not isinstance(waypoint, Waypoint):
        raise TypeError("waypoint must be a Waypoint")
    if not isinstance(config, VelocityPulseConfig):
        raise TypeError("config must be a VelocityPulseConfig")
    distance_m = math.hypot(waypoint.x_m, waypoint.y_m)
    required_s = max(
        distance_m / config.linear_speed_mps,
        abs(waypoint.yaw_rad) / config.yaw_rate_rps,
        config.min_duration_s,
    )
    duration_s = min(required_s, config.max_duration_s)

    def bounded(value: float, maximum: float) -> float:
        return max(-maximum, min(maximum, value))

    limits = config.limits
    return (
        PlanarVelocityCommand(
            bounded(waypoint.x_m / duration_s, limits.max_abs_vx_mps),
            bounded(waypoint.y_m / duration_s, limits.max_abs_vy_mps),
            bounded(waypoint.yaw_rad / duration_s, limits.max_abs_yaw_rate_rps),
            limits,
        ),
        duration_s,
    )


__all__ = [
    "DEFAULT_VELOCITY_PULSE_CONFIG",
    "VelocityPulseConfig",
    "waypoint_to_velocity_pulse",
]
