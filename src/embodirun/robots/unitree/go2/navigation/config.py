"""Validated configuration for navigation task sessions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .motion import DEFAULT_PLANAR_VELOCITY_LIMITS, PlanarVelocityLimits


def _normalize_number(
    instance: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    value = getattr(instance, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    object.__setattr__(instance, name, value)


def _require_positive_int(instance: object, name: str) -> None:
    value = getattr(instance, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class NavigationSessionConfig:
    control_hz: float = 10.0
    max_runtime_s: float = 120.0
    execute: bool = False
    lease_duration_s: float = 10.0
    max_events: int = 256

    def __post_init__(self) -> None:
        for name in ("control_hz", "max_runtime_s", "lease_duration_s"):
            _normalize_number(self, name)
        if not isinstance(self.execute, bool):
            raise TypeError("execute must be a boolean")
        _require_positive_int(self, "max_events")


DEFAULT_NAVIGATION_SESSION_CONFIG = NavigationSessionConfig()


@dataclass(frozen=True, slots=True)
class ReactiveNavigationSessionConfig:
    max_runtime_s: float = 120.0
    execute: bool = False
    linear_speed_mps: float = 0.30
    yaw_rate_rps: float = 0.60
    min_pulse_s: float = 0.05
    max_pulse_s: float = 1.50
    settle_s: float = 0.15
    state_attempts: int = 3
    state_retry_delay_s: float = 0.15
    max_events: int = 256
    limits: PlanarVelocityLimits = DEFAULT_PLANAR_VELOCITY_LIMITS
    max_waypoints_per_observation: int = 1
    terminal_after_waypoints: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.limits, PlanarVelocityLimits):
            raise TypeError("limits must be PlanarVelocityLimits")
        for name in (
            "max_runtime_s",
            "linear_speed_mps",
            "yaw_rate_rps",
            "min_pulse_s",
            "max_pulse_s",
        ):
            _normalize_number(self, name)
        for name in ("settle_s", "state_retry_delay_s"):
            _normalize_number(self, name, allow_zero=True)
        if self.min_pulse_s > self.max_pulse_s:
            raise ValueError("min_pulse_s must not exceed max_pulse_s")
        if self.linear_speed_mps > min(
            self.limits.max_abs_vx_mps,
            self.limits.max_abs_vy_mps,
        ):
            raise ValueError("linear_speed_mps exceeds planar velocity limits")
        if self.yaw_rate_rps > self.limits.max_abs_yaw_rate_rps:
            raise ValueError("yaw_rate_rps exceeds planar velocity limits")
        if not isinstance(self.execute, bool):
            raise TypeError("execute must be a boolean")
        if not isinstance(self.terminal_after_waypoints, bool):
            raise TypeError("terminal_after_waypoints must be a boolean")
        _require_positive_int(self, "state_attempts")
        _require_positive_int(self, "max_waypoints_per_observation")
        _require_positive_int(self, "max_events")


DEFAULT_REACTIVE_NAVIGATION_SESSION_CONFIG = ReactiveNavigationSessionConfig()


__all__ = [
    "DEFAULT_NAVIGATION_SESSION_CONFIG",
    "DEFAULT_REACTIVE_NAVIGATION_SESSION_CONFIG",
    "NavigationSessionConfig",
    "ReactiveNavigationSessionConfig",
]
