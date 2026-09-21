"""Continuous tracking of capture-time base-frame waypoint plans."""

from __future__ import annotations

import math
from dataclasses import dataclass

from embodirun.utils import Pose2D, compose_relative_pose, relative_pose

from ..motion import (
    DEFAULT_PLANAR_VELOCITY_LIMITS,
    PlanarVelocityCommand,
    PlanarVelocityLimits,
)
from ..plan import WaypointPlan


@dataclass(frozen=True, slots=True)
class WaypointFollowerConfig:
    forward_gain_per_s: float = 0.9
    lateral_gain_per_s: float = 0.8
    heading_gain_per_s: float = 1.1
    goal_yaw_gain_per_s: float = 0.25
    position_tolerance_m: float = 0.08
    yaw_tolerance_rad: float = 0.08
    limits: PlanarVelocityLimits = DEFAULT_PLANAR_VELOCITY_LIMITS

    def __post_init__(self) -> None:
        for name in (
            "forward_gain_per_s",
            "lateral_gain_per_s",
            "heading_gain_per_s",
            "goal_yaw_gain_per_s",
            "position_tolerance_m",
            "yaw_tolerance_rad",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.position_tolerance_m > 1.0:
            raise ValueError("position_tolerance_m must not exceed 1 metre")
        if self.yaw_tolerance_rad > math.pi:
            raise ValueError("yaw_tolerance_rad must not exceed pi")
        if not isinstance(self.limits, PlanarVelocityLimits):
            raise TypeError("limits must be PlanarVelocityLimits")


DEFAULT_WAYPOINT_FOLLOWER_CONFIG = WaypointFollowerConfig()


@dataclass(frozen=True, slots=True)
class FollowerSample:
    command: PlanarVelocityCommand
    active: bool
    terminal_reached: bool
    waypoint_index: int


def _clamp(value: float, maximum: float) -> float:
    return max(-maximum, min(maximum, value))


def relative_target_to_velocity(
    target: Pose2D,
    config: WaypointFollowerConfig = DEFAULT_WAYPOINT_FOLLOWER_CONFIG,
) -> PlanarVelocityCommand:
    if not isinstance(target, Pose2D):
        raise TypeError("target must be a Pose2D")
    distance = math.hypot(target.x_m, target.y_m)
    if distance <= config.position_tolerance_m:
        vx = 0.0
        vy = 0.0
        heading_error = 0.0
    else:
        vx = config.forward_gain_per_s * target.x_m
        vy = config.lateral_gain_per_s * target.y_m
        heading_error = math.atan2(target.y_m, target.x_m)
    goal_yaw = 0.0 if abs(target.yaw_rad) <= config.yaw_tolerance_rad else target.yaw_rad
    yaw_rate = config.heading_gain_per_s * heading_error + config.goal_yaw_gain_per_s * goal_yaw
    limits = config.limits
    return PlanarVelocityCommand(
        _clamp(vx, limits.max_abs_vx_mps),
        _clamp(vy, limits.max_abs_vy_mps),
        _clamp(yaw_rate, limits.max_abs_yaw_rate_rps),
        limits,
    )


class WorldWaypointFollower:
    """Anchor one model plan in odometry and sample it at control rate."""

    def __init__(
        self,
        config: WaypointFollowerConfig = DEFAULT_WAYPOINT_FOLLOWER_CONFIG,
    ) -> None:
        if not isinstance(config, WaypointFollowerConfig):
            raise TypeError("config must be a WaypointFollowerConfig")
        self.config = config
        self.latest_sequence: int | None = None
        self.clear()

    def clear(self) -> None:
        self.sequence: int | None = None
        self._world_targets: tuple[Pose2D, ...] = ()
        self._index = 0
        self._terminal = False
        self._valid_until_s: float | None = None

    def reset_episode(self) -> None:
        self.latest_sequence = None
        self.clear()

    def replace(self, plan: WaypointPlan, capture_pose: Pose2D, accepted_at_s: float) -> None:
        if not isinstance(plan, WaypointPlan):
            raise TypeError("plan must be a WaypointPlan")
        if not isinstance(capture_pose, Pose2D):
            raise TypeError("capture_pose must be a Pose2D")
        if self.latest_sequence is not None and plan.observation_sequence <= self.latest_sequence:
            raise ValueError("observation sequence must increase")
        self.latest_sequence = plan.observation_sequence
        self.sequence = plan.observation_sequence
        self._world_targets = tuple(
            compose_relative_pose(capture_pose, Pose2D(item.x_m, item.y_m, item.yaw_rad)) for item in plan.waypoints
        )
        self._index = 0
        self._terminal = plan.terminal
        self._valid_until_s = float(accepted_at_s) + plan.valid_for_s

    def sample(self, current_pose: Pose2D, monotonic_s: float) -> FollowerSample:
        if not isinstance(current_pose, Pose2D):
            raise TypeError("current_pose must be a Pose2D")
        stopped = PlanarVelocityCommand.stopped(self.config.limits)
        if self.sequence is None:
            return FollowerSample(stopped, False, False, 0)
        if self._valid_until_s is not None and monotonic_s > self._valid_until_s:
            self.clear()
            return FollowerSample(stopped, False, False, 0)
        while self._index < len(self._world_targets):
            relative = relative_pose(self._world_targets[self._index], current_pose)
            reached = (
                math.hypot(relative.x_m, relative.y_m) <= self.config.position_tolerance_m
                and abs(relative.yaw_rad) <= self.config.yaw_tolerance_rad
            )
            if reached:
                self._index += 1
                continue
            return FollowerSample(
                relative_target_to_velocity(relative, self.config),
                True,
                False,
                self._index,
            )
        return FollowerSample(stopped, False, self._terminal, self._index)


__all__ = [
    "DEFAULT_WAYPOINT_FOLLOWER_CONFIG",
    "FollowerSample",
    "WaypointFollowerConfig",
    "WorldWaypointFollower",
    "relative_target_to_velocity",
]
