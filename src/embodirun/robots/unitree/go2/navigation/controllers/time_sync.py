"""Project planar base odometry to an observation capture timestamp."""

from __future__ import annotations

import math

from embodirun.utils import Pose2D

from ..motion import MobileBaseState

MAX_PROJECTION_S = 1.0


def project_pose_to_time(state: MobileBaseState, captured_at_s: float) -> Pose2D:
    if not isinstance(state, MobileBaseState):
        raise TypeError("state must be a MobileBaseState")
    timestamp = state.received_at_s
    if timestamp is None:
        return state.pose
    delta_s = float(captured_at_s) - timestamp
    if not math.isfinite(delta_s) or abs(delta_s) > MAX_PROJECTION_S:
        return state.pose
    yaw_mid = state.pose.yaw_rad + 0.5 * state.yaw_rate_rps * delta_s
    world_vx = state.forward_velocity_mps * math.cos(yaw_mid) - state.lateral_velocity_mps * math.sin(yaw_mid)
    world_vy = state.forward_velocity_mps * math.sin(yaw_mid) + state.lateral_velocity_mps * math.cos(yaw_mid)
    return Pose2D(
        state.pose.x_m + world_vx * delta_s,
        state.pose.y_m + world_vy * delta_s,
        state.pose.yaw_rad + state.yaw_rate_rps * delta_s,
    )


def capture_pose(
    before: MobileBaseState,
    after: MobileBaseState,
    captured_at_s: float,
) -> Pose2D:
    if not isinstance(before, MobileBaseState) or not isinstance(after, MobileBaseState):
        raise TypeError("before and after must be MobileBaseState values")
    timestamped = tuple(state for state in (before, after) if state.received_at_s is not None)
    if not timestamped:
        return after.pose
    nearest = min(
        timestamped,
        key=lambda state: abs(float(state.received_at_s) - captured_at_s),
    )
    return project_pose_to_time(nearest, captured_at_s)


__all__ = ["MAX_PROJECTION_S", "capture_pose", "project_pose_to_time"]
