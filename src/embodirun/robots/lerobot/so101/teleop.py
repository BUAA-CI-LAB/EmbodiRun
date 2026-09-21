"""SO-101 teleoperation intent mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from embodirun.robots import RobotAction, RobotObservation

from .adapter import SO101_ACTION_SPACE


@dataclass(frozen=True, slots=True)
class SO101TeleopScale:
    joint_step_deg: float = 2.0
    gripper_step: float = 2.0


# The default scale is an immutable module-level singleton, so the default
# argument does not build a new object per call.
SO101_DEFAULT_SCALE = SO101TeleopScale()


def resolve_axes_action(
    observation: RobotObservation,
    axes: Mapping[object, object],
    *,
    timestamp_s: float,
    scale: SO101TeleopScale = SO101_DEFAULT_SCALE,
) -> RobotAction:
    action_space = observation.metadata.get("action_space")
    if action_space != SO101_ACTION_SPACE:
        raise ValueError(f"lerobot.so101 teleop cannot target action space {action_space!r}")
    if not isinstance(observation.values, Mapping):
        raise ValueError("SO-101 observation values must be an object")
    joints = _number_list(
        observation.values.get("joint_positions_deg"),
        "joint_positions_deg",
        5,
    )
    gripper = _number(observation.values.get("gripper_position"), "gripper_position")
    joint_axes = ("left_x", "left_y", "right_x", "right_y", "dpad_x")
    target_joints = [value + _axis(axes, axis) * scale.joint_step_deg for value, axis in zip(joints, joint_axes)]
    target_gripper = max(
        0.0,
        min(100.0, gripper + _axis(axes, "dpad_y") * scale.gripper_step),
    )
    return RobotAction(
        timestamp_s=timestamp_s,
        values={
            "type": "joint_position",
            "joint_positions_deg": target_joints,
            "gripper_position": target_gripper,
        },
        metadata={"action_space": SO101_ACTION_SPACE, "source": "teleop_axes"},
    )


def _axis(axes: Mapping[object, object], name: str) -> float:
    value = _number(axes.get(name, 0.0), f"axes.{name}")
    return max(-1.0, min(1.0, value))


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _number_list(value: object, name: str, length: int) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a sequence")
    try:
        result = [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]
    except TypeError:
        raise ValueError(f"{name} must be a sequence") from None
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values")
    return result


__all__ = ["SO101TeleopScale", "resolve_axes_action"]
