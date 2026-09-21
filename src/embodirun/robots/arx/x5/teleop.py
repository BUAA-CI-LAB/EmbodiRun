"""ARX5 teleoperation intent mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from embodirun.robots import RobotAction, RobotObservation

from .adapter import ARX5_ACTION_SPACE


@dataclass(frozen=True, slots=True)
class ARX5TeleopScale:
    translation_step_m: float = 0.01
    rotation_step_rad: float = 0.04
    gripper_step_m: float = 0.01


# Default scales are immutable module-level singletons, so the default argument
# does not build a new object per call.
ARX5_DEFAULT_SCALE = ARX5TeleopScale()


def resolve_axes_action(
    observation: RobotObservation,
    axes: Mapping[object, object],
    *,
    timestamp_s: float,
    scale: ARX5TeleopScale = ARX5_DEFAULT_SCALE,
) -> RobotAction:
    action_space = observation.metadata.get("action_space")
    if action_space != ARX5_ACTION_SPACE:
        raise ValueError(f"arx.x5 teleop cannot target action space {action_space!r}")
    if not isinstance(observation.values, Mapping):
        raise ValueError("ARX5 observation values must be an object")
    current = _number_list(
        observation.values.get("eef_xyzrpy_gripper"),
        "eef_xyzrpy_gripper",
        7,
    )
    target = list(current)
    target[0] += _axis(axes, "left_x") * scale.translation_step_m
    target[1] += _axis(axes, "left_y") * scale.translation_step_m
    target[2] += (_axis(axes, "rt") - _axis(axes, "lt")) * scale.translation_step_m
    target[5] += _axis(axes, "right_x") * scale.rotation_step_rad
    target[6] += _axis(axes, "dpad_y") * scale.gripper_step_m
    return RobotAction(
        timestamp_s=timestamp_s,
        values={"type": "eef_xyzrpy_gripper", "eef_xyzrpy_gripper": target},
        metadata={"action_space": ARX5_ACTION_SPACE, "source": "teleop_axes"},
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


__all__ = ["ARX5TeleopScale", "resolve_axes_action"]
