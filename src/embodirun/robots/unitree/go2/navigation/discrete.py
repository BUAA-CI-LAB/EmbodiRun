"""Metric discrete-navigation commands for the Go2 embodiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ....adapter import RobotAction


class NavigationCommandKind(str, Enum):
    STOP = "stop"
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass(frozen=True, slots=True)
class NavigationCommand:
    """One bounded metric navigation command for a Go2 target."""

    kind: NavigationCommandKind
    distance_m: float = 0.0
    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NavigationCommandKind):
            raise TypeError("navigation command kind is invalid")
        for name in ("distance_m", "angle_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"navigation command {name} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"navigation command {name} must be finite and non-negative")
            object.__setattr__(self, name, number)

        if self.kind is NavigationCommandKind.MOVE_FORWARD:
            if self.distance_m <= 0 or self.angle_deg != 0:
                raise ValueError("move_forward requires distance_m and no angle_deg")
        elif self.kind in {
            NavigationCommandKind.TURN_LEFT,
            NavigationCommandKind.TURN_RIGHT,
        }:
            if self.angle_deg <= 0 or self.distance_m != 0:
                raise ValueError("turn commands require angle_deg and no distance_m")
        elif self.distance_m != 0 or self.angle_deg != 0:
            raise ValueError("stop must not include a distance or angle")

    def to_action_values(self) -> dict[str, object]:
        values: dict[str, object] = {"command": self.kind.value}
        if self.kind is NavigationCommandKind.MOVE_FORWARD:
            values["distance_m"] = self.distance_m
        elif self.kind in {
            NavigationCommandKind.TURN_LEFT,
            NavigationCommandKind.TURN_RIGHT,
        }:
            values["angle_deg"] = self.angle_deg
        return values

    @classmethod
    def from_robot_action(cls, action: RobotAction) -> NavigationCommand:
        if not isinstance(action, RobotAction):
            raise TypeError("navigation action must be a RobotAction")
        values = action.values
        if not isinstance(values, Mapping):
            raise TypeError("navigation action values must be an object")
        values = dict(values)
        command = values.get("command")
        try:
            kind = NavigationCommandKind(command)
        except (TypeError, ValueError):
            available = ", ".join(item.value for item in NavigationCommandKind)
            raise ValueError(f"navigation action command must be one of: {available}") from None

        allowed = {"command"}
        distance_m = 0.0
        angle_deg = 0.0
        if kind is NavigationCommandKind.MOVE_FORWARD:
            allowed.add("distance_m")
            distance_m = _number(values.get("distance_m"), "distance_m")
        elif kind in {
            NavigationCommandKind.TURN_LEFT,
            NavigationCommandKind.TURN_RIGHT,
        }:
            allowed.add("angle_deg")
            angle_deg = _number(values.get("angle_deg"), "angle_deg")
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("unknown navigation action fields: " + ", ".join(unknown))
        return cls(kind=kind, distance_m=distance_m, angle_deg=angle_deg)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"navigation action {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"navigation action {name} must be finite and positive")
    return result


__all__ = [
    "NavigationCommand",
    "NavigationCommandKind",
]
