"""Capture-frame waypoint output contract for navigation policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ._validation import NavigationContractError, finite, sequence

MAX_WAYPOINTS = 64
MAX_WAYPOINT_COORDINATE_M = 20.0
MAX_OUTPUT_VALID_FOR_S = 10.0


@dataclass(frozen=True, slots=True)
class Waypoint:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x_m",
            finite(
                self.x_m,
                "waypoint.x_m",
                minimum=-MAX_WAYPOINT_COORDINATE_M,
                maximum=MAX_WAYPOINT_COORDINATE_M,
            ),
        )
        object.__setattr__(
            self,
            "y_m",
            finite(
                self.y_m,
                "waypoint.y_m",
                minimum=-MAX_WAYPOINT_COORDINATE_M,
                maximum=MAX_WAYPOINT_COORDINATE_M,
            ),
        )
        object.__setattr__(
            self,
            "yaw_rad",
            finite(self.yaw_rad, "waypoint.yaw_rad", minimum=-math.pi, maximum=math.pi),
        )


@dataclass(frozen=True, slots=True)
class WaypointPlan:
    observation_sequence: int
    waypoints: Sequence[Waypoint]
    terminal: bool = False
    confidence: float = 1.0
    valid_for_s: float = 5.0
    frame: Literal["base_link"] = "base_link"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_sequence",
            sequence(self.observation_sequence, "observation_sequence"),
        )
        if self.frame != "base_link":
            raise NavigationContractError("waypoint frame must be 'base_link'")
        if isinstance(self.waypoints, (str, bytes)) or not isinstance(self.waypoints, Sequence):
            raise NavigationContractError("waypoints must be a sequence")
        waypoints = tuple(self.waypoints)
        if not waypoints and not self.terminal:
            raise NavigationContractError("an empty waypoint plan must be terminal")
        if len(waypoints) > MAX_WAYPOINTS:
            raise NavigationContractError(f"waypoint plan exceeds {MAX_WAYPOINTS} points")
        if any(not isinstance(waypoint, Waypoint) for waypoint in waypoints):
            raise NavigationContractError("waypoints must contain Waypoint values")
        if not isinstance(self.terminal, bool):
            raise NavigationContractError("terminal must be a boolean")
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(
            self,
            "confidence",
            finite(self.confidence, "confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "valid_for_s",
            finite(
                self.valid_for_s,
                "valid_for_s",
                minimum=math.nextafter(0.0, 1.0),
                maximum=MAX_OUTPUT_VALID_FOR_S,
            ),
        )


NavigationResult: TypeAlias = WaypointPlan

__all__ = [
    "MAX_OUTPUT_VALID_FOR_S",
    "MAX_WAYPOINTS",
    "MAX_WAYPOINT_COORDINATE_M",
    "NavigationResult",
    "Waypoint",
    "WaypointPlan",
]
