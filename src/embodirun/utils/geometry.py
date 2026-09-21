"""Dependency-free SE(2) helpers shared by navigation integrations."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to the closed interval [-pi, pi]."""

    angle = _finite(angle_rad, "angle_rad")
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True, slots=True)
class Pose2D:
    """One planar pose using metres and radians."""

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite(self.y_m, "y_m"))
        object.__setattr__(self, "yaw_rad", normalize_angle(self.yaw_rad))


def compose_relative_pose(origin: Pose2D, relative: Pose2D) -> Pose2D:
    """Express a pose from ``origin``'s frame in the parent/world frame."""

    if not isinstance(origin, Pose2D) or not isinstance(relative, Pose2D):
        raise TypeError("origin and relative must be Pose2D values")
    cosine = math.cos(origin.yaw_rad)
    sine = math.sin(origin.yaw_rad)
    return Pose2D(
        origin.x_m + cosine * relative.x_m - sine * relative.y_m,
        origin.y_m + sine * relative.x_m + cosine * relative.y_m,
        origin.yaw_rad + relative.yaw_rad,
    )


def relative_pose(target: Pose2D, origin: Pose2D) -> Pose2D:
    """Express a parent/world-frame ``target`` in ``origin``'s frame."""

    if not isinstance(target, Pose2D) or not isinstance(origin, Pose2D):
        raise TypeError("target and origin must be Pose2D values")
    dx = target.x_m - origin.x_m
    dy = target.y_m - origin.y_m
    cosine = math.cos(origin.yaw_rad)
    sine = math.sin(origin.yaw_rad)
    return Pose2D(
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        target.yaw_rad - origin.yaw_rad,
    )


__all__ = ["Pose2D", "compose_relative_pose", "normalize_angle", "relative_pose"]
