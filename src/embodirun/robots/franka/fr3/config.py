"""Safety limits for Franky motion commands."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FR3Config:
    host: str
    robot_id: str = "fr3"
    relative_dynamics_factor: float = 0.05
    max_joint_step_rad: float = 0.20
    max_cartesian_step_m: float = 0.05
    enable_gripper: bool = True
    gripper_speed_mps: float = 0.02
    max_gripper_width_m: float = 0.08

    @classmethod
    def from_mapping(
        cls,
        robot_id: str,
        value: Mapping[str, Any],
    ) -> FR3Config:
        """Build one FR3 configuration from its deployment YAML options."""

        options = dict(value)
        allowed = {
            "host",
            "relative_dynamics_factor",
            "max_joint_step_rad",
            "max_cartesian_step_m",
            "enable_gripper",
            "gripper_speed_mps",
            "max_gripper_width_m",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown FR3 configuration fields: {', '.join(unknown)}")
        return cls(
            host=options.get("host"),
            robot_id=robot_id,
            relative_dynamics_factor=options.get(
                "relative_dynamics_factor",
                0.05,
            ),
            max_joint_step_rad=options.get("max_joint_step_rad", 0.20),
            max_cartesian_step_m=options.get("max_cartesian_step_m", 0.05),
            enable_gripper=options.get("enable_gripper", True),
            gripper_speed_mps=options.get("gripper_speed_mps", 0.02),
            max_gripper_width_m=options.get("max_gripper_width_m", 0.08),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must not be empty")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if (
            isinstance(self.relative_dynamics_factor, bool)
            or not isinstance(self.relative_dynamics_factor, (int, float))
            or not math.isfinite(self.relative_dynamics_factor)
            or not 0 < self.relative_dynamics_factor <= 1
        ):
            raise ValueError("relative_dynamics_factor must be in (0, 1]")
        if not isinstance(self.enable_gripper, bool):
            raise TypeError("enable_gripper must be a boolean")
        for name in (
            "max_joint_step_rad",
            "max_cartesian_step_m",
            "gripper_speed_mps",
            "max_gripper_width_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")


__all__ = ["FR3Config"]
