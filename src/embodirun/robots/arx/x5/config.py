"""Runtime settings for the physical ARX5 adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ARX5Config:
    can_port: str = "can1"
    robot_type: int = 0
    robot_id: str = "arx5"
    operator_confirmed: bool = False
    max_translation_step_m: float = 0.01
    max_rotation_step_rad: float = 0.05
    max_gripper_step_m: float = 0.01
    gripper_width_min_m: float = 0.0
    gripper_width_max_m: float = 0.088
    sdk_gripper_closed_position: float = 0.0
    sdk_gripper_open_position: float = 5.0
    sdk_module: str = "bimanual"
    sdk_path: str | None = None

    @classmethod
    def from_mapping(
        cls,
        robot_id: str,
        options: Mapping[str, Any],
    ) -> ARX5Config:
        """Build one ARX5 configuration from deployment options."""

        if not isinstance(options, Mapping):
            raise TypeError("ARX5 configuration options must be a mapping")
        values = dict(options)
        allowed = {
            "can_port",
            "robot_type",
            "operator_confirmed",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "max_gripper_step_m",
            "gripper_width_min_m",
            "gripper_width_max_m",
            "sdk_gripper_closed_position",
            "sdk_gripper_open_position",
            "sdk_module",
            "sdk_path",
        }
        unknown = sorted(
            (key for key in values if key not in allowed),
            key=str,
        )
        if unknown:
            raise ValueError(f"unknown ARX5 configuration fields: {', '.join(map(str, unknown))}")
        return cls(robot_id=robot_id, **values)

    def __post_init__(self) -> None:
        if not isinstance(self.can_port, str) or not self.can_port.strip():
            raise ValueError("can_port must not be empty")
        if isinstance(self.robot_type, bool) or not isinstance(self.robot_type, int):
            raise TypeError("robot_type must be an integer")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if not isinstance(self.operator_confirmed, bool):
            raise TypeError("operator_confirmed must be a boolean")
        if not isinstance(self.sdk_module, str) or not self.sdk_module.strip():
            raise ValueError("sdk_module must not be empty")
        if self.sdk_path is not None and (not isinstance(self.sdk_path, str) or not Path(self.sdk_path).is_absolute()):
            raise ValueError("sdk_path must be an absolute path on the control node")
        for name in (
            "max_translation_step_m",
            "max_rotation_step_rad",
            "max_gripper_step_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "gripper_width_min_m",
            "gripper_width_max_m",
            "sdk_gripper_closed_position",
            "sdk_gripper_open_position",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gripper_width_min_m >= self.gripper_width_max_m:
            raise ValueError("gripper_width_min_m must be less than gripper_width_max_m")
        if self.sdk_gripper_closed_position == self.sdk_gripper_open_position:
            raise ValueError("SDK gripper closed and open positions must differ")


__all__ = ["ARX5Config"]
