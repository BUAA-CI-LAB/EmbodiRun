"""Configuration and fail-closed step limits for a Feetech SO-101 follower."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

StepLimitMode = Literal["reject", "clip"]


@dataclass(frozen=True, slots=True)
class SO101Config:
    """Connection, calibration, and motion-safety settings for one SO-101."""

    port: str
    robot_id: str = "so101"
    calibration_id: str | None = None
    calibration_dir: Path | None = None
    disable_torque_on_disconnect: bool = True
    max_joint_step_deg: float = 12.0
    max_gripper_step: float = 20.0
    step_limit_mode: StepLimitMode = "reject"

    @classmethod
    def from_mapping(
        cls,
        robot_id: str,
        value: Mapping[str, Any],
    ) -> SO101Config:
        """Build one SO-101 configuration from its deployment YAML options."""

        options = dict(value)
        allowed = {
            "port",
            "calibration_id",
            "calibration_dir",
            "disable_torque_on_disconnect",
            "max_joint_step_deg",
            "max_gripper_step",
            "step_limit_mode",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown SO-101 configuration fields: {', '.join(unknown)}")
        calibration_dir = options.get("calibration_dir")
        if calibration_dir is not None and (not isinstance(calibration_dir, str) or not calibration_dir.strip()):
            raise ValueError("calibration_dir must be a non-empty string")
        return cls(
            port=options.get("port"),
            robot_id=robot_id,
            calibration_id=options.get("calibration_id"),
            calibration_dir=Path(calibration_dir) if calibration_dir else None,
            disable_torque_on_disconnect=options.get(
                "disable_torque_on_disconnect",
                True,
            ),
            max_joint_step_deg=options.get("max_joint_step_deg", 12.0),
            max_gripper_step=options.get("max_gripper_step", 20.0),
            step_limit_mode=options.get("step_limit_mode", "reject"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("port must not be empty")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if self.calibration_id is not None and (
            not isinstance(self.calibration_id, str) or not self.calibration_id.strip()
        ):
            raise ValueError("calibration_id must not be empty")
        if self.calibration_dir is not None and not isinstance(self.calibration_dir, Path):
            raise TypeError("calibration_dir must be a Path or None")
        if not isinstance(self.disable_torque_on_disconnect, bool):
            raise TypeError("disable_torque_on_disconnect must be a boolean")
        for name in ("max_joint_step_deg", "max_gripper_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.step_limit_mode, str) or self.step_limit_mode not in {
            "reject",
            "clip",
        }:
            raise ValueError("step_limit_mode must be 'reject' or 'clip'")


__all__ = ["SO101Config", "StepLimitMode"]
