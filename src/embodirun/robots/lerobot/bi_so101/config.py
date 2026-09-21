"""Configuration for a pair of independently calibrated SO-101 followers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..so101 import SO101Config


@dataclass(frozen=True, slots=True)
class BiSO101Config:
    """Connection and calibration settings for the left and right arms."""

    robot_id: str
    left: SO101Config
    right: SO101Config

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if self.left.port == self.right.port:
            raise ValueError("left_port and right_port must be different")
        left_calibration = (self.left.calibration_dir, self.left.calibration_id or self.left.robot_id)
        right_calibration = (self.right.calibration_dir, self.right.calibration_id or self.right.robot_id)
        if left_calibration == right_calibration:
            raise ValueError("left and right arms must use separate calibration files")

    @classmethod
    def from_mapping(cls, robot_id: str, value: Mapping[str, Any]) -> BiSO101Config:
        """Parse side-specific ports while sharing the SO-101 safety options."""

        options = dict(value)
        left_port = options.pop("left_port", None)
        right_port = options.pop("right_port", None)
        left_id = options.pop("left_calibration_id", f"{robot_id}_left")
        right_id = options.pop("right_calibration_id", f"{robot_id}_right")
        if "port" in options or "calibration_id" in options:
            raise ValueError("use left/right_port and left/right_calibration_id for two arms")
        return cls(
            robot_id=robot_id,
            left=SO101Config.from_mapping(
                f"{robot_id}_left",
                {**options, "port": left_port, "calibration_id": left_id},
            ),
            right=SO101Config.from_mapping(
                f"{robot_id}_right",
                {**options, "port": right_port, "calibration_id": right_id},
            ),
        )


__all__ = ["BiSO101Config"]
