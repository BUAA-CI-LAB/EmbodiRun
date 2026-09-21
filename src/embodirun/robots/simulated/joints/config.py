"""Configuration for the deterministic simulated SO-style arm."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

FakeStepLimitMode = Literal["reject", "clip"]
FAKE_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FakeJointsConfig:
    """Finite SO-style joint/gripper state used by the fake adapter."""

    robot_id: str
    initial_joint_positions_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    initial_gripper_position: float = 0.0
    max_joint_step_deg: float = 12.0
    max_gripper_step: float = 20.0
    step_limit_mode: FakeStepLimitMode = "reject"

    @classmethod
    def from_mapping(
        cls,
        robot_id: str,
        value: Mapping[str, Any],
    ) -> FakeJointsConfig:
        if not isinstance(value, Mapping):
            raise TypeError("fake joints configuration options must be a mapping")
        options = dict(value)
        allowed = {
            "initial_joint_positions_deg",
            "initial_gripper_position",
            "max_joint_step_deg",
            "max_gripper_step",
            "step_limit_mode",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError("unknown fake joints configuration fields: " + ", ".join(unknown))
        joints = options.get("initial_joint_positions_deg", cls.initial_joint_positions_deg)
        if isinstance(joints, (str, bytes)) or not isinstance(joints, Sequence):
            raise ValueError("initial_joint_positions_deg must be a sequence")
        if len(joints) != len(FAKE_JOINTS):
            raise ValueError(f"initial_joint_positions_deg must contain {len(FAKE_JOINTS)} values")
        return cls(
            robot_id=robot_id,
            initial_joint_positions_deg=tuple(
                _finite_number(item, f"initial_joint_positions_deg[{index}]") for index, item in enumerate(joints)
            ),
            initial_gripper_position=_finite_number(
                options.get("initial_gripper_position", 0.0),
                "initial_gripper_position",
            ),
            max_joint_step_deg=_finite_number(options.get("max_joint_step_deg", 12.0), "max_joint_step_deg"),
            max_gripper_step=_finite_number(options.get("max_gripper_step", 20.0), "max_gripper_step"),
            step_limit_mode=options.get("step_limit_mode", "reject"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if len(self.initial_joint_positions_deg) != len(FAKE_JOINTS):
            raise ValueError(f"initial_joint_positions_deg must contain {len(FAKE_JOINTS)} values")
        for index, value in enumerate(self.initial_joint_positions_deg):
            number = _finite_number(value, f"initial_joint_positions_deg[{index}]")
            if not -180.0 <= number <= 180.0:
                raise ValueError(f"initial_joint_positions_deg[{index}] must be in [-180, 180]")
        if not 0.0 <= self.initial_gripper_position <= 100.0:
            raise ValueError("initial_gripper_position must be in [0, 100]")
        for name in ("max_joint_step_deg", "max_gripper_step"):
            value = _finite_number(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.step_limit_mode not in {"reject", "clip"}:
            raise ValueError("step_limit_mode must be 'reject' or 'clip'")


__all__ = ["FAKE_JOINTS", "FakeJointsConfig", "FakeStepLimitMode"]
