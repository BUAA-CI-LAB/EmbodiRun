"""Translate between FR3 observations and Pi0.5 policy values."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from embodirun.model_services import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.franka.fr3 import FR3_ACTION_SPACE
from embodirun.robots.sensors.cameras import CameraFrame

POLICY_ACTION_SPACE = "pi05.action_chunk.v1"


class Pi05FR3MapperError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pi05FR3MapperConfig:
    joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    gripper_index: int | None = None

    def __post_init__(self) -> None:
        if not self.joint_indices:
            raise Pi05FR3MapperError("joint_indices must not be empty")
        for index in self.joint_indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 31:
                raise Pi05FR3MapperError("joint_indices must be ints in [0,31]")
        if self.gripper_index is not None and (
            isinstance(self.gripper_index, bool)
            or not isinstance(self.gripper_index, int)
            or not 0 <= self.gripper_index <= 31
        ):
            raise Pi05FR3MapperError("gripper_index must be an int in [0,31] or None")


class Pi05FR3Mapper:
    """Translate between FR3 observations and Pi0.5 policy values."""

    policy_action_space = POLICY_ACTION_SPACE

    def __init__(self, *, config: Pi05FR3MapperConfig | None = None) -> None:
        self.config = config or Pi05FR3MapperConfig()

    def map_observation(
        self,
        observation: RobotObservation,
        *,
        session_id: str,
        request_id: str,
        step_id: int,
        instruction: str,
        frames: Sequence[CameraFrame],
    ) -> PolicyObservation:
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state=dict(observation.values),
            images=tuple(ImagePayload(frame.name, frame.mime_type, frame.data) for frame in frames),
            reset=False,
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != POLICY_ACTION_SPACE:
            raise Pi05FR3MapperError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {self.policy_action_space!r}"
            )
        if not result.actions:
            raise Pi05FR3MapperError("result contains no action rows")
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise Pi05FR3MapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise Pi05FR3MapperError("action_chunk values.data must be a sequence")
        if not data:
            raise Pi05FR3MapperError("action_chunk values.data must not be empty")
        indices = self.config.joint_indices + (
            (self.config.gripper_index,) if self.config.gripper_index is not None else ()
        )
        max_index = max(indices)
        actions: list[RobotAction] = []
        for row_index, raw_row in enumerate(data):
            if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
                raise Pi05FR3MapperError(f"action_chunk row {row_index} must be a numeric sequence")
            row: list[float] = []
            for item in raw_row:
                if isinstance(item, bool):
                    raise Pi05FR3MapperError("action values must be numeric")
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    raise Pi05FR3MapperError("action values must be numeric") from None
                if not math.isfinite(value):
                    raise Pi05FR3MapperError("action values must be finite")
                row.append(value)
            if len(row) <= max_index:
                raise Pi05FR3MapperError(f"action_chunk row {row_index} is too short for configured indices")

            robot = {
                "type": "joint_position",
                "joint_positions_rad": [row[index] for index in self.config.joint_indices],
            }
            if self.config.gripper_index is not None:
                robot["gripper_width_m"] = row[self.config.gripper_index]
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values=robot,
                    metadata={
                        "action_space": FR3_ACTION_SPACE,
                        "request_id": result.request_id,
                        "session_id": result.session_id,
                        "step_id": result.step_id,
                        "session_revision": result.session_revision,
                        "chunk_index": row_index,
                        "chunk_size": len(data),
                    },
                )
            )
        return tuple(actions)


__all__ = [
    "POLICY_ACTION_SPACE",
    "Pi05FR3Mapper",
    "Pi05FR3MapperConfig",
    "Pi05FR3MapperError",
]
