"""Map Pi0.5 requests and action chunks for a Franka Panda EEF environment."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from embodirun.model_services import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
ACTION_DIMENSION = 7


class Pi05FrankaPandaMapper:
    policy_action_space = POLICY_ACTION_SPACE

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
            metadata={"simulator_timestamp_s": observation.timestamp_s},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != self.policy_action_space:
            raise ValueError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {self.policy_action_space!r}"
            )
        if len(result.actions) != 1 or result.actions[0].kind != "action_chunk":
            raise ValueError("Franka Panda expects one action_chunk")
        data = result.actions[0].values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or not data:
            raise ValueError("action_chunk data must be a non-empty sequence")
        actions: list[RobotAction] = []
        for row_index, raw_row in enumerate(data):
            if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
                raise ValueError(f"action_chunk row {row_index} must be a sequence")
            if len(raw_row) != ACTION_DIMENSION:
                raise ValueError(f"action_chunk row {row_index} must contain {ACTION_DIMENSION} values")
            row = tuple(_number(value) for value in raw_row)
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values={"action": row},
                    metadata={
                        "request_id": result.request_id,
                        "session_id": result.session_id,
                        "step_id": result.step_id,
                        "chunk_index": row_index,
                        "chunk_size": len(data),
                    },
                )
            )
        return tuple(actions)


def _number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("action values must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("action values must be numeric") from None
    if not math.isfinite(result):
        raise ValueError("action values must be finite")
    return result


__all__ = ["POLICY_ACTION_SPACE", "Pi05FrankaPandaMapper"]
