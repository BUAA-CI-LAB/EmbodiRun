"""Map StreamVLN requests and results for the Unitree Go2 embodiment."""

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
from embodirun.robots.unitree.go2.navigation.discrete import (
    NavigationCommand,
    NavigationCommandKind,
)

POLICY_ACTION_SPACE = "streamvln.action_chunk.v1"

_COMMANDS = {
    0: NavigationCommand(NavigationCommandKind.STOP),
    1: NavigationCommand(NavigationCommandKind.MOVE_FORWARD, distance_m=0.25),
    2: NavigationCommand(NavigationCommandKind.TURN_LEFT, angle_deg=15.0),
    3: NavigationCommand(NavigationCommandKind.TURN_RIGHT, angle_deg=15.0),
}


class StreamVLNGo2Mapper:
    """Translate StreamVLN's discrete action protocol for a Go2 target."""

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
        if not isinstance(observation, RobotObservation):
            raise TypeError("StreamVLN observation must be a RobotObservation")
        frames = tuple(frames)
        if len(frames) != 1 or not isinstance(frames[0], CameraFrame):
            raise ValueError("StreamVLN requires exactly one RGB camera frame")
        frame = frames[0]
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state=dict(observation.values),
            images=(ImagePayload(frame.name, frame.mime_type, frame.data),),
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != self.policy_action_space:
            raise ValueError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {self.policy_action_space!r}"
            )
        if len(result.actions) != 1 or result.actions[0].kind != "action_chunk":
            raise ValueError("StreamVLN expects one action_chunk")
        data = result.actions[0].values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or not data:
            raise ValueError("action_chunk data must be a non-empty sequence")
        if len(data) > 4:
            raise ValueError("StreamVLN action_chunk must contain at most four rows")

        actions: list[RobotAction] = []
        for index, raw_row in enumerate(data):
            command = _command(raw_row, index=index)
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values=command.to_action_values(),
                    metadata={
                        "request_id": result.request_id,
                        "session_id": result.session_id,
                        "step_id": result.step_id,
                        "chunk_index": index,
                        "chunk_size": len(data),
                    },
                )
            )
        return tuple(actions)


def _command(value: object, *, index: int) -> NavigationCommand:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"action_chunk row {index} must be a sequence")
    if len(value) != 2:
        raise ValueError(f"action_chunk row {index} must contain two values")
    action_id = _number(value[0], index=index)
    magnitude = _number(value[1], index=index)
    if not action_id.is_integer() or int(action_id) not in _COMMANDS:
        raise ValueError(f"action_chunk row {index} has an unknown action ID")
    command = _COMMANDS[int(action_id)]
    expected = command.distance_m or command.angle_deg
    if not math.isclose(magnitude, expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"action_chunk row {index} has magnitude {magnitude}, expected {expected}")
    return command


def _number(value: object, *, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"action_chunk row {index} must contain numeric values")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"action_chunk row {index} must contain finite values")
    return result


__all__ = ["POLICY_ACTION_SPACE", "StreamVLNGo2Mapper"]
