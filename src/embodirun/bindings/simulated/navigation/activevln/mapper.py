"""Translate typed actions, never model text, at the simulator boundary."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from embodirun.model_services import ImagePayload, PolicyObservation, PolicyResult
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame


class ActiveVLNSimulatorMapper:
    """Expand complete model chunks without sending evaluator state to inference."""

    policy_action_space = "activevln.r2r.discrete.v1"

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
        """Send RGB and instruction only; goal distance and pose remain evaluator-owned."""
        if len(frames) != 1 or frames[0].name != "observation.images.rgb":
            raise ValueError("ActiveVLN requires one observation.images.rgb frame")
        frame = frames[0]
        return PolicyObservation(
            session_id, request_id, step_id, instruction, {}, (ImagePayload(frame.name, frame.mime_type, frame.data),)
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        """Reject malformed chunks before returning any executable primitive."""
        if result.action_space != self.policy_action_space:
            raise ValueError("ActiveVLN action space mismatch")
        if len(result.actions) != 1 or result.actions[0].kind != "discrete_chunk":
            raise ValueError("expected one ActiveVLN discrete_chunk")
        values = result.actions[0].values
        if values.get("parsed_valid") is not True:
            raise ValueError("ActiveVLN output is not a valid action chunk")
        rows = values.get("rows")
        if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 3:
            raise ValueError("ActiveVLN expects one to three rows")
        actions: list[RobotAction] = []
        ended = False
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("each ActiveVLN row must contain action ID and magnitude")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
                raise ValueError("ActiveVLN row values must be finite numbers")
            kind, magnitude = row
            if kind == -1 and magnitude == 0:
                ended = True
                continue
            if ended:
                raise ValueError("an action cannot follow STOP or padding")
            if kind == 0 and magnitude == 0:
                commands = [{"command": "stop"}]
                ended = True
            elif kind in (1, 2, 3):
                unit = 25 if kind == 1 else 15
                count = magnitude / unit
                if not count.is_integer() or not 1 <= count <= 24:
                    raise ValueError("action magnitude must be a bounded multiple of 25 cm or 15 degrees")
                command = (
                    {"command": "move_forward", "distance_m": 0.25}
                    if kind == 1
                    else {"command": "turn_left" if kind == 2 else "turn_right", "angle_deg": 15.0}
                )
                commands = [command] * int(count)
            else:
                raise ValueError("unknown ActiveVLN action row")
            actions.extend(
                RobotAction(time.time(), dict(command), {"request_id": result.request_id}) for command in commands
            )
        if not actions:
            raise ValueError("ActiveVLN returned no actions")
        return tuple(actions)
