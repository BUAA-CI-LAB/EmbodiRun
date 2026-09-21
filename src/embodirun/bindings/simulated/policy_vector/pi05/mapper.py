"""Map Pi0.5 chunks to a unit-preserving simulated state-vector robot."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence

from embodirun.model_services import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.robots.simulated.policy_vector import (
    POLICY_VECTOR_ACTION_SPACE,
    STATE_FIELD,
    UNVERIFIED_UNITS,
)

POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
MAXIMUM_CHUNK_STEPS = 50
SO101_POLICY_FEATURE_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
REQUIRED_IMAGE_NAMES = (
    "observation.images.front",
    "observation.images.wrist",
)


class SimulatedPolicyVectorPi05MapperError(RuntimeError):
    """A model result or simulated observation violates the binding contract."""


def _vector(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SimulatedPolicyVectorPi05MapperError(f"{name} must be a sequence")
    if len(value) != len(SO101_POLICY_FEATURE_NAMES):
        raise SimulatedPolicyVectorPi05MapperError(f"{name} must contain {len(SO101_POLICY_FEATURE_NAMES)} values")
    numbers: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise SimulatedPolicyVectorPi05MapperError(f"{name}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError):
            raise SimulatedPolicyVectorPi05MapperError(f"{name}[{index}] must be numeric") from None
        if not math.isfinite(number):
            raise SimulatedPolicyVectorPi05MapperError(f"{name}[{index}] must be finite")
        numbers.append(number)
    return tuple(numbers)


class SimulatedPolicyVectorPi05Mapper:
    """Keep model vectors unchanged while adapting transport contracts."""

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
        if not isinstance(observation.values, Mapping):
            raise SimulatedPolicyVectorPi05MapperError("simulated policy-vector observation values must be an object")
        metadata = observation.metadata
        if metadata.get("simulated") is not True or metadata.get("hardware_access") is not False:
            raise SimulatedPolicyVectorPi05MapperError("policy-vector binding accepts simulated observations only")
        if metadata.get("units") != UNVERIFIED_UNITS:
            raise SimulatedPolicyVectorPi05MapperError(
                f"policy-vector observation must declare units {UNVERIFIED_UNITS!r}"
            )
        state = list(_vector(observation.values.get(STATE_FIELD), STATE_FIELD))
        by_name: dict[str, CameraFrame] = {}
        for frame in frames:
            if frame.name in by_name:
                raise SimulatedPolicyVectorPi05MapperError(f"duplicate camera frame {frame.name!r}")
            by_name[frame.name] = frame
        missing = [name for name in REQUIRED_IMAGE_NAMES if name not in by_name]
        if missing:
            raise SimulatedPolicyVectorPi05MapperError(
                "simulated policy-vector input is missing camera frames: " + ", ".join(missing)
            )
        selected = tuple(by_name[name] for name in REQUIRED_IMAGE_NAMES)
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state={STATE_FIELD: state},
            images=tuple(ImagePayload(frame.name, frame.mime_type, frame.data) for frame in selected),
            metadata={
                "robot_timestamp_s": observation.timestamp_s,
                "units": UNVERIFIED_UNITS,
                "simulated": True,
                "hardware_access": False,
            },
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != self.policy_action_space:
            raise SimulatedPolicyVectorPi05MapperError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {self.policy_action_space!r}"
            )
        if len(result.actions) != 1:
            raise SimulatedPolicyVectorPi05MapperError(
                "simulated policy-vector binding expects exactly one policy action"
            )
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise SimulatedPolicyVectorPi05MapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise SimulatedPolicyVectorPi05MapperError("action_chunk values.data must be a sequence")
        if len(data) != MAXIMUM_CHUNK_STEPS:
            raise SimulatedPolicyVectorPi05MapperError(
                f"simulated policy-vector binding expects exactly {MAXIMUM_CHUNK_STEPS} action rows"
            )
        declared = raw.values.get("feature_names")
        if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
            raise SimulatedPolicyVectorPi05MapperError("action_chunk feature_names must be the six SO-101 names")
        names = tuple(declared)
        if names != SO101_POLICY_FEATURE_NAMES:
            raise SimulatedPolicyVectorPi05MapperError(
                f"action_chunk feature_names must exactly match the six SO-101 names; got {names!r}"
            )
        actions: list[RobotAction] = []
        for row_index, row in enumerate(data):
            values = _vector(row, f"action_chunk row {row_index}")
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values={
                        "type": "state_native_vector",
                        STATE_FIELD: list(values),
                    },
                    metadata={
                        "action_space": POLICY_VECTOR_ACTION_SPACE,
                        "units": UNVERIFIED_UNITS,
                        "simulated": True,
                        "hardware_access": False,
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
    "MAXIMUM_CHUNK_STEPS",
    "POLICY_ACTION_SPACE",
    "REQUIRED_IMAGE_NAMES",
    "SO101_POLICY_FEATURE_NAMES",
    "SimulatedPolicyVectorPi05Mapper",
    "SimulatedPolicyVectorPi05MapperError",
]
