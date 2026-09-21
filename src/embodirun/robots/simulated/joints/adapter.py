"""Deterministic SO-style simulated robot adapter.

The adapter has the same explicit connect/prepare boundary as a physical
SO-101, but all state is held in memory.  It never imports a serial, CAN, or
vendor SDK and marks observations as simulated so they cannot be mistaken for
hardware evidence.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence

from ...adapter import RobotAction, RobotAdapter, RobotObservation
from .config import FAKE_JOINTS, FakeJointsConfig

FAKE_ACTION_SPACE = "simulated.so101.position.v1"
FAKE_POSITION_FEATURES = tuple(f"{name}.pos" for name in (*FAKE_JOINTS, "gripper"))


class FakeJointsAdapterError(RuntimeError):
    """A simulated robot request is invalid."""


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FakeJointsAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FakeJointsAdapterError(f"{name} must be finite")
    return result


def _numbers(value: object, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FakeJointsAdapterError(f"{name} must be a sequence")
    if len(value) != length:
        raise FakeJointsAdapterError(f"{name} must contain {length} values")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


class FakeJointsAdapter(RobotAdapter):
    """In-memory arm with finite five-joint and gripper state."""

    def __init__(self, config: FakeJointsConfig) -> None:
        self.config = config
        self.robot_id = config.robot_id
        self._lock = threading.RLock()
        self._connected = False
        self._prepared = False
        self._joint_positions = tuple(config.initial_joint_positions_deg)
        self._gripper_position = config.initial_gripper_position

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def prepared(self) -> bool:
        return self._prepared

    def connect(self, *, prepare: bool = True) -> None:
        if not isinstance(prepare, bool):
            raise TypeError("prepare must be a boolean")
        with self._lock:
            self._connected = True
            self._prepared = False
            if prepare:
                self.prepare()

    def prepare(self) -> None:
        with self._lock:
            if not self._connected:
                raise FakeJointsAdapterError("fake joints robot is not connected")
            self._prepared = True

    def _state(self) -> tuple[float, ...]:
        return (*self._joint_positions, self._gripper_position)

    def observe(self) -> RobotObservation:
        with self._lock:
            if not self._connected:
                raise FakeJointsAdapterError("fake joints robot is not connected")
            state = self._state()
        return RobotObservation(
            timestamp_s=time.time(),
            values={
                "joint_positions_deg": list(state[:-1]),
                "gripper_position": state[-1],
            },
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "simulated.so101",
                "action_space": FAKE_ACTION_SPACE,
                "simulated": True,
                "hardware_access": False,
                "state_source": "in_memory_fake",
                "captured_timestamp_ns": time.monotonic_ns(),
                "clock_domain": "host_monotonic_ns",
            },
        )

    def execute(self, action: RobotAction) -> None:
        with self._lock:
            if not self._connected:
                raise FakeJointsAdapterError("fake joints robot is not connected")
            if not self._prepared:
                raise FakeJointsAdapterError("fake joints robot is not prepared")
            declared_space = action.metadata.get("action_space")
            if declared_space is not None and declared_space != FAKE_ACTION_SPACE:
                raise FakeJointsAdapterError(
                    f"unsupported action space {declared_space!r}; expected {FAKE_ACTION_SPACE!r}"
                )
            if not isinstance(action.values, Mapping):
                raise FakeJointsAdapterError("fake joints action values must be an object")
            values = dict(action.values)
            action_type = values.pop("type", None)
            if action_type == "stop":
                if values:
                    raise FakeJointsAdapterError("stop action must not contain parameters")
                return
            if action_type != "joint_position":
                raise FakeJointsAdapterError(f"unsupported fake joints action type: {action_type!r}")
            if set(values) != {"joint_positions_deg", "gripper_position"}:
                raise FakeJointsAdapterError("fake joints action must contain joint_positions_deg and gripper_position")
            target_joints = _numbers(
                values["joint_positions_deg"],
                "joint_positions_deg",
                len(FAKE_JOINTS),
            )
            target_gripper = _number(values["gripper_position"], "gripper_position")
            if any(not -180.0 <= value <= 180.0 for value in target_joints):
                raise FakeJointsAdapterError("joint_positions_deg must be in [-180, 180]")
            if not 0.0 <= target_gripper <= 100.0:
                raise FakeJointsAdapterError("gripper_position must be in [0, 100]")
            joint_deltas = tuple(abs(target - current) for target, current in zip(target_joints, self._joint_positions))
            gripper_delta = abs(target_gripper - self._gripper_position)
            if self.config.step_limit_mode == "reject" and (
                max(joint_deltas, default=0.0) > self.config.max_joint_step_deg
                or gripper_delta > self.config.max_gripper_step
            ):
                raise FakeJointsAdapterError("fake joints action exceeds configured step limits")
            if self.config.step_limit_mode == "clip":
                target_joints = tuple(
                    current
                    + max(
                        -self.config.max_joint_step_deg,
                        min(self.config.max_joint_step_deg, target - current),
                    )
                    for target, current in zip(target_joints, self._joint_positions)
                )
                target_gripper = self._gripper_position + max(
                    -self.config.max_gripper_step,
                    min(
                        self.config.max_gripper_step,
                        target_gripper - self._gripper_position,
                    ),
                )
            self._joint_positions = target_joints
            self._gripper_position = target_gripper

    def stop(self) -> None:
        with self._lock:
            if not self._connected:
                raise FakeJointsAdapterError("fake joints robot is not connected")
            if not self._prepared:
                raise FakeJointsAdapterError("fake joints robot is not prepared")
            # The fake has no motion queue; holding the current values is the
            # complete stop operation and does not claim physical feedback.

    def close(self) -> None:
        with self._lock:
            self._prepared = False
            self._connected = False


__all__ = [
    "FAKE_ACTION_SPACE",
    "FAKE_JOINTS",
    "FAKE_POSITION_FEATURES",
    "FakeJointsAdapter",
    "FakeJointsAdapterError",
]
