"""Fail-closed FR3 action execution through Franky."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ...adapter import RobotAction, RobotAdapter, RobotObservation
from .config import FR3Config

FR3_ACTION_SPACE = "franka.fr3.control.v1"


class FR3AdapterError(RuntimeError):
    pass


def _numbers(value: object, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FR3AdapterError(f"{name} must be a sequence")
    if len(value) != length:
        raise FR3AdapterError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise FR3AdapterError(f"{name} must contain finite values")
    return result


class FR3Adapter(RobotAdapter):
    """Synchronous Franky adapter with bounded joint and Cartesian steps."""

    def __init__(self, config: FR3Config, *, franky_module: Any | None = None) -> None:
        if franky_module is None:
            try:
                import franky as franky_module
            except ImportError as error:
                raise FR3AdapterError(
                    "FR3 support requires franky-control; run `uv sync --frozen --no-dev --group robot-fr3`"
                ) from error
        self.config = config
        self.franky = franky_module
        self.robot_id = config.robot_id
        self.robot: Any | None = None
        self.gripper: Any | None = None

    def connect(self) -> None:
        if self.robot is not None:
            return
        robot = self.franky.Robot(self.config.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = self.config.relative_dynamics_factor
        gripper = self.franky.Gripper(self.config.host) if self.config.enable_gripper else None
        self.robot = robot
        self.gripper = gripper

    def _connected_robot(self) -> Any:
        if self.robot is None:
            raise FR3AdapterError("FR3 is not connected")
        return self.robot

    def observe(self) -> RobotObservation:
        state = self._connected_robot().state
        values: dict[str, object] = {
            "joint_positions_rad": list(_numbers(state.q, "state.q", 7)),
        }
        if hasattr(state, "dq"):
            values["joint_velocities_rad_s"] = list(_numbers(state.dq, "state.dq", 7))
        if hasattr(state, "O_T_EE"):
            values["base_to_end_effector"] = list(_numbers(state.O_T_EE, "state.O_T_EE", 16))
        if self.gripper is not None:
            values["gripper_width_m"] = float(self.gripper.width)
        return RobotObservation(
            timestamp_s=time.time(),
            values=values,
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "fr3",
                "action_space": FR3_ACTION_SPACE,
            },
        )

    def execute(self, action: RobotAction) -> None:
        self._connected_robot()
        declared_space = action.metadata.get("action_space")
        if declared_space is not None and declared_space != FR3_ACTION_SPACE:
            raise FR3AdapterError(f"unsupported action space {declared_space!r}; expected {FR3_ACTION_SPACE!r}")
        if not isinstance(action.values, Mapping):
            raise FR3AdapterError("FR3 action values must be an object")
        values = dict(action.values)
        kind = values.pop("type", None)
        if kind == "joint_position":
            self._move_joints(values)
        elif kind == "cartesian_delta":
            self._move_cartesian(values)
        elif kind == "gripper":
            self._move_gripper(values)
        elif kind == "stop":
            self.stop()
        else:
            raise FR3AdapterError(f"unsupported FR3 action type: {kind!r}")

    def _move_joints(self, values: Mapping[str, object]) -> None:
        robot = self._connected_robot()
        target = _numbers(values.get("joint_positions_rad"), "joint_positions_rad", 7)
        current = _numbers(robot.state.q, "state.q", 7)
        maximum = max(abs(next_value - old_value) for old_value, next_value in zip(current, target))
        if maximum > self.config.max_joint_step_rad:
            raise FR3AdapterError(f"joint step {maximum:.6f} exceeds {self.config.max_joint_step_rad:.6f} rad")
        robot.move(self.franky.JointMotion(list(target)))
        if "gripper_width_m" in values:
            self._move_gripper(values)

    def _move_cartesian(self, values: Mapping[str, object]) -> None:
        robot = self._connected_robot()
        delta = _numbers(values.get("translation_m"), "translation_m", 3)
        if max(abs(item) for item in delta) > self.config.max_cartesian_step_m:
            raise FR3AdapterError("Cartesian translation exceeds the configured per-step limit")
        target = self.franky.Affine(list(delta))
        motion = self.franky.CartesianMotion(
            target,
            reference_type=self.franky.ReferenceType.Relative,
        )
        robot.move(motion)

    def _move_gripper(self, values: Mapping[str, object]) -> None:
        if self.gripper is None:
            raise FR3AdapterError("gripper control is disabled")
        width = float(values.get("gripper_width_m"))
        if not math.isfinite(width) or not 0 <= width <= self.config.max_gripper_width_m:
            raise FR3AdapterError("gripper_width_m is outside the configured range")
        if not self.gripper.move(width, self.config.gripper_speed_mps):
            raise FR3AdapterError("Franky gripper did not accept the command")

    def stop(self) -> None:
        robot = self._connected_robot()
        stop = getattr(robot, "stop", None)
        if callable(stop):
            stop()
        else:
            robot.move(self.franky.JointStopMotion())
        if self.gripper is not None:
            self.gripper.stop()

    def close(self) -> None:
        if self.robot is None:
            return
        try:
            if self.gripper is not None:
                close_gripper = getattr(self.gripper, "close", None)
                if callable(close_gripper):
                    close_gripper()
            close_robot = getattr(self.robot, "close", None)
            if callable(close_robot):
                close_robot()
        finally:
            self.gripper = None
            self.robot = None


__all__ = ["FR3_ACTION_SPACE", "FR3Adapter", "FR3AdapterError"]
