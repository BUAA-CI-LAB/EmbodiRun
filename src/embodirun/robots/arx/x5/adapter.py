"""ARX5 robot adapter backed by the vendor ``SingleArm`` SDK."""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from ...adapter import RobotAction, RobotAdapter, RobotObservation
from .config import ARX5Config

ARX5_ACTION_SPACE = "arx.x5.eef_xyzrpy_gripper.absolute.v1"


class ARX5AdapterError(RuntimeError):
    """An ARX5 command or lifecycle operation cannot be completed."""


def _numbers(value: object, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ARX5AdapterError(f"{name} must be a sequence")
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        raise ARX5AdapterError(f"{name} must be a sequence") from None
    if len(items) != length:
        raise ARX5AdapterError(f"{name} must contain {length} values")
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool):
            raise ARX5AdapterError(f"{name}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError):
            raise ARX5AdapterError(f"{name}[{index}] must be numeric") from None
        if not math.isfinite(number):
            raise ARX5AdapterError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _bounded_delta(target: float, current: float, maximum: float) -> float:
    return current + min(max(target - current, -maximum), maximum)


def _bounded_angle(target: float, current: float, maximum: float) -> float:
    delta = (target - current + math.pi) % (2 * math.pi) - math.pi
    bounded = current + min(max(delta, -maximum), maximum)
    return (bounded + math.pi) % (2 * math.pi) - math.pi


class ARX5Adapter(RobotAdapter):
    """Translate normalized absolute EEF rows into vendor SDK commands.

    The adapter is deliberately inert at construction time.  Creating the
    vendor ``SingleArm`` can initialize SocketCAN and enable the motors, so it
    is deferred to :meth:`connect` and requires the explicit configuration
    flag ``operator_confirmed=True``.
    """

    def __init__(
        self,
        config: ARX5Config,
        *,
        single_arm_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.config = config
        self.robot_id = config.robot_id
        self._single_arm_factory = single_arm_factory
        self.arm: Any | None = None
        self.last_executed_rows: list[list[float]] = []
        self.last_executed_sdk_gripper_positions: list[float] = []
        self._stopped = False
        self.lifecycle_state = "disconnected"
        self.closed = False

    def connect(self) -> None:
        """Construct the SDK object only after explicit operator approval."""

        if self.closed:
            raise ARX5AdapterError("ARX5 adapter is closed")
        if self.arm is not None:
            return
        if self.config.operator_confirmed is not True:
            raise ARX5AdapterError(
                "constructing SingleArm may enable motors and enter GO_HOME; operator_confirmed=True is required"
            )
        factory = self._single_arm_factory
        if factory is None:
            # The vendor extension is built separately for the control Python.
            # Host starts an isolated environment, so a shell PYTHONPATH is not
            # a reliable way to locate it. Restore the search path after import.
            previous_path = list(sys.path)
            try:
                if self.config.sdk_path is not None:
                    sys.path.insert(0, self.config.sdk_path)
                module = importlib.import_module(self.config.sdk_module)
                factory = module.SingleArm
            except (ImportError, AttributeError) as error:
                raise ARX5AdapterError(f"could not load ARX5 SDK from {self.config.sdk_module!r}: {error}") from error
            finally:
                sys.path[:] = previous_path
        try:
            self.arm = factory({"can_port": self.config.can_port, "type": self.config.robot_type})
        except Exception:
            # A failed vendor constructor must not leave a half-connected
            # object that later appears usable.
            self.arm = None
            raise
        self._stopped = False
        self.lifecycle_state = "connected"

    def _connected_arm(self) -> Any:
        if self.closed:
            raise ARX5AdapterError("ARX5 adapter is closed")
        if self.arm is None:
            raise ARX5AdapterError("ARX5 adapter is not connected")
        return self.arm

    def _pose_and_gripper(self) -> tuple[float, ...]:
        arm = self._connected_arm()
        pose = _numbers(arm.get_ee_pose_xyzrpy(), "get_ee_pose_xyzrpy", 6)
        joints = _numbers(arm.get_joint_positions(), "get_joint_positions", 7)
        return pose + (self.sdk_gripper_to_width_m(joints[6]),)

    def sdk_gripper_to_width_m(self, position: float) -> float:
        fraction = (position - self.config.sdk_gripper_closed_position) / (
            self.config.sdk_gripper_open_position - self.config.sdk_gripper_closed_position
        )
        width = self.config.gripper_width_min_m + fraction * (
            self.config.gripper_width_max_m - self.config.gripper_width_min_m
        )
        return min(
            max(width, self.config.gripper_width_min_m),
            self.config.gripper_width_max_m,
        )

    def gripper_width_m_to_sdk(self, width_m: float) -> float:
        bounded_width = min(
            max(width_m, self.config.gripper_width_min_m),
            self.config.gripper_width_max_m,
        )
        fraction = (bounded_width - self.config.gripper_width_min_m) / (
            self.config.gripper_width_max_m - self.config.gripper_width_min_m
        )
        return self.config.sdk_gripper_closed_position + fraction * (
            self.config.sdk_gripper_open_position - self.config.sdk_gripper_closed_position
        )

    def observe(self) -> RobotObservation:
        arm = self._connected_arm()
        joints = _numbers(arm.get_joint_positions(), "get_joint_positions", 7)
        velocities = _numbers(arm.get_joint_velocities(), "get_joint_velocities", 7)
        currents = _numbers(arm.get_joint_currents(), "get_joint_currents", 7)
        pose = _numbers(arm.get_ee_pose_xyzrpy(), "get_ee_pose_xyzrpy", 6)
        return RobotObservation(
            timestamp_s=time.time(),
            values={
                "eef_xyzrpy_gripper": list(pose + (self.sdk_gripper_to_width_m(joints[6]),)),
                "joint_positions": list(joints),
                "joint_velocities": list(velocities),
                "joint_currents": list(currents),
                "catch_status": bool(arm.get_catch_status()),
            },
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "arx5",
                "action_space": ARX5_ACTION_SPACE,
                "state_source": "arx_x5_single_arm_sdk",
                "state_units": "m_rad_m",
                "gripper_representation": "opening_width_m",
                "hardware_access": True,
                "can_port": self.config.can_port,
                "sdk_robot_type": self.config.robot_type,
            },
        )

    def _bounded_row(
        self,
        target: object,
        current: tuple[float, ...],
    ) -> list[float]:
        row = _numbers(target, "action row", 7)
        bounded = [_bounded_delta(row[index], current[index], self.config.max_translation_step_m) for index in range(3)]
        bounded.extend(
            _bounded_angle(row[index], current[index], self.config.max_rotation_step_rad) for index in range(3, 6)
        )
        bounded.append(
            min(
                max(
                    _bounded_delta(row[6], current[6], self.config.max_gripper_step_m),
                    self.config.gripper_width_min_m,
                ),
                self.config.gripper_width_max_m,
            )
        )
        return bounded

    def execute(self, action: RobotAction) -> None:
        """Execute one normalized absolute EEF+gripper action row."""

        arm = self._connected_arm()
        if action.metadata.get("action_space") != ARX5_ACTION_SPACE:
            raise ARX5AdapterError(f"unsupported action space {action.metadata.get('action_space')!r}")
        if not isinstance(action.values, Mapping):
            raise ARX5AdapterError("ARX5 action values must be an object")
        if action.values.get("type") != "eef_xyzrpy_gripper":
            raise ARX5AdapterError("ARX5 action type must be 'eef_xyzrpy_gripper'")
        if set(action.values) != {"type", "eef_xyzrpy_gripper"}:
            raise ARX5AdapterError("ARX5 action must contain exactly one normalized EEF command row")
        row = action.values.get("eef_xyzrpy_gripper")
        if isinstance(row, (str, bytes)):
            raise ARX5AdapterError("ARX5 action row must be a numeric sequence")
        if row is None:
            raise ARX5AdapterError("ARX5 action row is missing")

        current = self._pose_and_gripper()
        bounded = self._bounded_row(row, current)
        sdk_gripper_position = self.gripper_width_m_to_sdk(bounded[6])
        self.last_executed_rows = []
        self.last_executed_sdk_gripper_positions = []
        self._stopped = False
        self.lifecycle_state = "active"
        try:
            arm.set_ee_pose_xyzrpy(bounded[:6])
            arm.set_gripper_pos(sdk_gripper_position)
        except Exception:
            with contextlib.suppress(Exception):
                self.stop()
            raise
        self.last_executed_rows.append(bounded)
        self.last_executed_sdk_gripper_positions.append(sdk_gripper_position)

    def stop(self) -> None:
        """Enter the vendor hold mode; disconnected stop is a no-op."""

        if self.closed or self.arm is None or self._stopped:
            return
        try:
            result = self.arm.protect_mode()
        except Exception:
            self.lifecycle_state = "fault"
            raise
        if result is False:
            self.lifecycle_state = "fault"
            raise ARX5AdapterError("ARX5 SDK rejected protect_mode")
        self._stopped = True
        self.lifecycle_state = "holding"

    def close(self) -> None:
        """Stop and release the SDK object, preserving cleanup failures."""

        if self.closed:
            return
        arm = self.arm
        primary_error: Exception | None = None
        if arm is not None:
            try:
                self.stop()
            except Exception as error:
                primary_error = error
            try:
                close = getattr(arm, "close", None)
                if callable(close):
                    close()
            except Exception as error:
                if primary_error is None:
                    primary_error = error
        self.arm = None
        self.closed = True
        self.lifecycle_state = "closed"
        if primary_error is not None:
            raise primary_error

    def __enter__(self) -> ARX5Adapter:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = ["ARX5_ACTION_SPACE", "ARX5Adapter", "ARX5AdapterError"]
