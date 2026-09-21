"""Translate between ARX5 observations and DM0.5 policy values."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from embodirun.model_services import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

from .contract import (
    ACTION_FEATURE_NAMES,
    ACTION_REPRESENTATION,
    POLICY_ACTION_SPACE,
    ROBOT_ACTION_SPACE,
    STATE_DESCRIPTION,
    STATE_REPRESENTATION,
    VIEW_ORDER,
)


class DM05ARX5MapperError(RuntimeError):
    """An observation or action does not match the ARX5 DM0.5 contract."""


@dataclass(frozen=True, slots=True)
class DM05ARX5MapperConfig:
    """Model-output playback settings from the OpenDM ARX5 deployment."""

    expected_horizon: int = 50
    target_steps: int = 25
    gripper_close_threshold_m: float = 0.01
    speed: str | float = "0.5"
    num_steps: int = 10

    def __post_init__(self) -> None:
        for name in ("expected_horizon", "target_steps", "num_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_steps > self.expected_horizon:
            raise ValueError("target_steps must not exceed expected_horizon")
        if (
            isinstance(self.gripper_close_threshold_m, bool)
            or not isinstance(self.gripper_close_threshold_m, (int, float))
            or not math.isfinite(self.gripper_close_threshold_m)
            or self.gripper_close_threshold_m < 0
        ):
            raise ValueError("gripper_close_threshold_m must be finite and non-negative")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (str, int, float)):
            raise TypeError("speed must be a string or number")
        if isinstance(self.speed, str) and not self.speed.strip():
            raise ValueError("speed must not be empty")
        if isinstance(self.speed, (int, float)) and not math.isfinite(self.speed):
            raise ValueError("speed must be finite")


class DM05ARX5Mapper:
    """Map one ARX5 state and two or three cameras to the DM0.5 service."""

    policy_action_space = POLICY_ACTION_SPACE

    def __init__(self, *, config: DM05ARX5MapperConfig | None = None) -> None:
        self.config = config or DM05ARX5MapperConfig()

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
        images, replicated_roles = _model_images(frames)
        state = _state_values(observation.values)
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state={
                "values": list(state),
                "representation": STATE_REPRESENTATION,
                "state_desc": list(STATE_DESCRIPTION),
                "source": observation.metadata.get("state_source", "robot_observation"),
                "units": observation.metadata.get("state_units", "m_rad_robot_gripper_units"),
            },
            images=images,
            reset=False,
            metadata={
                "robot_timestamp_s": observation.timestamp_s,
                "synthetic_views": bool(replicated_roles),
                "physical_view_count": len(frames),
                "replicated_roles": list(replicated_roles),
                "speed": self.config.speed,
                "control_mode": None,
                "num_steps": self.config.num_steps,
            },
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        rows = self._validated_rows(result)
        indices = tuple(
            int(step * (len(rows) - 1) / self.config.target_steps) for step in range(1, self.config.target_steps + 1)
        )
        selected = [list(rows[index]) for index in indices]
        _unwrap_euler_angles(selected)
        for row in selected:
            if row[6] < self.config.gripper_close_threshold_m:
                row[6] = 0.0

        return tuple(
            RobotAction(
                timestamp_s=time.time(),
                values={
                    "type": "eef_xyzrpy_gripper",
                    "eef_xyzrpy_gripper": row,
                },
                metadata={
                    "action_space": ROBOT_ACTION_SPACE,
                    "request_id": result.request_id,
                    "session_id": result.session_id,
                    "step_id": result.step_id,
                    "session_revision": result.session_revision,
                    "policy_revision": result.policy_revision,
                    "chunk_index": index,
                    "chunk_size": len(selected),
                    "source_action_horizon": len(rows),
                    "source_action_index": indices[index],
                    "output_transform_applied": True,
                },
            )
            for index, row in enumerate(selected)
        )

    def _validated_rows(self, result: PolicyResult) -> tuple[tuple[float, ...], ...]:
        if result.action_space != POLICY_ACTION_SPACE:
            raise DM05ARX5MapperError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {POLICY_ACTION_SPACE!r}"
            )
        if len(result.actions) != 1 or result.actions[0].kind != "action_chunk":
            raise DM05ARX5MapperError("DM0.5 result must contain exactly one action_chunk")
        values = result.actions[0].values
        if tuple(values.get("feature_names", ())) != ACTION_FEATURE_NAMES:
            raise DM05ARX5MapperError(f"action feature_names must be {ACTION_FEATURE_NAMES!r}")
        if values.get("representation") != ACTION_REPRESENTATION:
            raise DM05ARX5MapperError(f"action representation must be {ACTION_REPRESENTATION!r}")
        if values.get("output_transform_applied") is not True:
            raise DM05ARX5MapperError("DM0.5 output transform must be applied by EmbodiInfer")
        if values.get("internal_action_dim_exposed", False) is not False:
            raise DM05ARX5MapperError("DM0.5 internal flow state must not cross the service boundary")
        data = values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise DM05ARX5MapperError("action data must be a sequence")
        if len(data) != self.config.expected_horizon:
            raise DM05ARX5MapperError(f"action data must contain {self.config.expected_horizon} rows")
        return tuple(
            _finite_vector(row, f"action row {index}", len(ACTION_FEATURE_NAMES)) for index, row in enumerate(data)
        )


def _state_values(values: Mapping[str, object]) -> tuple[float, ...]:
    return _finite_vector(
        values.get("eef_xyzrpy_gripper"),
        "observation eef_xyzrpy_gripper",
        7,
    )


def _finite_vector(value: object, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise DM05ARX5MapperError(f"{name} must be a sequence")
    if len(value) != length:
        raise DM05ARX5MapperError(f"{name} must contain {length} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise DM05ARX5MapperError(f"{name} values must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError):
            raise DM05ARX5MapperError(f"{name} values must be numeric") from None
        if not math.isfinite(number):
            raise DM05ARX5MapperError(f"{name} values must be finite")
        result.append(number)
    return tuple(result)


def _model_images(
    frames: Sequence[CameraFrame],
) -> tuple[tuple[ImagePayload, ...], tuple[str, ...]]:
    by_name: dict[str, CameraFrame] = {}
    for frame in frames:
        if frame.name not in VIEW_ORDER:
            raise DM05ARX5MapperError(f"unknown DM0.5 camera role {frame.name!r}")
        if frame.name in by_name:
            raise DM05ARX5MapperError(f"duplicate DM0.5 camera role {frame.name!r}")
        by_name[frame.name] = frame
    if set(by_name) == set(VIEW_ORDER):
        replicated_roles: tuple[str, ...] = ()
    elif len(by_name) == 2 and "cam_global" in by_name:
        present_wrist = next((role for role in ("cam_side", "cam_arm") if role in by_name), None)
        if present_wrist is None:
            raise DM05ARX5MapperError("two-camera DM0.5 input requires cam_global and one wrist role")
        missing_wrist = "cam_arm" if present_wrist == "cam_side" else "cam_side"
        by_name[missing_wrist] = by_name[present_wrist]
        replicated_roles = (missing_wrist,)
    else:
        raise DM05ARX5MapperError("DM0.5 requires cam_global plus one wrist camera, or all three roles")
    return (
        tuple(ImagePayload(role, by_name[role].mime_type, by_name[role].data) for role in VIEW_ORDER),
        replicated_roles,
    )


def _unwrap_euler_angles(rows: list[list[float]]) -> None:
    for axis in range(3, 6):
        for index in range(1, len(rows)):
            previous = rows[index - 1][axis]
            value = rows[index][axis]
            delta = (value - previous + math.pi) % (2 * math.pi) - math.pi
            rows[index][axis] = previous + delta


__all__ = [
    "DM05ARX5Mapper",
    "DM05ARX5MapperConfig",
    "DM05ARX5MapperError",
]
