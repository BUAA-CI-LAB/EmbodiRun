"""Fail-closed composition of two calibrated SO-101 adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from embodirun.robots import RobotAction, RobotAdapter, RobotObservation

from ..so101 import (
    SO101_ACTION_SPACE,
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
)
from .config import BiSO101Config

BI_SO101_ACTION_SPACE = "lerobot.bi_so101.position.v1"
BI_SO101_POSITION_FEATURES = tuple(
    f"{side}_arm_{feature}" for side in ("left", "right") for feature in SO101_POSITION_FEATURES
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SO101AdapterError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SO101AdapterError(f"{label} must be numeric") from None
    if not math.isfinite(number):
        raise SO101AdapterError(f"{label} must be finite")
    return number


def _arm_values(value: object, side: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "joint_positions_deg",
        "gripper_position",
    }:
        raise SO101AdapterError(f"{side} target requires joints and gripper")
    joints = value["joint_positions_deg"]
    if isinstance(joints, (str, bytes)) or not isinstance(joints, Sequence):
        raise SO101AdapterError(f"{side}.joint_positions_deg must be a sequence")
    if len(joints) != 5:
        raise SO101AdapterError(f"{side}.joint_positions_deg must contain 5 values")
    values = [_finite_number(item, f"{side}.joint_positions_deg[{index}]") for index, item in enumerate(joints)]
    gripper = _finite_number(value["gripper_position"], f"{side}.gripper_position")
    if not 0.0 <= gripper <= 100.0:
        raise SO101AdapterError(f"{side}.gripper_position must be in [0, 100]")
    return {"joint_positions_deg": values, "gripper_position": gripper}


class BiSO101Adapter(RobotAdapter):
    """Own two serial buses and execute each accepted command sequentially.

    Shape and scalar validation happens before either child receives a write.  If
    the second write fails, both children receive a measured hold attempt and the
    original failure remains visible to the caller.
    """

    def __init__(
        self,
        config: BiSO101Config,
        *,
        left: SO101Adapter | None = None,
        right: SO101Adapter | None = None,
    ) -> None:
        self.config = config
        self.robot_id = config.robot_id
        self.left = left if left is not None else SO101Adapter(config.left)
        self.right = right if right is not None else SO101Adapter(config.right)

    def _call_both(self, method: str) -> None:
        errors: list[str] = []
        for side, arm in (("left", self.left), ("right", self.right)):
            try:
                getattr(arm, method)()
            except Exception as error:
                errors.append(f"{side}: {error}")
        if errors:
            raise SO101AdapterError(f"dual SO-101 {method} failed: {'; '.join(errors)}")

    def connect(self, *, prepare: bool = True) -> None:
        """Connect left then right; clean up a partial connection on failure."""

        try:
            self.left.connect(prepare=prepare)
            self.right.connect(prepare=prepare)
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise SO101AdapterError(
                    f"dual SO-101 connect failed ({error}); cleanup failed ({cleanup_error})"
                ) from error
            raise

    def observe(self) -> RobotObservation:
        """Read both arms and expose one ordered twelve-value observation."""

        left = self.left.observe()
        right = self.right.observe()
        left_values = left.values
        right_values = right.values
        if not isinstance(left_values, Mapping) or not isinstance(right_values, Mapping):
            raise SO101AdapterError("SO-101 observation values must be objects")
        positions = (
            *left_values["joint_positions_deg"],
            left_values["gripper_position"],
            *right_values["joint_positions_deg"],
            right_values["gripper_position"],
        )
        return RobotObservation(
            timestamp_s=min(left.timestamp_s, right.timestamp_s),
            values=dict(zip(BI_SO101_POSITION_FEATURES, positions)),
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "bi_so101_follower",
                "action_space": BI_SO101_ACTION_SPACE,
                "position_units": "degrees_and_normalized_gripper",
                "arm_timestamps_s": {
                    "left": left.timestamp_s,
                    "right": right.timestamp_s,
                },
                "arm_sample_skew_s": abs(left.timestamp_s - right.timestamp_s),
            },
        )

    def execute(self, action: RobotAction) -> None:
        """Validate both targets, then command left and right in that order."""

        space = action.metadata.get("action_space")
        if space is not None and space != BI_SO101_ACTION_SPACE:
            raise SO101AdapterError(f"unsupported dual SO-101 action space: {space!r}")
        if not isinstance(action.values, Mapping):
            raise SO101AdapterError("dual SO-101 action values must be an object")
        if action.values == {"type": "stop"}:
            self.stop()
            return
        if action.values.get("type") != "joint_position" or set(action.values) != {
            "type",
            "left",
            "right",
        }:
            raise SO101AdapterError("dual SO-101 joint_position requires left and right targets")
        left_values = _arm_values(action.values["left"], "left")
        right_values = _arm_values(action.values["right"], "right")
        actions = [
            RobotAction(
                timestamp_s=action.timestamp_s,
                values={"type": "joint_position", **left_values},
                metadata={**action.metadata, "action_space": SO101_ACTION_SPACE},
            ),
            RobotAction(
                timestamp_s=action.timestamp_s,
                values={"type": "joint_position", **right_values},
                metadata={**action.metadata, "action_space": SO101_ACTION_SPACE},
            ),
        ]
        self.left.validate_action(actions[0])
        self.right.validate_action(actions[1])
        try:
            self.left.execute(actions[0])
            self.right.execute(actions[1])
        except Exception as error:
            try:
                self.stop()
            except Exception as stop_error:
                raise SO101AdapterError(
                    f"dual SO-101 execute failed ({error}); hold failed ({stop_error})"
                ) from stop_error
            raise

    def stop(self) -> None:
        """Attempt a measured hold on both arms, including after one failure."""

        self._call_both("stop")

    def close(self) -> None:
        """Close both arms and report all side failures after attempting both."""

        self._call_both("close")


__all__ = [
    "BI_SO101_ACTION_SPACE",
    "BI_SO101_POSITION_FEATURES",
    "BiSO101Adapter",
]
