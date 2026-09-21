"""Pure contract checks for the Astra/pi0.5 cooperative decision.

This module deliberately has no model, image, robot, Deploy-runtime, or driver
imports. It does not perform unit conversion, IK, safety limiting, or action
execution. Those responsibilities stay with the inference and robot adapter
layers at the Deploy boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

HORIZON = 50
ACTION_DIM = 12
MAX_PREFIX = 15
MAX_CORRECTIONS = 5

_PROPOSAL_METADATA = {
    "action_semantics": "biso101_so101_v1",
    "action_layout": "left6_right6",
    "action_encoding": "absolute",
    "action_dim": ACTION_DIM,
}
_DECISION_KEYS = {
    "proposal_id",
    "observation_id",
    "decision",
    "execute_steps",
    "corrections",
    "reason",
}
_CORRECTION_KEYS = {"left", "right", "frame", "units", "duration_s"}
_WAYPOINT_KEYS = {
    "reach_m",
    "height_m",
    "pan_deg",
    "wrist_flex_deg",
    "wrist_roll_deg",
    "gripper",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _rows(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != HORIZON:
        raise ValueError("proposal actions must be a 50-step sequence")
    result: list[tuple[float, ...]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != ACTION_DIM:
            raise ValueError(f"proposal actions[{index}] must contain 12 values")
        if not all(_finite_number(item) for item in row):
            raise ValueError(f"proposal actions[{index}] contains a non-finite value")
        result.append(tuple(float(item) for item in row))
    return tuple(result)


def validate_proposal(actions: Any, metadata: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    """Validate and copy one reviewable 50x12 BiSO101 proposal.

    The returned tuple is a value-only snapshot. It is still a proposal, not an
    indication that a robot accepted or executed any row.
    """

    metadata = _mapping(metadata, "proposal metadata")
    for key, expected in _PROPOSAL_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(f"proposal metadata {key} must be {expected!r}")
    if metadata.get("joint_position_unit") not in {"degrees", "range_m100_100"}:
        raise ValueError("proposal metadata joint_position_unit must explicitly declare degrees or range_m100_100")
    if "horizon" in metadata and metadata["horizon"] != HORIZON:
        raise ValueError(f"proposal metadata horizon must be {HORIZON}")
    return _rows(actions)


def _validate_waypoint(value: Any, label: str) -> None:
    waypoint = _mapping(value, label)
    if set(waypoint) != _WAYPOINT_KEYS or not all(_finite_number(waypoint[key]) for key in _WAYPOINT_KEYS):
        raise ValueError(f"{label} is invalid")


def _validate_correction(value: Any) -> None:
    correction = _mapping(value, "correction")
    if set(correction) != _CORRECTION_KEYS:
        raise ValueError("correction has unexpected or missing keys")
    if correction["frame"] != "so101_shoulder_plane":
        raise ValueError("correction frame must be so101_shoulder_plane")
    if correction["units"] != "m_deg":
        raise ValueError("correction units must be m_deg")
    if not _finite_number(correction["duration_s"]) or correction["duration_s"] <= 0:
        raise ValueError("correction duration_s must be positive and finite")
    if correction["left"] is None and correction["right"] is None:
        raise ValueError("correction must target at least one arm")
    for side in ("left", "right"):
        if correction[side] is not None:
            _validate_waypoint(correction[side], f"correction.{side}")


def validate_decision(raw: Mapping[str, Any], *, proposal_id: str, observation_id: str) -> dict[str, Any]:
    """Validate one Astra branch and return a deep value copy.

    This function checks branch shape and declared units only. The correction
    waypoints still require a Deploy-side capability check and conversion before
    they can become joint actions.
    """

    result = _mapping(raw, "Astra decision")
    if set(result) != _DECISION_KEYS:
        raise ValueError("Astra decision has unexpected or missing keys")
    if result["proposal_id"] != proposal_id:
        raise ValueError("Astra decision does not match the current proposal")
    if result["observation_id"] != observation_id:
        raise ValueError("Astra decision does not match the current observation")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("Astra decision reason must be non-empty")
    steps = result["execute_steps"]
    corrections = result["corrections"]
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("Astra decision execute_steps must be an integer")
    if not isinstance(corrections, list):
        raise ValueError("Astra decision corrections must be a list")

    kind = result["decision"]
    if kind == "execute_prefix":
        if not 1 <= steps <= MAX_PREFIX or corrections:
            raise ValueError("execute_prefix requires 1-15 steps and no corrections")
    elif kind == "correct":
        if steps != 0 or not 1 <= len(corrections) <= MAX_CORRECTIONS:
            raise ValueError("correct requires zero steps and 1-5 corrections")
        for correction in corrections:
            _validate_correction(correction)
    elif kind == "hold":
        if steps != 0 or corrections:
            raise ValueError("hold requires zero steps and no corrections")
    else:
        raise ValueError("Astra decision kind is invalid")
    return deepcopy(dict(result))
