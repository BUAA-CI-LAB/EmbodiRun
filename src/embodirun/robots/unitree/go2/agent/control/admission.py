"""Translate HTTP action payloads into bounded motion admissions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from .config import (
    MAX_LINEAR_SPEED_MPS,
    MAX_MOTION_DURATION_S,
    MAX_MOVE_DISTANCE_M,
    MAX_TURN_ANGLE_RAD,
    MAX_YAW_RATE_RPS,
)
from .types import ApiError, require_number

POSTURE_ACTIONS = frozenset({"stand_up", "stand_down", "balance_stand", "recovery_stand"})


@dataclass(frozen=True)
class MotionAdmission:
    """A fully validated request for a new bounded motion lease."""

    action: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class MotionUpdate:
    """A fully validated velocity update for an existing motion lease."""

    action_id: str
    vx: float
    vy: float
    yaw_rate: float


def require_action(payload: dict[str, Any]) -> str:
    action = payload.get("action")
    if not isinstance(action, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, "action must be a string")
    return action


def _check_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if not minimum <= value <= maximum:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"{name} must be between {minimum} and {maximum}",
        )


def admit_motion(action: str, payload: dict[str, Any]) -> MotionAdmission | None:
    """Return a bounded motion admission, or ``None`` for another action kind."""

    if action in {"move", "stream_move"}:
        vx = require_number(payload, "vx")
        vy = require_number(payload, "vy")
        yaw_rate = require_number(payload, "yaw_rate")
        duration = require_number(payload, "duration_s")
        _check_range("vx", vx, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
        _check_range("vy", vy, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
        _check_range("yaw_rate", yaw_rate, -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS)
        _check_range("duration_s", duration, 0.05, MAX_MOTION_DURATION_S)
        return MotionAdmission(
            action,
            {"vx": vx, "vy": vy, "yaw_rate": yaw_rate, "duration_s": duration},
        )

    if action == "move_distance":
        distance = require_number(payload, "distance_m")
        speed = abs(require_number(payload, "speed_mps"))
        if distance == 0.0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "distance_m must be non-zero")
        _check_range("distance_m", distance, -MAX_MOVE_DISTANCE_M, MAX_MOVE_DISTANCE_M)
        _check_range("speed_mps", speed, 0.05, MAX_LINEAR_SPEED_MPS)
        timeout = min(MAX_MOTION_DURATION_S, abs(distance) / speed + 2.0)
        return MotionAdmission(
            action,
            {
                "vx": math.copysign(speed, distance),
                "vy": 0.0,
                "yaw_rate": 0.0,
                "distance_m": abs(distance),
                "duration_s": timeout,
            },
        )

    if action == "turn":
        angle = require_number(payload, "angle_rad")
        yaw_rate = abs(require_number(payload, "yaw_rate_rps"))
        if angle == 0.0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "angle_rad must be non-zero")
        _check_range("angle_rad", angle, -MAX_TURN_ANGLE_RAD, MAX_TURN_ANGLE_RAD)
        _check_range("yaw_rate_rps", yaw_rate, 0.1, MAX_YAW_RATE_RPS)
        timeout = min(MAX_MOTION_DURATION_S, abs(angle) / yaw_rate + 2.0)
        return MotionAdmission(
            action,
            {
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": math.copysign(yaw_rate, angle),
                "angle_rad": abs(angle),
                "duration_s": timeout,
            },
        )

    return None


def admit_motion_update(payload: dict[str, Any]) -> MotionUpdate:
    action_id = payload.get("action_id")
    if not isinstance(action_id, str) or not action_id:
        raise ApiError(HTTPStatus.BAD_REQUEST, "action_id must be a string")
    vx = require_number(payload, "vx")
    vy = require_number(payload, "vy")
    yaw_rate = require_number(payload, "yaw_rate")
    _check_range("vx", vx, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
    _check_range("vy", vy, -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
    _check_range("yaw_rate", yaw_rate, -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS)
    return MotionUpdate(action_id, vx, vy, yaw_rate)


__all__ = [
    "POSTURE_ACTIONS",
    "MotionAdmission",
    "MotionUpdate",
    "admit_motion",
    "admit_motion_update",
    "require_action",
]
