"""Domain values and API errors used by the Go2 control agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any


class ApiError(Exception):
    """An expected client-facing API failure."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class RobotState:
    """Latest raw SportModeState sample in process-local and wall-clock time."""

    received_at: float
    sequence: int
    position: tuple[float, float, float]
    roll: float
    pitch: float
    yaw: float
    velocity: tuple[float, float, float]
    yaw_rate: float
    received_at_unix: float = 0.0
    mode: int = 0
    gait_type: int = 0
    error_code: int = 0


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def require_number(payload: dict[str, Any], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be finite")
    return result


__all__ = ["ApiError", "RobotState", "normalize_angle", "require_number"]
