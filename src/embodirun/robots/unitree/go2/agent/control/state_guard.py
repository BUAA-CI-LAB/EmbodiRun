"""Robot-state freshness and motion-readiness checks."""

from __future__ import annotations

import math
import time
from http import HTTPStatus

from .config import MAX_TILT_RAD, STATE_MAX_AGE_S
from .transport import RobotTransport
from .types import ApiError, RobotState


def require_fresh_state(transport: RobotTransport) -> RobotState:
    state = transport.state()
    if state is None:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "robot state is unavailable")
    if (
        isinstance(state.sequence, bool)
        or not isinstance(state.sequence, int)
        or state.sequence < 1
        or not math.isfinite(state.received_at)
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "robot state has an invalid sequence or timestamp",
        )
    state_age_s = time.monotonic() - state.received_at
    if state_age_s < 0.0 or state_age_s > STATE_MAX_AGE_S:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "robot state is stale")
    return state


def require_motion_ready(
    transport: RobotTransport,
    operator_motion_ready: bool,
) -> RobotState:
    if not operator_motion_ready:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "operator motion-ready interlock is not enabled",
        )
    state = require_fresh_state(transport)
    numeric_state = (
        *state.position,
        state.roll,
        state.pitch,
        state.yaw,
        *state.velocity,
        state.yaw_rate,
    )
    if len(state.position) < 2 or len(state.velocity) < 2 or not all(math.isfinite(value) for value in numeric_state):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "robot state contains invalid numeric data",
        )
    if state.error_code != 0:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"robot reports error code {state.error_code}",
        )
    if abs(state.roll) > MAX_TILT_RAD or abs(state.pitch) > MAX_TILT_RAD:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "robot tilt exceeds the motion safety limit",
        )
    return state


__all__ = ["require_fresh_state", "require_motion_ready"]
