"""Coordinate action admission, motion leases, posture, and shutdown."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from typing import Any, ClassVar

from .admission import (
    POSTURE_ACTIONS,
    MotionAdmission,
    MotionUpdate,
    admit_motion,
    admit_motion_update,
    require_action,
)
from .config import (
    MAX_LINEAR_SPEED_MPS,
    MAX_MOTION_DURATION_S,
    MAX_MOVE_DISTANCE_M,
    MAX_TILT_RAD,
    MAX_TURN_ANGLE_RAD,
    MAX_YAW_RATE_RPS,
    STATE_MAX_AGE_S,
    WORKER_STOP_JOIN_TIMEOUT_S,
)
from .lease import MotionLease
from .state_guard import require_fresh_state, require_motion_ready
from .transport import RobotTransport
from .types import ApiError, RobotState


class ActionExecutor:
    """Validate API actions and execute at most one bounded motion at a time."""

    POSTURE_ACTIONS: ClassVar[frozenset[str]] = POSTURE_ACTIONS

    def __init__(
        self,
        transport: RobotTransport,
        *,
        operator_motion_ready: bool = False,
    ) -> None:
        self.transport = transport
        # This is an explicit operator interlock. Do not infer readiness from
        # undocumented numeric gait or mode values.
        self.operator_motion_ready = bool(operator_motion_ready)
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._active: MotionLease | None = None
        self._last_result: dict[str, Any] | None = None
        self._closing = False
        self._closed = False

    def snapshot(self) -> dict[str, Any]:
        state = self.transport.state()
        with self._lock:
            active = self._active.snapshot() if self._active else None
            result = dict(self._last_result) if self._last_result else None
            operator_motion_ready = self.operator_motion_ready
            closing = self._closing
        state_json = None
        state_age_s = None
        if state is not None:
            state_age_s = max(0.0, time.monotonic() - state.received_at)
            state_json = asdict(state)
        return {
            "transport": self.transport.name,
            "operator_motion_ready": operator_motion_ready,
            "closing": closing,
            "robot_state_available": state is not None,
            "robot_state_fresh": state_age_s is not None and state_age_s <= STATE_MAX_AGE_S,
            "robot_state_age_s": round(state_age_s, 3) if state_age_s is not None else None,
            "robot_state": state_json,
            "active_action": active,
            "last_result": result,
            "limits": {
                "linear_speed_mps": MAX_LINEAR_SPEED_MPS,
                "yaw_rate_rps": MAX_YAW_RATE_RPS,
                "move_distance_m": MAX_MOVE_DISTANCE_M,
                "turn_angle_rad": MAX_TURN_ANGLE_RAD,
                "motion_duration_s": MAX_MOTION_DURATION_S,
                "tilt_rad": MAX_TILT_RAD,
            },
        }

    def execute(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        action = require_action(payload)
        if action == "stop":
            return HTTPStatus.OK, self.stop("api_stop")
        if action in self.POSTURE_ACTIONS:
            return HTTPStatus.OK, self._execute_posture(action)
        if action == "update_move":
            update = admit_motion_update(payload)
            with self._lifecycle_lock:
                self._require_not_closing()
                self._require_motion_ready()
                return HTTPStatus.OK, self._update_move(update)

        admission = admit_motion(action, payload)
        if admission is not None:
            with self._lifecycle_lock:
                self._require_not_closing()
                self._require_motion_ready()
                return HTTPStatus.ACCEPTED, self._start_motion(admission)
        raise ApiError(HTTPStatus.BAD_REQUEST, f"unsupported action: {action}")

    def _require_not_closing(self) -> None:
        with self._lock:
            if self._closing or self._closed:
                raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "executor is closing")

    def _require_fresh_state(self) -> RobotState:
        return require_fresh_state(self.transport)

    def _require_motion_ready(self) -> RobotState:
        with self._lock:
            operator_motion_ready = self.operator_motion_ready
        return require_motion_ready(self.transport, operator_motion_ready)

    def _execute_posture(self, action: str) -> dict[str, Any]:
        with self._lifecycle_lock:
            self._require_not_closing()
            if not self.operator_motion_ready:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "operator motion-ready interlock is not enabled",
                )
            stop_result = self._cancel_and_stop_locked("posture_change")
            if not stop_result["worker_joined"] or not stop_result["stopped"]:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "could not confirm cancellation before posture change",
                )
            self._require_fresh_state()
            # A posture RPC can have an uncertain outcome on timeout, so revoke
            # the operator attestation before sending it.
            with self._lock:
                self.operator_motion_ready = False
            with self._command_lock:
                code = self.transport.posture(action)
            if code != 0:
                raise ApiError(HTTPStatus.BAD_GATEWAY, f"SDK returned code {code}")
            return {
                "accepted": True,
                "action": action,
                "sdk_code": code,
                "operator_motion_ready": False,
            }

    def _start_motion(self, admission: MotionAdmission) -> dict[str, Any]:
        with self._lock:
            if self._worker and self._worker.is_alive():
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "another motion is active; call stop first",
                )
            lease = MotionLease(admission)
            worker = threading.Thread(
                target=self._motion_worker,
                args=(lease,),
                daemon=True,
                name=f"go2-{lease.action}",
            )
            lease.worker = worker
            self._active = lease
            # Keep these aliases for diagnostics and race-oriented tests.
            self._stop_event = lease.stop_event
            self._worker = worker
            worker.start()
            return {"accepted": True, "action_id": lease.id, "action": lease.action}

    def _update_move(self, update: MotionUpdate) -> dict[str, Any]:
        with self._lock:
            lease = self._active
            if lease is None or not lease.is_alive():
                raise ApiError(HTTPStatus.CONFLICT, "no timed move is active")
            return lease.update(update)

    def _motion_worker(self, lease: MotionLease) -> None:
        result = lease.run(
            self.transport,
            self._command_lock,
            self._require_motion_ready,
        )
        with self._lock:
            self._last_result = result
            if self._active is lease:
                self._active = None

    def _cancel_and_stop_locked(self, reason: str) -> dict[str, Any]:
        with self._lock:
            active_id = self._active.id if self._active else None
            stop_event = self._stop_event
            worker = self._worker
            stop_event.set()

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=WORKER_STOP_JOIN_TIMEOUT_S)
        worker_joined = worker is None or not worker.is_alive()

        stop_error = None
        try:
            with self._command_lock:
                code = self.transport.stop()
        except Exception as exc:  # noqa: BLE001 - emergency stop returns structured status
            code = -1
            stop_error = str(exc)
        return {
            "stopped": code == 0 and worker_joined,
            "reason": reason,
            "cancelled_action_id": active_id,
            "sdk_code": code,
            "worker_joined": worker_joined,
            "error": stop_error,
        }

    def stop(self, reason: str = "stop") -> dict[str, Any]:
        with self._lifecycle_lock:
            result = self._cancel_and_stop_locked(reason)
            # Re-arming requires an intentional process restart with
            # --operator-ready; there is no remote re-arm endpoint.
            with self._lock:
                self.operator_motion_ready = False
            return {**result, "operator_motion_ready": False}

    def close(self, send_stop: bool = True) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    return
                self._closing = True
            try:
                reason = "close" if send_stop else "close_after_signal"
                self._cancel_and_stop_locked(reason)
                with self._lock:
                    self.operator_motion_ready = False
            finally:
                try:
                    self.transport.close()
                finally:
                    with self._lock:
                        self._closed = True


__all__ = ["ActionExecutor"]
