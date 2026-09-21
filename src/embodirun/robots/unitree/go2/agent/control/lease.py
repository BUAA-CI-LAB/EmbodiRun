"""Mutable motion lease state and its bounded control-loop execution."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from .admission import MotionAdmission, MotionUpdate
from .config import CONTROL_PERIOD_S, STREAM_HEARTBEAT_TIMEOUT_S
from .transport import RobotTransport
from .types import ApiError, RobotState, normalize_angle


class MotionLease:
    """Own one action's deadline, heartbeat, velocity and cancellation state."""

    def __init__(self, admission: MotionAdmission) -> None:
        self.id = uuid.uuid4().hex
        self.action = admission.action
        self._parameters = dict(admission.parameters)
        self._lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.started_at_unix = time.time()
        self.updated_at_unix: float | None = None
        self.heartbeat_at_monotonic: float | None = time.monotonic() if self.action == "stream_move" else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value: dict[str, Any] = {
                "id": self.id,
                "action": self.action,
                "parameters": dict(self._parameters),
                "started_at_unix": self.started_at_unix,
            }
            if self.heartbeat_at_monotonic is not None:
                value["heartbeat_at_monotonic"] = self.heartbeat_at_monotonic
            if self.updated_at_unix is not None:
                value["updated_at_unix"] = self.updated_at_unix
            return value

    def update(self, update: MotionUpdate) -> dict[str, Any]:
        with self._lock:
            if update.action_id != self.id:
                raise ApiError(HTTPStatus.CONFLICT, "action_id is not active")
            if self.action not in {"move", "stream_move"}:
                raise ApiError(HTTPStatus.CONFLICT, "only a timed move can be updated")
            self._parameters.update(
                vx=update.vx,
                vy=update.vy,
                yaw_rate=update.yaw_rate,
            )
            self.updated_at_unix = time.time()
            if self.action == "stream_move":
                self.heartbeat_at_monotonic = time.monotonic()
        return {
            "accepted": True,
            "action_id": self.id,
            "action": "update_move",
            "vx": update.vx,
            "vy": update.vy,
            "yaw_rate": update.yaw_rate,
        }

    def is_alive(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _parameter(self, name: str) -> float:
        with self._lock:
            return self._parameters[name]

    def _velocity(self) -> tuple[float, float, float]:
        with self._lock:
            return (
                self._parameters["vx"],
                self._parameters["vy"],
                self._parameters["yaw_rate"],
            )

    def _heartbeat_expired(self) -> bool:
        with self._lock:
            heartbeat_at = self.heartbeat_at_monotonic
        return (
            isinstance(heartbeat_at, bool)
            or not isinstance(heartbeat_at, (int, float))
            or time.monotonic() - float(heartbeat_at) > STREAM_HEARTBEAT_TIMEOUT_S
        )

    def run(
        self,
        transport: RobotTransport,
        command_lock: threading.Lock,
        require_ready: Callable[[], RobotState],
    ) -> dict[str, Any]:
        """Run the lease synchronously; callers choose and own the worker thread."""

        status = "completed"
        detail = "target reached"
        started = time.monotonic()
        start_state: RobotState | None = None
        previous_yaw = 0.0
        accumulated_yaw = 0.0
        travelled_distance = 0.0
        turn_direction = math.copysign(1.0, self._parameter("yaw_rate"))
        try:
            start_state = require_ready()
            previous_yaw = start_state.yaw
            while not self.stop_event.is_set():
                elapsed = time.monotonic() - started
                if elapsed >= self._parameter("duration_s"):
                    if self.action in {"move_distance", "turn"}:
                        status, detail = "failed", "safety timeout before target"
                    else:
                        detail = "duration elapsed"
                    break

                current = require_ready()
                if self.action == "stream_move" and self._heartbeat_expired():
                    status, detail = "failed", "stream heartbeat expired"
                    break
                if self.action == "move_distance":
                    assert start_state is not None
                    travelled_distance = math.hypot(
                        current.position[0] - start_state.position[0],
                        current.position[1] - start_state.position[1],
                    )
                    if travelled_distance >= self._parameter("distance_m"):
                        break
                elif self.action == "turn":
                    signed_delta = turn_direction * normalize_angle(current.yaw - previous_yaw)
                    accumulated_yaw = max(0.0, accumulated_yaw + signed_delta)
                    previous_yaw = current.yaw
                    if accumulated_yaw >= self._parameter("angle_rad"):
                        break

                vx, vy, yaw_rate = self._velocity()
                with command_lock:
                    # stop() sets the event before waiting for this lock. This
                    # closes the StopMove/Move race.
                    if self.stop_event.is_set():
                        status, detail = "cancelled", "stop requested"
                        break
                    code = transport.move(vx, vy, yaw_rate)
                if code != 0:
                    status, detail = "failed", f"SDK Move returned code {code}"
                    break
                self.stop_event.wait(CONTROL_PERIOD_S)
            if self.stop_event.is_set():
                status, detail = "cancelled", "stop requested"
        except Exception as exc:  # noqa: BLE001 - lease must always reach StopMove
            status, detail = "failed", str(exc)
        finally:
            try:
                with command_lock:
                    code = transport.stop()
                if code != 0 and status == "completed":
                    status, detail = "failed", f"SDK StopMove returned code {code}"
            except Exception as exc:  # noqa: BLE001 - report final safety-stop failure
                status, detail = "failed", f"stop failed: {exc}"

        result: dict[str, Any] = {
            "id": self.id,
            "action": self.action,
            "status": status,
            "detail": detail,
            "elapsed_s": round(time.monotonic() - started, 3),
            "finished_at_unix": time.time(),
        }
        if self.action == "move_distance":
            result["measured_distance_m"] = round(travelled_distance, 5)
        elif self.action == "turn":
            result["measured_angle_rad"] = round(accumulated_yaw, 5)
        return result


__all__ = ["MotionLease"]
