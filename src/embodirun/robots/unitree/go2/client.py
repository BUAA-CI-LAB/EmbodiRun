"""Client for the bounded Go2 SDK2 HTTP control service."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from embodirun.robots.unitree.go2.navigation.interfaces import MobileBaseError
from embodirun.robots.unitree.go2.navigation.motion import MobileBaseState, PlanarVelocityCommand
from embodirun.utils import HttpClientError, JsonHttpClient, Pose2D


class Go2ClientError(MobileBaseError):
    pass


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Go2ClientError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Go2ClientError(f"{name} must be finite")
    return result


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Go2ClientError(f"{name} must be an object")
    return dict(value)


class Go2ControlClient:
    """Synchronous state/command client; it never imports the Unitree SDK."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 1.0,
        expected_transport: str | None = "unitree-sdk2",
    ) -> None:
        self.http = JsonHttpClient(base_url, token=token, timeout_s=timeout_s)
        self.expected_transport = expected_transport

    def _request(self, method: str, path: str, payload: Mapping[str, object] | None = None):
        try:
            return self.http.request_json(method, path, payload)
        except HttpClientError as error:
            raise Go2ClientError(str(error)) from error

    def _state_from_payload(self, payload: Mapping[str, object], *, require_fresh: bool) -> MobileBaseState:
        if self.expected_transport and payload.get("transport") != self.expected_transport:
            raise Go2ClientError(
                f"unexpected transport {payload.get('transport')!r}; expected {self.expected_transport!r}"
            )
        if payload.get("robot_state_available") is not True:
            raise Go2ClientError("robot state is unavailable")
        if require_fresh and payload.get("robot_state_fresh") is not True:
            raise Go2ClientError("robot state is stale")
        state = _object(payload.get("robot_state"), "robot_state")
        position = state.get("position")
        velocity = state.get("velocity")
        if isinstance(position, (str, bytes)) or not isinstance(position, Sequence) or len(position) < 2:
            raise Go2ClientError("robot position must contain x and y")
        if isinstance(velocity, (str, bytes)) or not isinstance(velocity, Sequence) or len(velocity) < 2:
            raise Go2ClientError("robot velocity must contain forward and lateral values")
        sequence = state.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise Go2ClientError("robot sequence must be a non-negative integer")
        timestamp_value = state.get("received_at_unix")
        received_at = None
        if isinstance(timestamp_value, (int, float)) and not isinstance(timestamp_value, bool):
            timestamp = float(timestamp_value)
            if math.isfinite(timestamp) and timestamp >= 0:
                received_at = timestamp
        active_value = payload.get("active_action")
        active = None if active_value is None else _object(active_value, "active_action")
        active_lease_id = None
        if active is not None:
            action_id = active.get("id")
            if not isinstance(action_id, str) or not action_id:
                raise Go2ClientError("active_action.id must be a non-empty string")
            active_lease_id = action_id
        return MobileBaseState(
            pose=Pose2D(
                _finite(position[0], "position[0]"),
                _finite(position[1], "position[1]"),
                _finite(state.get("yaw"), "yaw"),
            ),
            forward_velocity_mps=_finite(velocity[0], "velocity[0]"),
            lateral_velocity_mps=_finite(velocity[1], "velocity[1]"),
            yaw_rate_rps=_finite(state.get("yaw_rate"), "yaw_rate"),
            sequence=sequence,
            received_at_s=received_at,
            active_velocity_lease_id=active_lease_id,
            raw=state,
        )

    def state(self, *, require_fresh: bool = True) -> MobileBaseState:
        payload = self._request("GET", "/v1/state")
        return self._state_from_payload(payload, require_fresh=require_fresh)

    def preflight(self) -> MobileBaseState:
        """Require an idle, explicitly armed live control service.

        This check is read-only.  The control API independently rechecks the
        interlock when a motion lease is created, so passing preflight never
        grants motion authority by itself.
        """

        payload = self._request("GET", "/v1/state")
        state = self._state_from_payload(payload, require_fresh=True)
        if payload.get("closing") is True:
            raise Go2ClientError("control service is closing")
        if payload.get("operator_motion_ready") is not True:
            raise Go2ClientError("operator motion-ready interlock is not enabled")
        if payload.get("active_action") is not None:
            raise Go2ClientError("another robot action is active")
        return state

    def start_velocity_lease(self, command: PlanarVelocityCommand, *, duration_s: float = 10.0) -> str:
        result = self._request(
            "POST",
            "/v1/actions",
            {
                "action": "stream_move",
                "vx": command.vx_mps,
                "vy": command.vy_mps,
                "yaw_rate": command.yaw_rate_rps,
                "duration_s": duration_s,
            },
        )
        action_id = result.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise Go2ClientError("control service did not return an action_id")
        return action_id

    def update_velocity_lease(self, action_id: str, command: PlanarVelocityCommand) -> None:
        result = self._request(
            "POST",
            "/v1/actions",
            {
                "action": "update_move",
                "action_id": action_id,
                "vx": command.vx_mps,
                "vy": command.vy_mps,
                "yaw_rate": command.yaw_rate_rps,
            },
        )
        if result.get("accepted") is not True:
            raise Go2ClientError("velocity update was not accepted")

    def stop(self) -> None:
        result = self._request("POST", "/v1/stop", {})
        if result.get("stopped") is not True:
            raise Go2ClientError(f"control service did not confirm stop: {result}")


__all__ = ["Go2ClientError", "Go2ControlClient"]
