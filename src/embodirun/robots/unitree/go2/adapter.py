"""Robot contract adapter for the Go2 control agent."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from ...adapter import RobotAction, RobotAdapter, RobotObservation
from .client import Go2ClientError, Go2ControlClient
from .config import Go2Config
from .navigation.motion import MobileBaseState, PlanarVelocityCommand

GO2_ACTION_SPACE = "unitree.go2.planar_velocity.v1"


class Go2AdapterError(Go2ClientError):
    pass


class Go2Adapter(RobotAdapter):
    """Translate Go2 agent state and commands to Deploy robot contracts."""

    def __init__(
        self,
        config: Go2Config,
        *,
        control_client: Any | None = None,
    ) -> None:
        self.config = config
        self.robot_id = config.robot_id
        self.control = control_client or Go2ControlClient(
            config.control_url,
            token=config.api_token,
            timeout_s=config.timeout_s,
            expected_transport=config.expected_transport,
        )
        self._velocity_lease_id: str | None = None
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        state = self.control.state()
        if not isinstance(state, MobileBaseState):
            raise Go2AdapterError("Go2 control client must return MobileBaseState")
        self._connected = True

    def _require_connected(self) -> None:
        if not self._connected:
            raise Go2AdapterError("Go2 is not connected")

    def observe(self) -> RobotObservation:
        self._require_connected()
        state = self.control.state()
        if not isinstance(state, MobileBaseState):
            raise Go2AdapterError("Go2 control client must return MobileBaseState")
        metadata: dict[str, object] = {
            "robot_id": self.robot_id,
            "robot_type": "go2",
            "action_space": GO2_ACTION_SPACE,
            "sequence": state.sequence,
        }
        if state.active_velocity_lease_id is not None:
            metadata["active_velocity_lease_id"] = state.active_velocity_lease_id
        return RobotObservation(
            timestamp_s=(state.received_at_s if state.received_at_s is not None else time.time()),
            values={
                "position_m": [state.pose.x_m, state.pose.y_m],
                "yaw_rad": state.pose.yaw_rad,
                "linear_velocity_mps": [
                    state.forward_velocity_mps,
                    state.lateral_velocity_mps,
                ],
                "yaw_rate_rps": state.yaw_rate_rps,
            },
            metadata=metadata,
        )

    def execute(self, action: RobotAction) -> None:
        self._require_connected()
        declared_space = action.metadata.get("action_space")
        if declared_space is not None and declared_space != GO2_ACTION_SPACE:
            raise Go2AdapterError(f"unsupported action space {declared_space!r}; expected {GO2_ACTION_SPACE!r}")
        if not isinstance(action.values, Mapping):
            raise Go2AdapterError("Go2 action values must be an object")
        values = dict(action.values)
        kind = values.pop("type", None)
        if kind == "stop":
            if values:
                raise Go2AdapterError("stop action must not contain parameters")
            self.stop()
            return
        if kind != "planar_velocity":
            raise Go2AdapterError(f"unsupported Go2 action type: {kind!r}")
        expected_fields = {"vx_mps", "vy_mps", "yaw_rate_rps"}
        if set(values) != expected_fields:
            missing = expected_fields - values.keys()
            unexpected = values.keys() - expected_fields
            raise Go2AdapterError(
                "Go2 planar_velocity fields do not match the contract; "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        try:
            command = PlanarVelocityCommand(
                vx_mps=values["vx_mps"],
                vy_mps=values["vy_mps"],
                yaw_rate_rps=values["yaw_rate_rps"],
                limits=self.config.velocity_limits,
            )
        except (TypeError, ValueError) as error:
            raise Go2AdapterError(str(error)) from error
        if self._velocity_lease_id is None:
            self.control.preflight()
            self._velocity_lease_id = self.control.start_velocity_lease(
                command,
                duration_s=self.config.velocity_lease_duration_s,
            )
        else:
            self.control.update_velocity_lease(self._velocity_lease_id, command)

    def stop(self) -> None:
        """Stop the active motion and forget its lease after confirmation."""

        self._require_connected()
        self.control.stop()
        self._velocity_lease_id = None

    def close(self) -> None:
        if not self._connected:
            return
        try:
            if self._velocity_lease_id is not None:
                self.stop()
        finally:
            self._connected = False


__all__ = [
    "GO2_ACTION_SPACE",
    "Go2Adapter",
    "Go2AdapterError",
]
