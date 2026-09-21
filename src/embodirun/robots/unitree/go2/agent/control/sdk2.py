"""Unitree SDK2 transport composed from loader, state, and owner services."""

from __future__ import annotations

from typing import Any

from .sdk_loader import (
    dds_library_has_unsafe_iceoryx,
    ensure_cyclonedds_library_dir,
    load_sdk_components,
    loaded_dds_library_path,
)
from .sdk_owner import SdkCall, SdkOwner
from .sdk_state import SportModeStateStore
from .transport import RobotTransport
from .types import RobotState

# Retain the private type name used by older diagnostics and tests.
_SdkCall = SdkCall


class UnitreeTransport(RobotTransport):
    """Adapt the SportClient API while one owner serializes native operations."""

    name = "unitree-sdk2"

    def __init__(
        self,
        interface: str,
        state_topic: str = "rt/lf/sportmodestate",
        rpc_timeout_s: float = 2.0,
        *,
        required_dds_lib_dir: str | None = None,
        _sdk_components: tuple[Any, Any, Any, Any] | None = None,
    ) -> None:
        # SDK imports stay inside live construction so importing the package
        # and running dry-run tests works on macOS and CI.
        components = load_sdk_components(required_dds_lib_dir) if _sdk_components is None else _sdk_components
        self._state_store = SportModeStateStore()
        self._owner = SdkOwner(
            interface,
            state_topic,
            rpc_timeout_s,
            components,
            self._on_state,
        )
        # Compatibility alias for existing thread-liveness diagnostics.
        self._sdk_thread = self._owner.thread

    def _on_state(self, message: Any) -> None:
        self._state_store.on_message(message)

    def state(self) -> RobotState | None:
        return self._state_store.state()

    def _enqueue_sdk_call(
        self,
        method: str,
        args: tuple[Any, ...],
        *,
        allow_closing: bool = False,
    ) -> SdkCall:
        return self._owner.enqueue(method, args, allow_closing=allow_closing)

    @staticmethod
    def _completed_sdk_result(call: SdkCall) -> int:
        return SdkOwner._completed_result(call)

    def _wait_sdk_call(
        self,
        call: SdkCall,
        *,
        cancel_if_pending: bool = True,
    ) -> int:
        return self._owner.wait(call, cancel_if_pending=cancel_if_pending)

    def _sdk_call(self, method: str, *args: Any) -> int:
        return self._owner.call(method, *args)

    def move(self, vx: float, vy: float, yaw_rate: float) -> int:
        return self._owner.call("Move", vx, vy, yaw_rate)

    def stop(self) -> int:
        return self._owner.stop()

    def posture(self, action: str) -> int:
        methods = {
            "stand_up": "StandUp",
            "stand_down": "StandDown",
            "balance_stand": "BalanceStand",
            "recovery_stand": "RecoveryStand",
        }
        return self._owner.call(methods[action])

    def close(self) -> None:
        self._owner.close()


__all__ = [
    "UnitreeTransport",
    "dds_library_has_unsafe_iceoryx",
    "ensure_cyclonedds_library_dir",
    "loaded_dds_library_path",
]
