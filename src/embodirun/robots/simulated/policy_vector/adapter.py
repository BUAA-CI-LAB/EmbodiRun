"""Unit-preserving six-value simulated robot adapter.

This adapter is an inference and lifecycle test endpoint.  It stores values
in memory, never imports a hardware SDK, and does not decide whether the
checkpoint's native values are degrees, normalized positions, or another
representation.  That boundary lets an experiment exercise the real policy
transport without making a physical-unit claim.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

from ...adapter import RobotAction, RobotAdapter, RobotObservation
from .config import (
    POLICY_VECTOR_SIZE,
    STATE_FIELD,
    UNVERIFIED_UNITS,
    PolicyVectorConfig,
    finite_vector,
)

POLICY_VECTOR_ACTION_SPACE = "simulated.policy_vector.state_native.v1"
POLICY_VECTOR_ACTION_TYPE = "state_native_vector"


class PolicyVectorAdapterError(RuntimeError):
    """A simulated policy-vector lifecycle or action request is invalid."""


class PolicyVectorAdapter(RobotAdapter):
    """In-memory six-value adapter with an explicit passive/prepare boundary."""

    def __init__(self, config: PolicyVectorConfig) -> None:
        self.config = config
        self.robot_id = config.robot_id
        self._lock = threading.RLock()
        self._connected = False
        self._prepared = False
        self._state = tuple(config.initial_state_native)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def prepared(self) -> bool:
        return self._prepared

    def connect(self, *, prepare: bool = True) -> None:
        """Open only the in-memory endpoint, optionally preparing commands."""

        if not isinstance(prepare, bool):
            raise TypeError("prepare must be a boolean")
        with self._lock:
            self._connected = True
            self._prepared = False
            if prepare:
                self._prepared = True

    def prepare(self) -> None:
        """Upgrade a passive in-memory owner for explicit command execution."""

        with self._lock:
            if not self._connected:
                raise PolicyVectorAdapterError("policy-vector robot is not connected")
            self._prepared = True

    def _state_payload(self) -> dict[str, object]:
        return {
            STATE_FIELD: list(self._state),
            "units": UNVERIFIED_UNITS,
        }

    def observe(self) -> RobotObservation:
        """Return the current simulated state with no physical interpretation."""

        with self._lock:
            if not self._connected:
                raise PolicyVectorAdapterError("policy-vector robot is not connected")
            state = list(self._state)
        return RobotObservation(
            timestamp_s=time.time(),
            values={STATE_FIELD: state},
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "simulated.policy_vector",
                "action_space": POLICY_VECTOR_ACTION_SPACE,
                "units": UNVERIFIED_UNITS,
                "simulated": True,
                "hardware_access": False,
                "state_source": "in_memory_policy_vector",
                "captured_timestamp_ns": time.monotonic_ns(),
                "clock_domain": "host_monotonic_ns",
            },
        )

    def execute(self, action: RobotAction) -> dict[str, object]:
        """Apply one explicit native-vector action and return simulation facts."""

        with self._lock:
            if not self._connected:
                raise PolicyVectorAdapterError("policy-vector robot is not connected")
            if not self._prepared:
                raise PolicyVectorAdapterError("policy-vector robot is not prepared")
            if not isinstance(action, RobotAction):
                raise TypeError("policy-vector action must be a RobotAction")
            if action.metadata.get("action_space") != POLICY_VECTOR_ACTION_SPACE:
                raise PolicyVectorAdapterError(
                    f"policy-vector action must declare action_space {POLICY_VECTOR_ACTION_SPACE!r}"
                )
            if not isinstance(action.values, Mapping):
                raise PolicyVectorAdapterError("policy-vector action values must be an object")
            values = dict(action.values)
            if values.pop("type", None) != POLICY_VECTOR_ACTION_TYPE:
                raise PolicyVectorAdapterError(f"policy-vector action type must be {POLICY_VECTOR_ACTION_TYPE!r}")
            if set(values) != {STATE_FIELD}:
                raise PolicyVectorAdapterError(f"policy-vector action must contain only {STATE_FIELD!r}")
            try:
                requested = finite_vector(values[STATE_FIELD], STATE_FIELD)
            except ValueError as error:
                raise PolicyVectorAdapterError(str(error)) from error
            self._state = requested
            state = self._state_payload()
            return {
                "requested": {STATE_FIELD: list(requested), "units": UNVERIFIED_UNITS},
                "applied": dict(state),
                "measured": dict(state),
                "simulated": True,
                "hardware_access": False,
            }

    def stop(self) -> dict[str, object]:
        """Hold the in-memory state; this method performs no hardware write."""

        with self._lock:
            if not self._connected:
                raise PolicyVectorAdapterError("policy-vector robot is not connected")
            state = self._state_payload()
            return {
                "requested": "stop",
                "applied": dict(state),
                "measured": dict(state),
                "stop_confirmed": True,
                "simulated": True,
                "hardware_access": False,
            }

    def close(self) -> None:
        """Release only in-memory ownership while retaining no live worker."""

        with self._lock:
            self._prepared = False
            self._connected = False


__all__ = [
    "POLICY_VECTOR_ACTION_SPACE",
    "POLICY_VECTOR_ACTION_TYPE",
    "POLICY_VECTOR_SIZE",
    "PolicyVectorAdapter",
    "PolicyVectorAdapterError",
]
