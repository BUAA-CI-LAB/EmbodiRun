"""Decode and retain the latest Unitree SportModeState subscription sample."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from .types import RobotState


def start_state_subscription(
    subscriber_factory: Callable[..., Any],
    state_topic: str,
    state_type: Any,
    callback: Callable[[Any], None],
) -> Any:
    """Construct and start the SDK subscriber on its native owner thread."""

    subscriber = subscriber_factory(state_topic, state_type)
    subscriber.Init(callback, 10)
    return subscriber


class SportModeStateStore:
    """Thread-safe last-value store used by the SDK callback and HTTP readers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: RobotState | None = None
        self._sequence = 0

    def on_message(self, message: Any) -> None:
        try:
            received_at = time.monotonic()
            received_at_unix = time.time()
            position = tuple(float(value) for value in message.position)
            roll = float(message.imu_state.rpy[0])
            pitch = float(message.imu_state.rpy[1])
            yaw = float(message.imu_state.rpy[2])
            velocity = tuple(float(value) for value in message.velocity)
            yaw_rate = float(message.yaw_speed)
            mode = int(message.mode)
            gait_type = int(message.gait_type)
            error_code = int(message.error_code)
            with self._lock:
                self._sequence += 1
                self._state = RobotState(
                    received_at=received_at,
                    sequence=self._sequence,
                    position=position,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    velocity=velocity,
                    yaw_rate=yaw_rate,
                    received_at_unix=received_at_unix,
                    mode=mode,
                    gait_type=gait_type,
                    error_code=error_code,
                )
        except Exception as exc:  # noqa: BLE001 - SDK callback thread must survive
            print(f"state callback error: {exc}", file=sys.stderr, flush=True)

    def state(self) -> RobotState | None:
        with self._lock:
            return self._state


__all__ = ["SportModeStateStore", "start_state_subscription"]
