"""Transport abstraction and a dependency-free dry-run simulator."""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod

from .types import RobotState, normalize_angle


class RobotTransport(ABC):
    name = "base"

    @abstractmethod
    def state(self) -> RobotState | None:
        raise NotImplementedError

    @abstractmethod
    def move(self, vx: float, vy: float, yaw_rate: float) -> int:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def posture(self, action: str) -> int:
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - a default no-op is intentional
        """Release transport resources; transports that own resources override this."""


class DryRunTransport(RobotTransport):
    """Minimal kinematic simulator used to test the API without a robot."""

    name = "dry-run"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._position = [0.0, 0.0, 0.0]
        self._yaw = 0.0
        self._velocity = [0.0, 0.0, 0.0]
        self._yaw_rate = 0.0
        self._last_update = time.monotonic()
        self._sequence = 0

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        forward, lateral, _ = self._velocity
        cos_yaw, sin_yaw = math.cos(self._yaw), math.sin(self._yaw)
        self._position[0] += (forward * cos_yaw - lateral * sin_yaw) * dt
        self._position[1] += (forward * sin_yaw + lateral * cos_yaw) * dt
        self._yaw = normalize_angle(self._yaw + self._yaw_rate * dt)

    def state(self) -> RobotState:
        with self._lock:
            self._integrate()
            self._sequence += 1
            return RobotState(
                received_at=time.monotonic(),
                sequence=self._sequence,
                position=tuple(self._position),
                roll=0.0,
                pitch=0.0,
                yaw=self._yaw,
                velocity=tuple(self._velocity),
                yaw_rate=self._yaw_rate,
                received_at_unix=time.time(),
            )

    def move(self, vx: float, vy: float, yaw_rate: float) -> int:
        with self._lock:
            self._integrate()
            self._velocity = [vx, vy, 0.0]
            self._yaw_rate = yaw_rate
        return 0

    def stop(self) -> int:
        return self.move(0.0, 0.0, 0.0)

    def posture(self, action: str) -> int:
        del action
        return 0


__all__ = ["DryRunTransport", "RobotTransport"]
