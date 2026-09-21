"""Common values and boundary implemented by simulator adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.types import Metadata


@dataclass(frozen=True, slots=True)
class SimulatorObservation:
    """One simulated embodiment state and its rendered camera frames."""

    robot: RobotObservation
    frames: tuple[CameraFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RobotObservation):
            raise TypeError("simulator observation robot must be a RobotObservation")
        frames = tuple(self.frames)
        if not frames or any(not isinstance(frame, CameraFrame) for frame in frames):
            raise ValueError("simulator observation frames must contain at least one CameraFrame")
        names = [frame.name for frame in frames]
        if len(names) != len(set(names)):
            raise ValueError("simulator observation frame names must be unique")
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """One normalized transition returned after advancing a simulator."""

    observation: SimulatorObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, SimulatorObservation):
            raise TypeError("simulation step observation must be a SimulatorObservation")
        if isinstance(self.reward, bool) or not isinstance(self.reward, Real):
            raise TypeError("simulation step reward must be a real number")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("simulation step reward must be finite")
        object.__setattr__(self, "reward", reward)
        for name in ("terminated", "truncated"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"simulation step {name} must be a boolean")
        if not isinstance(self.info, Mapping):
            raise TypeError("simulation step info must be a mapping")
        object.__setattr__(self, "info", dict(self.info))

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class SimulatorAdapter(ABC):
    """Own one environment and translate its observations and actions."""

    @property
    @abstractmethod
    def simulator_id(self) -> str:
        """Return the configured identity used for policy sessions."""

    @abstractmethod
    def reset(
        self,
        *,
        task: str | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SimulatorObservation:
        """Start an episode and return its initial observation."""

    @abstractmethod
    def step(self, action: RobotAction) -> SimulationStep:
        """Advance the environment by one normalized robot action."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator, renderer, and process resources."""


__all__ = [
    "SimulationStep",
    "SimulatorAdapter",
    "SimulatorObservation",
]
