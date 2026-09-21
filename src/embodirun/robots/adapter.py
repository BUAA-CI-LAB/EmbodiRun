"""Common values and boundary implemented by physical robot adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from embodirun.types import Metadata, TensorTree


@dataclass(slots=True)
class RobotObservation:
    """Timestamped sensor state in one robot's observation schema."""

    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class RobotAction:
    """Timestamped command in one robot's declared action space."""

    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


class RobotPreparationRefused(RuntimeError):
    """A robot explicitly refused preparation before any write occurred."""

    def __init__(self, message: str, receipt: object = None):
        super().__init__(message)
        self.receipt = receipt


class RobotAdapter(ABC):
    """Own one hardware connection and translate Deploy's robot contracts."""

    @abstractmethod
    def connect(self, *, prepare: bool = True) -> None:
        """Establish a connection, optionally leaving actuators unprepared.

        ``prepare=False`` is the passive boundary used by observation paths.
        Adapters that cannot prove a side-effect-free connection must reject
        that request rather than silently falling back to their legacy startup
        behavior.
        """

    def prepare(self) -> None:
        """Prepare an already connected adapter for commanded control."""
        raise NotImplementedError(f"{type(self).__name__} does not expose an explicit prepare operation")

    @abstractmethod
    def observe(self) -> RobotObservation:
        """Read SDK state and return a normalized robot observation."""

    @abstractmethod
    def execute(self, action: RobotAction) -> None:
        """Validate and translate a normalized action into an SDK command."""

    @abstractmethod
    def stop(self) -> None:
        """Bring commanded motion to a safe stop or hold state."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware and transport resources."""


__all__ = ["RobotAction", "RobotAdapter", "RobotObservation", "RobotPreparationRefused"]
