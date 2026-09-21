"""Shared lifecycle values for navigation task sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NavigationEndReason = Literal["terminal", "max_runtime"]


class NavigationSessionError(RuntimeError):
    """An observation, policy, or session invariant prevented navigation."""


@dataclass(frozen=True, slots=True)
class NavigationSessionEvent:
    kind: str
    elapsed_s: float
    observation_sequence: int | None = None
    waypoint_index: int | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class NavigationSessionResult:
    episode_id: str
    reason: NavigationEndReason
    elapsed_s: float
    inference_count: int
    plans_accepted: int
    control_ticks: int
    motion_commands: int
    last_observation_sequence: int | None
    events: tuple[NavigationSessionEvent, ...]
    events_dropped: int = 0


__all__ = [
    "NavigationEndReason",
    "NavigationSessionError",
    "NavigationSessionEvent",
    "NavigationSessionResult",
]
