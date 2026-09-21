"""Structural policy, observation, and mobile-base task boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .motion import MobileBaseState, PlanarVelocityCommand
from .observation import NavigationObservation
from .plan import WaypointPlan
from .request import NavigationRequest


class MobileBaseError(RuntimeError):
    """A mobile-base transport or command operation failed."""


@runtime_checkable
class NavigationPolicy(Protocol):
    """Task-semantic policy boundary, independent of model serving."""

    async def plan(self, request: NavigationRequest) -> WaypointPlan: ...


@runtime_checkable
class ObservationSource(Protocol):
    """Capture an episode-scoped navigation observation."""

    def capture(
        self,
        *,
        episode_id: str,
        reset: bool,
        robot_state: Mapping[str, object],
    ) -> NavigationObservation: ...


@runtime_checkable
class MobileBase(Protocol):
    """Minimal lease-based planar control surface used by navigation sessions."""

    def state(self) -> MobileBaseState: ...

    def preflight(self) -> MobileBaseState: ...

    def start_velocity_lease(
        self,
        command: PlanarVelocityCommand,
        *,
        duration_s: float,
    ) -> str: ...

    def update_velocity_lease(
        self,
        lease_id: str,
        command: PlanarVelocityCommand,
    ) -> None: ...

    def stop(self) -> None: ...


__all__ = [
    "MobileBase",
    "MobileBaseError",
    "NavigationPolicy",
    "ObservationSource",
]
