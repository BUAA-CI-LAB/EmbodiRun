"""Bounded event history and run statistics for navigation sessions."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from .events import (
    NavigationEndReason,
    NavigationSessionEvent,
    NavigationSessionResult,
)


class NavigationTelemetry:
    """Record one session run without owning any navigation decisions."""

    def __init__(
        self,
        *,
        max_events: int,
        event_sink: Callable[[NavigationSessionEvent], None] | None = None,
    ) -> None:
        self.event_sink = event_sink
        self._events: deque[NavigationSessionEvent] = deque(maxlen=max_events)
        self._started_at_s = 0.0
        self.events_dropped = 0
        self.inference_count = 0
        self.plans_accepted = 0
        self.control_ticks = 0
        self.motion_commands = 0
        self.last_observation_sequence: int | None = None

    @property
    def events(self) -> tuple[NavigationSessionEvent, ...]:
        return tuple(self._events)

    def reset(self) -> None:
        self._events.clear()
        self._started_at_s = time.monotonic()
        self.events_dropped = 0
        self.inference_count = 0
        self.plans_accepted = 0
        self.control_ticks = 0
        self.motion_commands = 0
        self.last_observation_sequence = None

    def emit(
        self,
        kind: str,
        *,
        observation_sequence: int | None = None,
        waypoint_index: int | None = None,
        message: str = "",
    ) -> None:
        if len(self._events) == self._events.maxlen:
            self.events_dropped += 1
        event = NavigationSessionEvent(
            kind=kind,
            elapsed_s=max(0.0, time.monotonic() - self._started_at_s),
            observation_sequence=observation_sequence,
            waypoint_index=waypoint_index,
            message=message,
        )
        self._events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)

    def result(
        self,
        *,
        episode_id: str,
        reason: NavigationEndReason,
    ) -> NavigationSessionResult:
        return NavigationSessionResult(
            episode_id=episode_id,
            reason=reason,
            elapsed_s=time.monotonic() - self._started_at_s,
            inference_count=self.inference_count,
            plans_accepted=self.plans_accepted,
            control_ticks=self.control_ticks,
            motion_commands=self.motion_commands,
            last_observation_sequence=self.last_observation_sequence,
            events=self.events,
            events_dropped=self.events_dropped,
        )


__all__ = ["NavigationTelemetry"]
