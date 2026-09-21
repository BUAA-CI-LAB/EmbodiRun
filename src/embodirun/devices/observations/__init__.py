"""Bounded shared observations for control-service consumers.

This public module is intentionally a small import surface.  Device owners
create camera/state sources; :class:`ObservationProducer` consumes them and
publishes immutable snapshots through :class:`ObservationStore`.
"""

from typing import TYPE_CHECKING, Any

from .hub import SharedSensorError, SharedSensorHub
from .producer import ObservationProducer
from .store import ObservationStore, ObservationSubscription
from .values import (
    ObservationError,
    ObservationExpiredError,
    ObservationSnapshot,
    ObservationSubscriptionClosed,
    ObservationUnavailableError,
    SourceStatus,
)

if TYPE_CHECKING:  # pragma: no cover - only needed by type checkers
    from ..recording import (
        ActionEvent,
        ObservationRecorder,
        RecordingError,
        RecordingStatus,
    )

_RECORDING_REEXPORTS = frozenset({"ActionEvent", "ObservationRecorder", "RecordingError", "RecordingStatus"})


def __getattr__(name: str) -> Any:
    # ``devices.recording`` imports ``devices.observations.store``; importing it
    # eagerly here would make the two packages depend on each other and break
    # callers that import ``devices.recording`` first.  Resolve these public
    # re-exports lazily instead.
    if name in _RECORDING_REEXPORTS:
        from .. import recording

        return getattr(recording, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ObservationError",
    "ObservationExpiredError",
    "ObservationProducer",
    "ObservationSnapshot",
    "ObservationStore",
    "ObservationSubscription",
    "ObservationSubscriptionClosed",
    "ObservationUnavailableError",
    "ActionEvent",
    "ObservationRecorder",
    "RecordingError",
    "RecordingStatus",
    "SourceStatus",
    "SharedSensorError",
    "SharedSensorHub",
]
