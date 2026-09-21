"""Immutable values used by the shared observation service."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from embodirun.robots.sensors.cameras import CameraFrame


def _freeze(value: Any) -> Any:
    """Deep-copy and freeze ordinary observation values.

    Camera payloads are already immutable ``bytes``.  Robot state commonly
    contains nested dictionaries/lists, so all containers are converted to
    immutable equivalents before a snapshot is published.  Unsupported SDK
    objects are rejected at this boundary rather than retained as mutable
    aliases.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({copy.deepcopy(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in copy.deepcopy(value))
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in copy.deepcopy(value))
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in copy.deepcopy(value))
    if isinstance(value, frozenset):
        return frozenset(_freeze(item) for item in copy.deepcopy(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return MappingProxyType({item.name: _freeze(getattr(value, item.name)) for item in dataclasses.fields(value)})
    # numpy arrays and similar tensor values expose a detached ``tolist``
    # representation.  Convert that representation into immutable tuples
    # rather than retaining a mutable SDK object or array view.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _freeze(to_list())
        except Exception as error:
            raise TypeError(f"observation value {type(value).__name__} cannot be frozen") from error
    if isinstance(value, (str, bytes, int, float, bool, complex, type(None))):
        return value
    raise TypeError(f"observation value {type(value).__name__} is not immutable")


def _thaw(value: Any) -> Any:
    """Return a detached, mutable representation for an explicit export."""

    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw(item) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, complex, type(None))):
        return value
    raise TypeError(f"observation value {type(value).__name__} cannot be exported")


def _validate_timestamp(value: int | None, field_name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{field_name} must be a non-negative integer or None")


class ObservationError(RuntimeError):
    """Base error for bounded observation access and production."""


class ObservationUnavailableError(ObservationError):
    """A requested observation is missing, expired, or crossed a reconnect."""

    def __init__(self, observation_id: str | None, reason: str) -> None:
        self.observation_id = observation_id
        self.reason = reason
        label = "observation" if observation_id is None else repr(observation_id)
        super().__init__(f"{label} is unavailable: {reason}")


class ObservationExpiredError(ObservationUnavailableError):
    """An observation ID was valid but is outside the bounded retention window."""


class ObservationSubscriptionClosed(ObservationError):
    """A subscription was closed and has no queued snapshots left."""


@dataclass(frozen=True, slots=True)
class SourceStatus:
    """Diagnostic state for one injected camera or state source."""

    source_id: str
    generation: int
    available: bool = False
    stale: bool = True
    last_error: str | None = None
    capture_count: int = 0
    error_count: int = 0
    last_captured_timestamp_ns: int | None = None
    last_received_timestamp_ns: int | None = None
    clock_domain: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise ValueError("source generation must be an integer")
        _validate_timestamp(self.last_captured_timestamp_ns, "last_captured_timestamp_ns")
        _validate_timestamp(self.last_received_timestamp_ns, "last_received_timestamp_ns")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "generation": self.generation,
            "available": self.available,
            "stale": self.stale,
            "last_error": self.last_error,
            "capture_count": self.capture_count,
            "error_count": self.error_count,
            "last_captured_timestamp_ns": self.last_captured_timestamp_ns,
            "last_received_timestamp_ns": self.last_received_timestamp_ns,
            "clock_domain": self.clock_domain,
        }


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    """One immutable camera/state sample shared by all consumers.

    ``cameras`` contains only frames obtained for this publication.  A failed
    source never causes its previous frame to be copied into a new snapshot;
    the source appears in ``errors`` and ``stale``/``available`` diagnostics.
    ``source_timestamps_ns`` and ``clock_domains`` retain per-source timing so
    a caller can reject unknown or cross-domain skew rather than assuming the
    aggregate timestamp is synchronized.
    """

    observation_id: str
    service_instance_id: str
    generation: int
    sequence: int
    state: Any = None
    cameras: tuple[CameraFrame, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)
    stale: bool = False
    available: bool = True
    captured_timestamp_ns: int | None = None
    received_timestamp_ns: int | None = None
    published_timestamp_ns: int | None = None
    source_timestamps_ns: Mapping[str, int | None] = field(default_factory=dict)
    source_received_timestamps_ns: Mapping[str, int | None] = field(default_factory=dict)
    clock_domains: Mapping[str, str | None] = field(default_factory=dict)
    skew_ns: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("observation_id", "service_instance_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("generation", "sequence"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.cameras, tuple) or any(not isinstance(frame, CameraFrame) for frame in self.cameras):
            raise ValueError("cameras must be a tuple of CameraFrame values")
        for field_name in (
            "captured_timestamp_ns",
            "received_timestamp_ns",
            "published_timestamp_ns",
            "skew_ns",
        ):
            _validate_timestamp(getattr(self, field_name), field_name)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if not isinstance(self.errors, Mapping):
            raise ValueError("errors must be a mapping")
        if any(not isinstance(key, str) for key in self.errors):
            raise ValueError("observation error keys must be strings")
        if any(not isinstance(key, str) for key in self.source_timestamps_ns):
            raise ValueError("source timestamp keys must be strings")
        if any(not isinstance(key, str) for key in self.source_received_timestamps_ns):
            raise ValueError("source receive timestamp keys must be strings")
        if any(not isinstance(key, str) for key in self.clock_domains):
            raise ValueError("clock domain keys must be strings")
        for mapping_name in (
            "source_timestamps_ns",
            "source_received_timestamps_ns",
        ):
            for key, value in getattr(self, mapping_name).items():
                _validate_timestamp(value, f"{mapping_name}[{key!r}]")
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in self.clock_domains.values()
        ):
            raise ValueError("clock domains must be non-empty strings or None")
        object.__setattr__(self, "state", _freeze(self.state))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "errors", _freeze(self.errors))
        object.__setattr__(self, "source_timestamps_ns", _freeze(self.source_timestamps_ns))
        object.__setattr__(
            self,
            "source_received_timestamps_ns",
            _freeze(self.source_received_timestamps_ns),
        )
        object.__setattr__(self, "clock_domains", _freeze(self.clock_domains))

    @property
    def frames(self) -> tuple[CameraFrame, ...]:
        """Alias used by consumers that call the camera values ``frames``."""

        return self.cameras

    @property
    def timestamp_known(self) -> bool:
        """Whether this aggregate has a comparable capture timestamp."""

        return self.captured_timestamp_ns is not None and self.skew_ns is not None

    def age_ns(self, now_ns: int) -> int | None:
        _validate_timestamp(now_ns, "now_ns")
        if self.captured_timestamp_ns is None:
            return None
        age = now_ns - self.captured_timestamp_ns
        # A future timestamp indicates a clock-domain mismatch or a broken
        # source.  Returning ``None`` makes the freshness check fail closed.
        return age if age >= 0 else None

    def is_fresh(
        self,
        *,
        now_ns: int,
        now_clock_domain: str | None = None,
        max_age_ns: int,
        max_skew_ns: int | None = None,
    ) -> bool:
        """Apply freshness limits in the caller's explicitly named clock domain.

        A nanosecond value has no meaning without its clock domain.  The
        caller must therefore identify the domain that produced ``now_ns``;
        an omitted or mismatched domain fails closed instead of comparing a
        remote monotonic clock with this snapshot's local monotonic clock.
        """

        if now_clock_domain is not None and (not isinstance(now_clock_domain, str) or not now_clock_domain.strip()):
            raise ValueError("now_clock_domain must be a non-empty string")

        if isinstance(max_age_ns, bool) or not isinstance(max_age_ns, int) or max_age_ns < 0:
            raise ValueError("max_age_ns must be a non-negative integer")
        if max_skew_ns is not None and (
            isinstance(max_skew_ns, bool) or not isinstance(max_skew_ns, int) or max_skew_ns < 0
        ):
            raise ValueError("max_skew_ns must be a non-negative integer or None")
        age = self.age_ns(now_ns)
        timing_domains = {domain for domain in self.clock_domains.values() if domain is not None}
        aggregate_domain = self.metadata.get("clock_domain")
        return bool(
            self.available
            and not self.stale
            and age is not None
            and age <= max_age_ns
            and self.skew_ns is not None
            and len(timing_domains) == 1
            and aggregate_domain in timing_domains
            and now_clock_domain is not None
            and aggregate_domain == now_clock_domain
            and (max_skew_ns is None or self.skew_ns <= max_skew_ns)
        )

    def to_dict(self) -> dict[str, Any]:
        """Export detached values for a transport/recorder boundary."""

        return {
            "observation_id": self.observation_id,
            "service_instance_id": self.service_instance_id,
            "generation": self.generation,
            "sequence": self.sequence,
            "state": _thaw(self.state),
            "cameras": tuple(self.cameras),
            "metadata": _thaw(self.metadata),
            "errors": _thaw(self.errors),
            "stale": self.stale,
            "available": self.available,
            "captured_timestamp_ns": self.captured_timestamp_ns,
            "received_timestamp_ns": self.received_timestamp_ns,
            "published_timestamp_ns": self.published_timestamp_ns,
            "source_timestamps_ns": _thaw(self.source_timestamps_ns),
            "source_received_timestamps_ns": _thaw(self.source_received_timestamps_ns),
            "clock_domains": _thaw(self.clock_domains),
            "skew_ns": self.skew_ns,
        }


__all__ = [
    "ObservationError",
    "ObservationExpiredError",
    "ObservationSnapshot",
    "ObservationSubscriptionClosed",
    "ObservationUnavailableError",
    "SourceStatus",
]
