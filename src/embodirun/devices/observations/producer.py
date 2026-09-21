"""Injected-source observation producer."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from embodirun.robots.sensors.cameras import CameraFrame

from .store import ObservationStore, ObservationSubscription
from .values import (
    ObservationError,
    ObservationSnapshot,
    ObservationUnavailableError,
    SourceStatus,
    _freeze,
)

_CLOCK_DOMAIN = "host_monotonic_ns"
_STATE_SOURCE_ID = "state"


def default_interval_s() -> float:
    """Return the observation poll interval in seconds.

    A fixed 0.1 s poll caps every observation stream at 10 Hz regardless of the
    configured control rate, which starves a 20 Hz control loop of fresh frames.
    A poll costs roughly 20 ms on a Pi 4B, so 0.03 s yields a ~50 ms period and
    reaches 20 Hz.

    Override with ``RLINF_DEPLOY_OBSERVATION_INTERVAL_S``; invalid values fall
    back to the default.
    """

    raw = os.environ.get("RLINF_DEPLOY_OBSERVATION_INTERVAL_S", "0.03")
    try:
        value = float(raw)
    except ValueError:
        return 0.03
    return value if value > 0 and math.isfinite(value) else 0.03


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    source_id: str
    capture: Callable[[], Any]
    close: Callable[[], None] | None


@dataclass(frozen=True, slots=True)
class _SourceResult:
    """One completed capture from a single bounded source worker."""

    token: int
    value: Any = None
    error: BaseException | None = None


class _SourceWorker:
    """One daemon worker per source; never queues a second capture."""

    def __init__(self, capture: Callable[[], Any], name: str) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._job: tuple[int, int] | None = None
        self._completed: _SourceResult | None = None
        self._running = False
        self._generation = 0
        self._next_token = 0
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"rlinf-observation-source-{name}",
            daemon=True,
        )
        self._thread.start()

    def request(self) -> int | None:
        """Schedule one capture, or return ``None`` while one is pending."""

        with self._condition:
            if self._stop or self._running or self._job is not None or self._completed is not None:
                return None
            self._next_token += 1
            token = self._next_token
            self._job = (self._generation, token)
            self._condition.notify()
            return token

    def take_completed(self) -> _SourceResult | None:
        with self._condition:
            result = self._completed
            self._completed = None
            return result

    @property
    def running(self) -> bool:
        with self._condition:
            return self._running or self._job is not None

    @property
    def active_token(self) -> int | None:
        with self._condition:
            if self._job is not None:
                return self._job[1]
            if self._running:
                return self._next_token
            return None

    def invalidate(self) -> None:
        """Drop completed/late results after a source reconnect."""

        with self._condition:
            self._generation += 1
            self._completed = None

    def close(self, timeout_s: float) -> bool:
        with self._condition:
            self._stop = True
            self._job = None
            self._condition.notify_all()
        self._thread.join(max(0.0, timeout_s))
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._job is None and not self._stop:
                    self._condition.wait()
                if self._stop and self._job is None:
                    return
                assert self._job is not None
                generation, token = self._job
                self._job = None
                self._running = True
            value: Any = None
            error: BaseException | None = None
            try:
                value = self._capture()
            except BaseException as capture_error:  # source diagnostics are explicit
                error = capture_error
            with self._condition:
                self._running = False
                if not self._stop and generation == self._generation:
                    self._completed = _SourceResult(token, value, error)
                self._condition.notify_all()


class ObservationProducer:
    """Capture injected sources once and publish bounded immutable snapshots.

    ``state_reader`` and camera sources are called outside the store lock.  A
    source may itself be backed by the #22 device owner or by #24's scheduled
    state reader.  By default the producer does not close injected sources;
    set ``own_sources=True`` only when this lifecycle owns those adapters.
    """

    def __init__(
        self,
        state_reader: Callable[[], Any] | None = None,
        camera_sources: Mapping[str, Any] | None = None,
        *,
        interval_s: float | None = None,
        source_timeout_s: float = 0.05,
        max_retained: int = 32,
        subscription_queue_size: int = 8,
        store: ObservationStore | None = None,
        clock_ns: Callable[[], int] | None = None,
        clock_domain: str = _CLOCK_DOMAIN,
        stale_after_ns: int | None = None,
        own_sources: bool = False,
        service_instance_id: str | None = None,
    ) -> None:
        if interval_s is None:
            interval_s = default_interval_s()
        if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)):
            raise TypeError("interval_s must be a number")
        if interval_s <= 0 or not math.isfinite(float(interval_s)):
            raise ValueError("interval_s must be finite and positive")
        if (
            isinstance(source_timeout_s, bool)
            or not isinstance(source_timeout_s, (int, float))
            or source_timeout_s < 0
            or not math.isfinite(float(source_timeout_s))
        ):
            raise ValueError("source_timeout_s must be finite and non-negative")
        if isinstance(stale_after_ns, bool) or (
            stale_after_ns is not None and (not isinstance(stale_after_ns, int) or stale_after_ns < 0)
        ):
            raise ValueError("stale_after_ns must be a non-negative integer or None")
        if not isinstance(clock_domain, str) or not clock_domain.strip():
            raise ValueError("clock_domain must not be empty")
        if not isinstance(own_sources, bool):
            raise TypeError("own_sources must be a boolean")
        if isinstance(subscription_queue_size, bool) or not isinstance(subscription_queue_size, int):
            raise TypeError("subscription_queue_size must be an integer")
        if subscription_queue_size <= 0:
            raise ValueError("subscription_queue_size must be positive")
        self.interval_s = float(interval_s)
        self.source_timeout_s = float(source_timeout_s)
        self.subscription_queue_size = subscription_queue_size
        self._clock_ns = clock_ns or time.monotonic_ns
        self.clock_domain = clock_domain
        self.stale_after_ns = stale_after_ns
        self.own_sources = own_sources
        self.store = store or ObservationStore(
            max_retained=max_retained,
            service_instance_id=service_instance_id,
        )
        self._state_reader = _as_reader(state_reader)
        self._sources = self._normalise_sources(camera_sources)
        if self._state_reader is None and not self._sources:
            raise ValueError("at least one state reader or camera source is required")
        self._status_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._source_workers = {
            source.source_id: _SourceWorker(
                source.capture,
                source.source_id.replace("/", "_").replace(" ", "_"),
            )
            for source in self._sources
        }
        self._reported_timeouts: set[tuple[str, int]] = set()
        self._statuses: dict[str, SourceStatus] = {
            source.source_id: SourceStatus(
                source.source_id,
                self.store.generation,
                available=False,
                stale=True,
            )
            for source in self._sources
        }
        if self._state_reader is not None:
            self._statuses[_STATE_SOURCE_ID] = SourceStatus(
                _STATE_SOURCE_ID,
                self.store.generation,
                available=False,
                stale=True,
            )
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._last_error: str | None = None

    @staticmethod
    def _normalise_sources(
        camera_sources: Mapping[str, Any] | None,
    ) -> tuple[_SourceBinding, ...]:
        values: list[tuple[str, Any]] = []
        if camera_sources is not None:
            if not isinstance(camera_sources, Mapping):
                raise TypeError("camera_sources must map source IDs to callables or camera sources")
            values.extend(camera_sources.items())
        bindings: list[_SourceBinding] = []
        seen: set[str] = set()
        for source_name, source in values:
            if not isinstance(source_name, str) or not source_name.strip():
                raise ValueError("camera source IDs must not be empty")
            if source_name in seen:
                raise ValueError(f"duplicate camera source ID {source_name!r}")
            seen.add(source_name)
            capture = getattr(source, "capture", None)
            if not callable(capture):
                if not callable(source):
                    raise TypeError(f"camera source {source_name!r} must be callable")
                capture = source
            close = getattr(source, "close", None)
            bindings.append(
                _SourceBinding(
                    source_name,
                    capture,
                    close if callable(close) else None,
                )
            )
        return tuple(bindings)

    def add_camera_source(self, source_id: str, source: Any) -> None:
        """Attach one physical source before the next shared publication.

        Service owners use this when a lazily opened camera joins the shared
        service.  The capture lock makes the source list change atomic with
        ``publish_once``; an existing source ID is never silently replaced.
        """

        bindings = self._normalise_sources({source_id: source})
        binding = bindings[0]
        with self._lifecycle_lock:
            if self._closed:
                raise ObservationError("observation producer is closed")
        with self._capture_lock:
            if any(item.source_id == source_id for item in self._sources):
                raise ValueError(f"camera source ID {source_id!r} is already attached")
            self._sources = (*self._sources, binding)
            self._source_workers[source_id] = _SourceWorker(
                binding.capture,
                source_id.replace("/", "_").replace(" ", "_"),
            )
            with self._status_lock:
                self._statuses[source_id] = SourceStatus(
                    source_id,
                    self.store.generation,
                    available=False,
                    stale=True,
                )

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def generation(self) -> int:
        return self.store.generation

    @property
    def last_error(self) -> str | None:
        with self._status_lock:
            return self._last_error

    def source_status(self, source_id: str | None = None) -> SourceStatus | Mapping[str, SourceStatus]:
        with self._status_lock:
            if source_id is not None:
                try:
                    return self._statuses[source_id]
                except KeyError as error:
                    raise ObservationUnavailableError(source_id, "source ID is unknown") from error
            return MappingProxyType(dict(self._statuses))

    def status(self) -> Mapping[str, Any]:
        """Return detached producer diagnostics without touching a source."""

        with self._status_lock:
            values = {source_id: status.to_dict() for source_id, status in self._statuses.items()}
            return MappingProxyType(
                {
                    "running": self.running,
                    "closed": self.closed,
                    "generation": self.generation,
                    "last_error": self._last_error,
                    "sources": _freeze(values),
                }
            )

    def start(self) -> ObservationProducer:
        with self._lifecycle_lock:
            if self._closed:
                raise ObservationError("observation producer is closed")
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="rlinf-observation-producer",
                daemon=True,
            )
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.publish_once()
            except (ObservationError, ObservationUnavailableError) as error:
                with self._status_lock:
                    self._last_error = str(error)
            except Exception as error:  # pragma: no cover - defensive worker guard
                with self._status_lock:
                    self._last_error = str(error)
            self._stop_event.wait(self.interval_s)

    def publish_once(self) -> ObservationSnapshot:
        with self._lifecycle_lock:
            if self._closed:
                raise ObservationError("observation producer is closed")
        # Serialise source captures so a source owner sees one producer call at
        # a time, but do not hold the store lock during capture or encoding.
        with self._capture_lock:
            generation = self.store.generation
            (
                state,
                state_timestamp,
                state_received,
                state_domain,
                state_error,
                state_metadata,
            ) = self._read_state()
            frames: list[CameraFrame] = []
            errors: dict[str, str] = {}
            source_timestamps: dict[str, int | None] = {}
            source_received_timestamps: dict[str, int | None] = {}
            clock_domains: dict[str, str | None] = {}
            source_frame_names: dict[str, tuple[str, ...]] = {}
            successful_sources = 0
            source_stale = False
            if self._state_reader is not None:
                if state_error is None and state_received is None and state is None:
                    # A dynamic owner may intentionally have no robot source
                    # yet (camera-only preview).  Do not turn that absence
                    # into a stale state error for otherwise valid cameras.
                    pass
                elif state_error is not None:
                    errors[_STATE_SOURCE_ID] = state_error
                    self._update_status_error(_STATE_SOURCE_ID, state_error)
                else:
                    successful_sources += 1
                    source_timestamps[_STATE_SOURCE_ID] = state_timestamp
                    source_received_timestamps[_STATE_SOURCE_ID] = state_received
                    clock_domains[_STATE_SOURCE_ID] = state_domain
                    self._update_status_success(
                        _STATE_SOURCE_ID,
                        state_timestamp,
                        state_received,
                        state_domain,
                        stale=(state_timestamp is None or state_domain is None or state_domain != self.clock_domain),
                    )
                    if state_timestamp is None or state_domain is None or state_domain != self.clock_domain:
                        source_stale = True
                        errors[_STATE_SOURCE_ID] = "state capture timestamp is unknown or uses a foreign clock domain"

            source_results, source_timeouts = self._collect_source_results()
            for binding in self._sources:
                result = source_results.get(binding.source_id)
                if result is None:
                    message = source_timeouts.get(
                        binding.source_id,
                        "camera capture did not complete within the bounded source wait",
                    )
                    errors[binding.source_id] = message
                    continue
                if result.error is not None:
                    message = str(result.error)
                    errors[binding.source_id] = message
                    self._update_status_error(binding.source_id, message)
                    continue
                try:
                    source_frames = _normalise_frames(result.value)
                    if not source_frames:
                        raise ObservationError("camera source returned no frames")
                    names = [frame.name for frame in source_frames]
                    if len(names) != len(set(names)):
                        raise ObservationError("camera source returned duplicate frame names")
                    if any(frame.name in {existing.name for existing in frames} for frame in source_frames):
                        raise ObservationError("camera frame name was returned by two sources")
                except Exception as error:
                    message = str(error)
                    errors[binding.source_id] = message
                    self._update_status_error(binding.source_id, message)
                    continue
                successful_sources += 1
                frames.extend(source_frames)
                source_frame_names[binding.source_id] = tuple(names)
                timestamps = [frame.captured_timestamp_ns for frame in source_frames]
                receives = [frame.received_timestamp_ns for frame in source_frames]
                domains = {frame.clock_domain for frame in source_frames}
                source_timestamp = _aggregate_timestamp(timestamps)
                source_received = _aggregate_timestamp(receives)
                source_domain = next(iter(domains)) if len(domains) == 1 else None
                source_timestamps[binding.source_id] = source_timestamp
                source_received_timestamps[binding.source_id] = source_received
                clock_domains[binding.source_id] = source_domain
                for frame in source_frames:
                    source_timestamps[frame.name] = frame.captured_timestamp_ns
                    source_received_timestamps[frame.name] = frame.received_timestamp_ns
                    clock_domains[frame.name] = frame.clock_domain
                freshness_now = self._now_ns() if self.stale_after_ns is not None else None
                stale = any(
                    timestamp is None
                    or frame.clock_domain is None
                    or frame.clock_domain != self.clock_domain
                    or (
                        self.stale_after_ns is not None
                        and timestamp is not None
                        and (
                            freshness_now is not None
                            and (freshness_now < timestamp or freshness_now - timestamp > self.stale_after_ns)
                        )
                    )
                    for timestamp, frame in zip(timestamps, source_frames)
                )
                source_stale = source_stale or stale
                self._update_status_success(
                    binding.source_id,
                    source_timestamp,
                    source_received,
                    source_domain,
                    stale=stale,
                )
                if stale:
                    errors.setdefault(
                        binding.source_id,
                        "capture timestamp is unknown, foreign-domain, or older than the configured freshness limit",
                    )

            if generation != self.store.generation:
                # A reconnect invalidated everything sampled above.  Never
                # publish that pre-reconnect data under the new generation.
                raise ObservationUnavailableError(
                    None,
                    "capture crossed a source reconnect generation",
                )
            received_timestamp_ns = self._now_ns()
            known_timestamps = [timestamp for timestamp in source_timestamps.values() if timestamp is not None]
            known_domains = {clock_domains.get(source_id) for source_id, timestamp in source_timestamps.items()}
            skew_ns = (
                max(known_timestamps) - min(known_timestamps)
                if known_timestamps
                and len(known_timestamps) == len(source_timestamps)
                and len(known_domains) == 1
                and None not in known_domains
                else None
            )
            captured_timestamp_ns = max(known_timestamps) if known_timestamps else None
            published_timestamp_ns = self._now_ns()
            metadata: dict[str, Any] = {
                "clock_domain": self.clock_domain,
                "capture_semantics": "host_read_before_encode",
                "source_count": len(self._sources) + (1 if self._state_reader is not None else 0),
                "successful_source_count": successful_sources,
                "source_frames": source_frame_names,
                "source_status": {
                    source_id: status.to_dict() for source_id, status in self._statuses_snapshot().items()
                },
            }
            if state_metadata:
                metadata["state_metadata"] = state_metadata
            observation_id, snapshot_generation, sequence = self.store.next_observation_id()
            snapshot = ObservationSnapshot(
                observation_id=observation_id,
                service_instance_id=self.store.service_instance_id,
                generation=snapshot_generation,
                sequence=sequence,
                state=state,
                cameras=tuple(frames),
                metadata=metadata,
                errors=errors,
                stale=bool(errors or source_stale),
                available=successful_sources > 0,
                captured_timestamp_ns=captured_timestamp_ns,
                received_timestamp_ns=received_timestamp_ns,
                published_timestamp_ns=published_timestamp_ns,
                source_timestamps_ns=source_timestamps,
                source_received_timestamps_ns=source_received_timestamps,
                clock_domains=clock_domains,
                skew_ns=skew_ns,
            )
            published = self.store.publish(snapshot)
            with self._status_lock:
                self._last_error = (
                    None if not errors else "; ".join(f"{source}: {message}" for source, message in errors.items())
                )
            return published

    def _collect_source_results(
        self,
    ) -> tuple[dict[str, _SourceResult], dict[str, str]]:
        """Collect one bounded result per source without serializing sources.

        Every source has one daemon worker and one in-flight capture at most.
        A blocked worker remains isolated; the bounded wait lets other workers
        finish and their frames enter this observation normally.
        """

        results: dict[str, _SourceResult] = {}
        pending: dict[str, tuple[_SourceWorker, int]] = {}
        timeouts: dict[str, str] = {}
        for binding in self._sources:
            worker = self._source_workers[binding.source_id]
            completed = worker.take_completed()
            if completed is not None:
                timeout_key = (binding.source_id, completed.token)
                if timeout_key not in self._reported_timeouts:
                    results[binding.source_id] = completed
                    continue
                # This result arrived after the bounded wait expired.  It
                # belongs to the prior observation and cannot be fresh now.
                self._reported_timeouts.discard(timeout_key)
            token = worker.request()
            if token is not None:
                pending[binding.source_id] = (worker, token)
                continue
            # A completion can race the request check.  Consume it before
            # reporting a timeout; otherwise the next observation would lose
            # a valid frame despite the source never being blocked.
            completed = worker.take_completed()
            if completed is not None:
                timeout_key = (binding.source_id, completed.token)
                if timeout_key not in self._reported_timeouts:
                    results[binding.source_id] = completed
                    continue
                self._reported_timeouts.discard(timeout_key)
                token = worker.request()
                if token is not None:
                    pending[binding.source_id] = (worker, token)
                    continue
            active_token = worker.active_token or 0
            timeouts[binding.source_id] = "camera capture remains in flight; previous source call is bounded to one"
            self._report_source_timeout(binding.source_id, active_token)

        deadline = time.monotonic() + self.source_timeout_s
        while pending:
            completed_ids: list[str] = []
            for source_id, (worker, token) in pending.items():
                completed = worker.take_completed()
                if completed is None:
                    continue
                # A worker cannot publish an older result after a newer token,
                # but keep the check explicit at this boundary.
                if completed.token != token:
                    continue
                results[source_id] = completed
                self._reported_timeouts.discard((source_id, token))
                completed_ids.append(source_id)
            for source_id in completed_ids:
                pending.pop(source_id, None)
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.001, remaining))

        for source_id, (_worker, token) in pending.items():
            timeouts[source_id] = "camera capture exceeded source_timeout_s; source remains unavailable"
            self._report_source_timeout(source_id, token)
        return results, timeouts

    def _report_source_timeout(self, source_id: str, token: int) -> None:
        key = (source_id, token)
        if key in self._reported_timeouts:
            return
        self._reported_timeouts.add(key)
        self._update_status_error(source_id, "camera capture timed out or remains in flight")

    def _read_state(
        self,
    ) -> tuple[Any, int | None, int | None, str | None, str | None, Mapping[str, Any]]:
        if self._state_reader is None:
            return None, None, None, None, None, {}
        try:
            raw = self._state_reader()
            if raw is None:
                return None, None, None, None, None, {}
            # Receipt belongs to this host and is sampled only after the source
            # returns. Preserve any source capture timestamp and clock domain.
            received = self._now_ns()
            state, timestamp, domain, metadata = _normalise_state(raw)
            return state, timestamp, received, domain, None, metadata
        except Exception as error:
            # A failed read is received when the source reports the failure.
            received = self._now_ns()
            return None, None, received, None, str(error), {}

    def _now_ns(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ObservationError("observation clock must return a non-negative integer nanosecond value")
        return value

    def _statuses_snapshot(self) -> dict[str, SourceStatus]:
        with self._status_lock:
            return dict(self._statuses)

    def _update_status_success(
        self,
        source_id: str,
        captured_timestamp_ns: int | None,
        received_timestamp_ns: int | None,
        clock_domain: str | None,
        *,
        stale: bool,
    ) -> None:
        with self._status_lock:
            previous = self._statuses[source_id]
            self._statuses[source_id] = SourceStatus(
                source_id,
                self.store.generation,
                available=True,
                stale=stale,
                last_error=(
                    "capture timestamp is unknown, foreign-domain, or older than the configured freshness limit"
                    if stale
                    else None
                ),
                capture_count=previous.capture_count + 1,
                error_count=previous.error_count,
                last_captured_timestamp_ns=captured_timestamp_ns,
                last_received_timestamp_ns=received_timestamp_ns,
                clock_domain=clock_domain,
            )

    def _update_status_error(self, source_id: str, error: str) -> None:
        with self._status_lock:
            previous = self._statuses[source_id]
            self._statuses[source_id] = SourceStatus(
                source_id,
                self.store.generation,
                available=False,
                stale=True,
                last_error=error,
                capture_count=previous.capture_count,
                error_count=previous.error_count + 1,
                # Keep the last successful sample for diagnostics, while
                # marking the current source unavailable so it cannot be
                # mistaken for a fresh frame.
                last_captured_timestamp_ns=previous.last_captured_timestamp_ns,
                last_received_timestamp_ns=previous.last_received_timestamp_ns,
                clock_domain=previous.clock_domain,
            )

    def reconnect(self, source_id: str | None = None) -> int:
        """Invalidate retained IDs for an owner reconnect.

        This method only changes observation generation.  It does not open,
        close, or reconfigure a physical source; the device owner performs
        that lifecycle operation and calls this method once its boundary is
        explicit.
        """

        if source_id is not None:
            self._require_source(source_id)
        # Drop completed and late captures before advancing the store
        # generation.  Otherwise a result captured before reconnect could be
        # consumed by the first publish in the new generation.
        if source_id is None:
            for worker in self._source_workers.values():
                worker.invalidate()
        elif source_id in self._source_workers:
            self._source_workers[source_id].invalidate()
        generation = self.store.begin_generation()
        with self._status_lock:
            for key, status in self._statuses.items():
                self._statuses[key] = SourceStatus(
                    key,
                    generation,
                    available=False,
                    stale=True,
                    last_error=(
                        "source reconnect in progress" if source_id is None or key == source_id else status.last_error
                    ),
                    capture_count=status.capture_count,
                    error_count=status.error_count,
                    last_captured_timestamp_ns=status.last_captured_timestamp_ns,
                    last_received_timestamp_ns=status.last_received_timestamp_ns,
                    clock_domain=status.clock_domain,
                )
        return generation

    def mark_source_unavailable(self, source_id: str, reason: str) -> None:
        self._require_source(source_id)
        self._update_status_error(source_id, reason)

    def _require_source(self, source_id: str) -> None:
        with self._status_lock:
            if source_id not in self._statuses:
                raise ObservationUnavailableError(source_id, "source ID is unknown")

    def latest(self) -> ObservationSnapshot | None:
        return self.store.latest()

    def get(self, observation_id: str) -> ObservationSnapshot:
        return self.store.get(observation_id)

    def subscribe(
        self,
        max_queue: int | None = None,
        *,
        queue_size: int | None = None,
        replay_latest: bool = False,
    ) -> ObservationSubscription:
        value = self.subscription_queue_size if max_queue is None else max_queue
        return self.store.subscribe(
            value,
            queue_size=queue_size,
            replay_latest=replay_latest,
        )

    def close(self, timeout_s: float = 1.0) -> bool:
        """Stop the producer and close owned sources within a bounded budget."""

        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a number")
        if timeout_s < 0 or not math.isfinite(float(timeout_s)):
            raise ValueError("timeout_s must be finite and non-negative")
        with self._lifecycle_lock:
            if self._closed:
                return True
            self._closed = True
            self._stop_event.set()
            thread = self._thread
        deadline = time.monotonic() + float(timeout_s)
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
        complete = thread is None or not thread.is_alive()
        if complete:
            for binding in self._sources:
                worker = self._source_workers[binding.source_id]
                remaining = max(0.0, deadline - time.monotonic())
                if not worker.close(remaining):
                    complete = False
                    self._update_status_error(
                        binding.source_id,
                        "source worker did not stop within producer timeout",
                    )
        if complete and self.own_sources:
            for binding in self._sources:
                if binding.close is None:
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                finished, error = _bounded_close(binding.close, remaining)
                if not finished:
                    complete = False
                    self._update_status_error(
                        binding.source_id,
                        "source close exceeded producer timeout",
                    )
                elif error is not None:
                    complete = False
                    self._update_status_error(binding.source_id, str(error))
        self.store.close()
        return complete


def _as_reader(value: Any) -> Callable[[], Any] | None:
    if value is None:
        return None
    if callable(value):
        return value
    raise TypeError("state_reader must be callable")


def _normalise_frames(value: Any) -> tuple[CameraFrame, ...]:
    if isinstance(value, CameraFrame):
        return (value,)
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise TypeError("camera source must return CameraFrame values")
    try:
        frames = tuple(value)
    except TypeError as error:
        raise TypeError("camera source must return an iterable of CameraFrame values") from error
    if any(not isinstance(frame, CameraFrame) for frame in frames):
        raise TypeError("camera source returned a non-CameraFrame value")
    return frames


def _normalise_state(
    value: Any,
) -> tuple[Any, int | None, str | None, Mapping[str, Any]]:
    """Accept #24-style observations and simple fake-reader tuples."""

    metadata: Mapping[str, Any] = {}
    if hasattr(value, "values") and hasattr(value, "timestamp_s"):
        state = value.values
        raw_timestamp = value.timestamp_s
        raw_metadata = getattr(value, "metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        # Keep the legacy wall-clock field available for bindings and
        # recording, but never reinterpret it as a local monotonic capture
        # time.  A producer can only use the explicit pair below for
        # freshness.
        metadata.setdefault("timestamp_s", raw_timestamp)
        domain = metadata.get("clock_domain")
        valid_domain = _valid_domain(domain)
        explicit_capture = metadata.get("captured_timestamp_ns")
        if (
            valid_domain is not None
            and isinstance(explicit_capture, int)
            and not isinstance(explicit_capture, bool)
            and explicit_capture >= 0
        ):
            return state, explicit_capture, valid_domain, metadata
        return state, None, valid_domain, metadata
    if isinstance(value, tuple) and len(value) in (2, 3):
        candidate_timestamp = value[1]
        if isinstance(candidate_timestamp, (int, float)) and not isinstance(candidate_timestamp, bool):
            domain = value[2] if len(value) == 3 else None
            return (
                value[0],
                _timestamp_to_ns(candidate_timestamp),
                _valid_domain(domain),
                {},
            )
    return value, None, None, metadata


def _valid_domain(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("clock_domain must be a non-empty string")
    return value


def _seconds_to_ns(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or float(value) < 0:
        return None
    return int(float(value) * 1_000_000_000)


def _timestamp_to_ns(value: int | float) -> int | None:
    # Explicit integer values from fake readers are treated as nanoseconds;
    # fractional values are seconds, matching RobotObservation.timestamp_s.
    if isinstance(value, int):
        return value if value >= 0 else None
    return _seconds_to_ns(value)


def _aggregate_timestamp(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return max(known) if known else None


def _bounded_close(close: Callable[[], None], timeout_s: float) -> tuple[bool, BaseException | None]:
    result: list[BaseException | None] = [None]

    def run() -> None:
        try:
            close()
        except BaseException as error:  # pragma: no cover - defensive cleanup path
            result[0] = error

    thread = threading.Thread(target=run, name="rlinf-observation-close", daemon=True)
    thread.start()
    thread.join(max(0.0, timeout_s))
    return (not thread.is_alive(), result[0])


__all__ = ["ObservationProducer"]
