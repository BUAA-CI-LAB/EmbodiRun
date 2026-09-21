"""Service-level shared camera/state snapshots.

``SharedSensorHub`` owns one :class:`ObservationProducer` and store for the
control service.  Physical sources are attached lazily by identity, while
runtime profiles receive lightweight frame-name views over the same immutable
snapshot and encoded bytes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from embodirun.robots.sensors.cameras import CameraFrame, CameraSource

from .producer import ObservationProducer
from .store import ObservationStore, ObservationSubscription
from .values import ObservationSnapshot


class SharedSensorError(RuntimeError):
    """A runtime view cannot be resolved from the shared snapshot."""


class _NamespacedCameraSource:
    """Give one source unique internal frame names without changing bytes."""

    def __init__(self, source: CameraSource, prefix: str) -> None:
        self._source = source
        self.prefix = prefix

    def capture(self) -> tuple[CameraFrame, ...]:
        return tuple(
            replace(frame, name=f"{self.prefix}{index}:{frame.name}")
            for index, frame in enumerate(self._source.capture())
        )


class SharedSensorHub:
    """Keep one background producer/store for all service consumers."""

    def __init__(
        self,
        state_reader: Callable[[], Any | None],
        *,
        interval_s: float | None = None,
        service_instance_id: str | None = None,
    ) -> None:
        self.producer = ObservationProducer(
            state_reader=state_reader,
            interval_s=interval_s,
            service_instance_id=service_instance_id,
        )
        self._lifecycle_lock = threading.RLock()
        self._publish_lock = threading.Lock()
        self._source_prefixes: dict[str, str] = {}
        self._started = False
        self._closed = False

    @property
    def store(self) -> ObservationStore:
        return self.producer.store

    @property
    def started(self) -> bool:
        with self._lifecycle_lock:
            return self._started

    def has_source(self, source_id: str) -> bool:
        """Return whether a physical source is registered with the producer."""

        with self._lifecycle_lock:
            return source_id in self._source_prefixes

    def add_source(self, source_id: str, source: CameraSource) -> None:
        """Attach one newly owned physical source and start the producer."""

        with self._lifecycle_lock:
            if self._closed:
                raise SharedSensorError("shared sensor hub is closed")
            prefix = f"shared:{len(self._source_prefixes)}:{source_id}:"
            self._source_prefixes[source_id] = prefix
            self.producer.add_camera_source(
                source_id,
                _NamespacedCameraSource(source, prefix),
            )
            if not self._started:
                self.producer.start()
                self._started = True

    def ensure_started(self) -> None:
        """Start the state-only producer for an explicit robot observation."""

        with self._lifecycle_lock:
            if self._closed:
                raise SharedSensorError("shared sensor hub is closed")
            if not self._started:
                self.producer.start()
                self._started = True

    def snapshot(
        self,
        *,
        require_state: bool = False,
        required_source_ids: Sequence[str] = (),
        force_publish: bool = False,
    ) -> ObservationSnapshot:
        """Return one retained snapshot, publishing only when needed.

        ``require_state`` ensures the returned snapshot has state.  Use
        ``force_publish`` when a caller needs a new capture, such as a model
        execution step, so its frames and state share one observation ID.
        """

        self.ensure_started()
        with self._publish_lock:
            latest = self.producer.latest()
            required = tuple(dict.fromkeys(required_source_ids))
            latest_sources = latest.metadata.get("source_frames", {}) if latest is not None else {}
            needs_source = latest is None or any(
                (source_id not in latest_sources and source_id not in latest.source_timestamps_ns)
                or source_id in latest.errors
                for source_id in required
            )
            state_missing = require_state and (latest is None or latest.state is None)
            if force_publish or state_missing or needs_source:
                return self.producer.publish_once()
            return latest

    def latest(self) -> ObservationSnapshot | None:
        return self.producer.latest()

    def get_snapshot(self, observation_id: str) -> ObservationSnapshot:
        return self.producer.get(observation_id)

    def subscribe(
        self,
        max_queue: int = 8,
        *,
        replay_latest: bool = False,
    ) -> ObservationSubscription:
        self.ensure_started()
        return self.producer.subscribe(max_queue, replay_latest=replay_latest)

    def frames_for(
        self,
        snapshot: ObservationSnapshot,
        requested: Sequence[tuple[str, str]],
    ) -> tuple[CameraFrame, ...]:
        """Resolve ``(physical_source_id, runtime_name)`` views by bytes."""

        if not requested:
            return ()
        source_frame_names = snapshot.metadata.get("source_frames", {})
        if not isinstance(source_frame_names, Mapping):
            source_frame_names = {}
        by_name: dict[str, CameraFrame] = {}
        for frame in snapshot.cameras:
            if frame.name in by_name:
                raise SharedSensorError(f"shared snapshot contains duplicate frame {frame.name!r}")
            by_name[frame.name] = frame
        resolved: list[CameraFrame] = []
        used: set[str] = set()
        for source_id, runtime_name in requested:
            actual_names = source_frame_names.get(source_id, ())
            if not isinstance(actual_names, (tuple, list)):
                actual_names = ()
            prefix = self._source_prefixes.get(source_id, "")
            public_names = tuple(_public_frame_name(prefix, name) for name in actual_names)
            if len(actual_names) > 1 and public_names and runtime_name == public_names[0]:
                for actual_name, public_name in zip(actual_names, public_names):
                    if actual_name not in by_name or actual_name in used:
                        continue
                    resolved.append(replace(by_name[actual_name], name=public_name))
                    used.add(actual_name)
                continue
            actual_name: str | None = None
            if runtime_name in public_names:
                actual_name = actual_names[public_names.index(runtime_name)]
            elif len(actual_names) == 1:
                actual_name = actual_names[0]
            if actual_name is not None and actual_name in by_name and actual_name not in used:
                frame = by_name[actual_name]
                if frame.name != runtime_name:
                    frame = replace(frame, name=runtime_name)
                resolved.append(frame)
                used.add(actual_name)
                continue
            if not actual_names and runtime_name in by_name and runtime_name not in used:
                # A legacy source may expose several named streams even when
                # the profile declares only the first stream.  Preserve that
                # source's old ordered output for the canonical first name;
                # an alias still has to identify one exact stream below.
                resolved.append(by_name[runtime_name])
                used.add(runtime_name)
                continue
            if len(actual_names) > 1:
                raise SharedSensorError("multi-frame camera sources require an explicit frame-name mapping")
            raise SharedSensorError(
                f"snapshot {snapshot.observation_id!r} has no frame mapping for "
                f"source {source_id!r} as {runtime_name!r}"
            )
        return tuple(resolved)

    def view_status(
        self,
        snapshot: ObservationSnapshot,
        required_source_ids: Sequence[str],
        *,
        require_state: bool = False,
    ) -> dict[str, Any]:
        """Report availability for one runtime view plus global diagnostics."""

        required = list(dict.fromkeys(required_source_ids))
        if require_state:
            required.append("state")
        errors = {source_id: snapshot.errors[source_id] for source_id in required if source_id in snapshot.errors}
        timestamps = [snapshot.source_timestamps_ns.get(source_id) for source_id in required]
        domains = [snapshot.clock_domains.get(source_id) for source_id in required]
        known_timestamps = [value for value in timestamps if value is not None]
        known_domains = {value for value in domains if value is not None}
        aggregate_domain = snapshot.metadata.get("clock_domain")
        foreign_domain = bool(known_domains and (len(known_domains) != 1 or aggregate_domain not in known_domains))
        view_skew = (
            max(known_timestamps) - min(known_timestamps)
            if len(known_timestamps) == len(required) and len(known_domains) == 1 and len(required) > 0
            else None
        )
        missing = [
            source_id
            for source_id, timestamp in zip(required, timestamps)
            if source_id not in snapshot.source_timestamps_ns or timestamp is None
        ]
        stale = bool(errors or missing or foreign_domain)
        return {
            "available": not missing and not errors,
            "stale": stale,
            "errors": errors,
            "missing_sources": missing,
            "captured_timestamp_ns": max(known_timestamps) if known_timestamps else None,
            "received_timestamp_ns": max(
                value
                for source_id in required
                if (value := snapshot.source_received_timestamps_ns.get(source_id)) is not None
            )
            if any(snapshot.source_received_timestamps_ns.get(source_id) is not None for source_id in required)
            else None,
            "clock_domains": {source_id: snapshot.clock_domains.get(source_id) for source_id in required},
            "skew_ns": view_skew,
            "global": {
                "available": snapshot.available,
                "stale": snapshot.stale,
                "errors": dict(snapshot.errors),
            },
        }

    def get_media(self, observation_id: str, frame_name: str) -> CameraFrame:
        """Return one immutable source frame without copying another snapshot."""

        snapshot = self.get_snapshot(observation_id)
        for frame in snapshot.cameras:
            if frame.name == frame_name:
                return frame
        raise SharedSensorError(f"snapshot {observation_id!r} has no frame named {frame_name!r}")

    def close(self, timeout_s: float = 1.0) -> bool:
        """Stop the producer; physical owner release remains the service's job."""

        with self._lifecycle_lock:
            if self._closed:
                return True
            self._closed = True
        return self.producer.close(timeout_s=timeout_s)


def _public_frame_name(prefix: str, name: str) -> str:
    if not prefix or not name.startswith(prefix):
        return name
    rest = name[len(prefix) :]
    index, separator, public_name = rest.partition(":")
    return public_name if separator and index.isdigit() else rest


__all__ = ["SharedSensorError", "SharedSensorHub"]
