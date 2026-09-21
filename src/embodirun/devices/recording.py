"""Bounded, asynchronous raw observation recording.

The recorder consumes immutable :class:`ObservationSnapshot` values through a
bounded subscription.  It owns only its subscription and files: stopping a
recorder never closes the observation store or any physical source.  Camera
bytes are retained as received; this module does not claim an exposure time or
perform a second video encoding pass.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from .observations.store import ObservationStore, ObservationSubscription
from .observations.values import ObservationSnapshot

_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "access_token",
    "authorization",
    "private_key",
)
_MEDIA_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png"}


class RecordingError(RuntimeError):
    """The bounded recorder cannot start, write, or read its raw files."""


@dataclass(frozen=True, slots=True)
class ActionEvent:
    """An explicit action lifecycle event associated with an observation.

    ``executed`` is supplied by the action owner.  A proposal therefore stays
    a proposal in the recording; this class never promotes it to an executed
    action merely because it has an ``observation_id``.
    """

    action_id: str
    source: str
    stage: str
    executed: bool
    observation_id: str | None = None
    outcome: str | None = None
    timestamp_ns: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("action_id", "source", "stage"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.observation_id is not None and (
            not isinstance(self.observation_id, str) or not self.observation_id.strip()
        ):
            raise ValueError("observation_id must be a non-empty string or None")
        if not isinstance(self.executed, bool):
            raise TypeError("executed must be a boolean")
        if self.outcome is not None and not isinstance(self.outcome, str):
            raise TypeError("outcome must be a string or None")
        if self.timestamp_ns is not None and (
            isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer or None")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")


@dataclass(frozen=True, slots=True)
class RecordingStatus:
    """Detached, queryable state for one background recording."""

    recording_id: str
    state: str
    path: str
    observation_count: int
    action_count: int
    dropped_observations: int
    dropped_actions: int
    missing_frames: int
    missing_states: int
    bytes_written: int
    max_bytes: int
    incomplete: bool
    error: str | None
    started_timestamp_ns: int | None
    stopped_timestamp_ns: int | None


class ObservationRecorder:
    """Write a bounded raw snapshot timeline in a background worker.

    Files are ``observations.jsonl``, ``actions.jsonl`` and source-encoded
    camera bytes under ``media/``.  JSON references the media by relative
    path, so consumers can query records without copying frame bytes into
    every record.  Queue overflow, missing expected sources, storage errors,
    and incomplete shutdown remain visible in :meth:`status` and the final
    manifest.
    """

    def __init__(
        self,
        store: ObservationStore,
        root_dir: str | os.PathLike[str],
        recording_id: str,
        *,
        queue_size: int = 8,
        max_records: int = 10_000,
        max_bytes: int = 1_000_000_000,
        expected_frame_names: Iterable[str] = (),
        require_state: bool = False,
        clock_ns: Callable[[], int] | None = None,
        clock_domain: str = "host_monotonic_ns",
        expected_mount: str | os.PathLike[str] | None = None,
        mount_check: Callable[[Path], bool] | None = None,
    ) -> None:
        if not isinstance(store, ObservationStore):
            raise TypeError("store must be an ObservationStore")
        if isinstance(queue_size, bool) or not isinstance(queue_size, int):
            raise TypeError("queue_size must be an integer")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if isinstance(max_records, bool) or not isinstance(max_records, int):
            raise TypeError("max_records must be an integer")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if not isinstance(require_state, bool):
            raise TypeError("require_state must be a boolean")
        if not isinstance(clock_domain, str) or not clock_domain.strip():
            raise ValueError("clock_domain must be a non-empty string")
        if expected_mount is not None and not isinstance(expected_mount, (str, os.PathLike)):
            raise TypeError("expected_mount must be a path or None")
        if mount_check is not None and not callable(mount_check):
            raise TypeError("mount_check must be callable or None")
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")
        if recording_id in {".", ".."} or any(separator in recording_id for separator in ("/", "\\", "\x00")):
            raise ValueError("recording_id must be one safe path component")
        names = tuple(expected_frame_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("expected_frame_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("expected_frame_names must not contain duplicates")

        self.store = store
        self.root_dir = Path(root_dir).resolve()
        self.recording_id = recording_id
        self.queue_size = queue_size
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.expected_frame_names = frozenset(names)
        self.require_state = require_state
        self._clock_ns = clock_ns or time.monotonic_ns
        self.clock_domain = clock_domain
        self.expected_mount = Path(expected_mount).resolve() if expected_mount is not None else None
        self._mount_check = mount_check or os.path.ismount
        self._recording_dir = self.root_dir / recording_id
        self._observations_path = self._recording_dir / "observations.jsonl"
        self._actions_path = self._recording_dir / "actions.jsonl"
        self._manifest_path = self._recording_dir / "manifest.json"
        self._media_dir = self._recording_dir / "media"

        self._lock = threading.RLock()
        self._state = "created"
        self._error: str | None = None
        self._incomplete = False
        self._observation_count = 0
        self._action_count = 0
        self._dropped_actions = 0
        self._missing_frames = 0
        self._missing_states = 0
        self._bytes_written = 0
        self._started_timestamp_ns: int | None = None
        self._stopped_timestamp_ns: int | None = None
        self._last_sequence: int | None = None
        self._action_queue: deque[ActionEvent] = deque(maxlen=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscription: ObservationSubscription | None = None
        self._observations_file: Any = None
        self._actions_file: Any = None

    @property
    def path(self) -> Path:
        """The resolved directory containing this recording's raw files."""

        return self._recording_dir

    def start(self) -> ObservationRecorder:
        """Create files, subscribe, and start the non-blocking writer."""

        with self._lock:
            if self._state != "created":
                raise RecordingError(f"recording is already {self._state}")
            try:
                self._prepare_directory()
                self._observations_file = self._observations_path.open("x", encoding="utf-8")
                self._actions_file = self._actions_path.open("x", encoding="utf-8")
                self._subscription = self.store.subscribe(
                    max_queue=self.queue_size,
                    replay_latest=False,
                )
                self._started_timestamp_ns = self._now_ns()
                self._state = "running"
                self._write_manifest()
            except Exception as error:
                self._state = "failed"
                self._error = str(error)
                self._incomplete = True
                self._close_files()
                if self._subscription is not None:
                    self._subscription.close()
                    self._subscription = None
                raise RecordingError(str(error)) from error
            self._thread = threading.Thread(
                target=self._run,
                name=f"rlinf-observation-recorder-{self.recording_id}",
                daemon=True,
            )
            self._thread.start()
        return self

    def record_action(self, event: ActionEvent) -> bool:
        """Queue one explicit action event without performing file I/O.

        ``False`` means the recorder is stopped/failed or the event displaced
        an older queued event.  The displacement remains visible in status.
        """

        if not isinstance(event, ActionEvent):
            raise TypeError("event must be an ActionEvent")
        with self._lock:
            if self._state not in {"created", "running"}:
                return False
            displaced = len(self._action_queue) == self._action_queue.maxlen
            if displaced:
                self._dropped_actions += 1
                self._incomplete = True
            self._action_queue.append(event)
            return not displaced

    def stop(self, timeout_s: float = 1.0) -> bool:
        """Request bounded shutdown; return whether the writer joined."""

        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a number")
        if timeout_s < 0 or not math.isfinite(float(timeout_s)):
            raise ValueError("timeout_s must be finite and non-negative")
        with self._lock:
            if self._state == "created":
                self._state = "stopped"
                self._stopped_timestamp_ns = self._now_ns()
                return True
            if self._state in {"stopped", "failed"}:
                return True
            self._state = "stopping"
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(float(timeout_s))
        joined = thread is None or not thread.is_alive()
        if not joined:
            with self._lock:
                self._incomplete = True
                self._error = self._error or "recorder worker did not stop within timeout"
        return joined

    def status(self) -> RecordingStatus:
        """Return a detached status snapshot suitable for another service."""

        with self._lock:
            dropped_observations = self._subscription.dropped_count if self._subscription is not None else 0
            return RecordingStatus(
                recording_id=self.recording_id,
                state=self._state,
                path=str(self._recording_dir),
                observation_count=self._observation_count,
                action_count=self._action_count,
                dropped_observations=dropped_observations,
                dropped_actions=self._dropped_actions,
                missing_frames=self._missing_frames,
                missing_states=self._missing_states,
                bytes_written=self._bytes_written,
                max_bytes=self.max_bytes,
                incomplete=self._incomplete,
                error=self._error,
                started_timestamp_ns=self._started_timestamp_ns,
                stopped_timestamp_ns=self._stopped_timestamp_ns,
            )

    def iter_records(self) -> Iterable[dict[str, Any]]:
        """Read complete observation records without touching the store."""

        if not self._observations_path.exists():
            return iter(())

        def read() -> Iterable[dict[str, Any]]:
            with self._observations_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and value.get("kind") == "observation":
                        yield value

        return read()

    def get_record(self, observation_id: str) -> dict[str, Any] | None:
        """Find one raw observation record by its original observation ID."""

        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        return next(
            (record for record in self.iter_records() if record.get("observation_id") == observation_id),
            None,
        )

    def iter_actions(self) -> Iterable[dict[str, Any]]:
        """Read complete action lifecycle events, including proposals."""

        if not self._actions_path.exists():
            return iter(())

        def read() -> Iterable[dict[str, Any]]:
            with self._actions_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and value.get("kind") == "action":
                        yield value

        return read()

    def export_lerobot(self, destination: str | os.PathLike[str]) -> None:
        """State explicitly that LeRobot conversion is outside raw recording."""

        del destination
        raise NotImplementedError("LeRobot conversion is unsupported; read observations.jsonl and media/")

    def _prepare_directory(self) -> None:
        try:
            self._check_expected_mount()
            root = self.root_dir.resolve()
            candidate = self._recording_dir.resolve()
            candidate.relative_to(root)
            self._recording_dir.mkdir(parents=True, exist_ok=True)
            if any(self._recording_dir.iterdir()):
                raise RecordingError("recording directory already contains files")
            self._media_dir.mkdir()
        except RecordingError:
            raise
        except Exception as error:
            raise RecordingError(f"recording directory is unavailable: {error}") from error

    def _check_expected_mount(self) -> None:
        """Reject a missing mount before creating a fallback same-named folder."""

        if self.expected_mount is None:
            return
        mount = self.expected_mount
        if not mount.exists() or not mount.is_dir() or not self._mount_check(mount):
            raise RecordingError(f"expected recording mount is unavailable: {mount}")
        try:
            self.root_dir.resolve().relative_to(mount)
        except ValueError as error:
            raise RecordingError(f"recording root {self.root_dir} is outside expected mount {mount}") from error

    def _run(self) -> None:
        assert self._subscription is not None
        subscription = self._subscription
        drain_budget: int | None = None
        try:
            while True:
                self._drain_actions()
                timeout = 0.05 if not self._stop_event.is_set() else 0.0
                try:
                    snapshot = subscription.get(timeout=timeout)
                except Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                self._write_snapshot(snapshot)
                if self._stop_event.is_set():
                    drain_budget = self.queue_size if drain_budget is None else drain_budget - 1
                    if drain_budget <= 0:
                        self._incomplete = True
                        break
        except Exception as error:
            self._fail(str(error))
        finally:
            try:
                self._drain_actions(max_events=self.queue_size)
            except Exception as error:
                self._fail(str(error))
            subscription.close()
            with self._lock:
                self._close_files()
                if self._state != "failed":
                    self._state = "stopped"
                self._stopped_timestamp_ns = self._now_ns()
                try:
                    self._write_manifest()
                except Exception as error:
                    self._state = "failed"
                    self._incomplete = True
                    self._error = str(error)

    def _write_snapshot(self, snapshot: ObservationSnapshot) -> None:
        if not isinstance(snapshot, ObservationSnapshot):
            raise RecordingError("subscription yielded a non-observation value")
        with self._lock:
            if self._observation_count >= self.max_records:
                raise RecordingError("recording max_records retention limit reached")
            recording_index = self._observation_count + 1
            dropped = self._subscription.dropped_count if self._subscription is not None else 0
            if self._last_sequence is not None and snapshot.sequence > self._last_sequence + 1:
                gap = snapshot.sequence - self._last_sequence - 1
                self._incomplete = True
            else:
                gap = 0
            self._last_sequence = snapshot.sequence
            missing = sorted(self.expected_frame_names - {frame.name for frame in snapshot.cameras})
            if missing:
                self._missing_frames += len(missing)
                self._incomplete = True
            if self.require_state and snapshot.state is None:
                self._missing_states += 1
                self._incomplete = True
            if snapshot.errors:
                self._incomplete = True

        if gap:
            self._append_json(
                self._observations_file,
                {
                    "kind": "gap",
                    "reason": "subscription_overflow_or_unobserved_sequence",
                    "missing_observations": gap,
                    "dropped_observations": dropped,
                },
            )

        media: list[dict[str, Any]] = []
        # Store-local sequences restart after a reconnect generation.  The
        # recorder's own ordinal keeps media paths unique without hiding a
        # collision behind ``exist_ok=True``.
        media_snapshot_dir = self._media_dir / f"{recording_index:08d}"
        if snapshot.cameras:
            media_snapshot_dir.mkdir()
        for index, frame in enumerate(snapshot.cameras):
            suffix = _MEDIA_SUFFIX.get(frame.mime_type)
            if suffix is None:
                raise RecordingError(f"unsupported source media type {frame.mime_type!r}")
            relative = Path("media") / media_snapshot_dir.name / f"{index:03d}{suffix}"
            path = self._recording_dir / relative
            self._write_media(path, frame.data)
            media.append(
                {
                    "name": frame.name,
                    "mime_type": frame.mime_type,
                    "path": relative.as_posix(),
                    "bytes": len(frame.data),
                    "captured_timestamp_ns": frame.captured_timestamp_ns,
                    "received_timestamp_ns": frame.received_timestamp_ns,
                    "clock_domain": frame.clock_domain,
                    "profile": _json_safe(frame.profile),
                    "encoding": "source_encoded_bytes",
                    "encoding_timestamp_ns": None,
                }
            )
        recorded_timestamp = self._now_ns()
        snapshot_domain = snapshot.metadata.get("clock_domain")
        capture_to_record = None
        if (
            snapshot.captured_timestamp_ns is not None
            and snapshot_domain == self.clock_domain
            and recorded_timestamp >= snapshot.captured_timestamp_ns
        ):
            capture_to_record = recorded_timestamp - snapshot.captured_timestamp_ns
        self._append_json(
            self._observations_file,
            {
                "kind": "observation",
                "observation_id": snapshot.observation_id,
                "service_instance_id": snapshot.service_instance_id,
                "generation": snapshot.generation,
                "sequence": snapshot.sequence,
                "recording_index": recording_index,
                "state": _json_safe(snapshot.state),
                "cameras": media,
                "metadata": _json_safe(snapshot.metadata),
                "errors": _json_safe(snapshot.errors),
                "stale": snapshot.stale,
                "available": snapshot.available,
                "source_timestamps_ns": _json_safe(snapshot.source_timestamps_ns),
                "source_received_timestamps_ns": _json_safe(snapshot.source_received_timestamps_ns),
                "clock_domains": _json_safe(snapshot.clock_domains),
                "skew_ns": snapshot.skew_ns,
                "timestamps": {
                    "captured_timestamp_ns": snapshot.captured_timestamp_ns,
                    "received_timestamp_ns": snapshot.received_timestamp_ns,
                    "published_timestamp_ns": snapshot.published_timestamp_ns,
                    "recorded_timestamp_ns": recorded_timestamp,
                    "clock_domain": snapshot_domain,
                    "capture_to_record_ns": capture_to_record,
                },
                "missing_frames": missing,
                "missing_state": self.require_state and snapshot.state is None,
                "dropped_observations": dropped,
            },
        )
        with self._lock:
            self._observation_count += 1

    def _drain_actions(self, *, max_events: int | None = None) -> None:
        count = 0
        while max_events is None or count < max_events:
            with self._lock:
                if not self._action_queue:
                    return
                event = self._action_queue.popleft()
            self._append_json(
                self._actions_file,
                {
                    "kind": "action",
                    "action_id": event.action_id,
                    "observation_id": event.observation_id,
                    "source": event.source,
                    "stage": event.stage,
                    "executed": event.executed,
                    "outcome": event.outcome,
                    "timestamp_ns": event.timestamp_ns,
                    "payload": _json_safe(event.payload),
                },
            )
            with self._lock:
                self._action_count += 1
            count += 1

    def _write_media(self, path: Path, data: bytes) -> None:
        with self._lock:
            if self._bytes_written + len(data) > self.max_bytes:
                raise RecordingError("recording max_bytes retention limit reached")
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        with self._lock:
            self._bytes_written += len(data)

    def _append_json(self, stream: Any, value: Mapping[str, Any]) -> None:
        if stream is None:
            raise RecordingError("recording file is closed")
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self._lock:
            if self._bytes_written + len(encoded) > self.max_bytes:
                raise RecordingError("recording max_bytes retention limit reached")
        stream.write(encoded.decode())
        stream.flush()
        with self._lock:
            self._bytes_written += len(encoded)

    def _fail(self, error: str) -> None:
        with self._lock:
            self._state = "failed"
            self._error = error
            self._incomplete = True
            self._stop_event.set()

    def _write_manifest(self) -> None:
        temporary = self._manifest_path.with_name(".manifest.json.tmp")
        payload = {
            "recording_id": self.recording_id,
            "state": self._state,
            "clock_domain": self.clock_domain,
            "started_timestamp_ns": self._started_timestamp_ns,
            "stopped_timestamp_ns": self._stopped_timestamp_ns,
            "observation_count": self._observation_count,
            "action_count": self._action_count,
            "dropped_observations": self.status().dropped_observations,
            "dropped_actions": self._dropped_actions,
            "missing_frames": self._missing_frames,
            "missing_states": self._missing_states,
            "bytes_written": self._bytes_written,
            "incomplete": self._incomplete,
            "error": self._error,
        }
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._manifest_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def _close_files(self) -> None:
        for name in ("_observations_file", "_actions_file"):
            stream = getattr(self, name)
            if stream is None:
                continue
            try:
                stream.close()
            finally:
                setattr(self, name, None)

    def _now_ns(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecordingError("recorder clock must return non-negative nanoseconds")
        return value


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    """Convert immutable snapshot values to JSON without persisting credentials."""

    if key is not None and any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(item_key): _json_safe(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    return {"type": type(value).__name__, "value": "<unsupported>"}


__all__ = ["ActionEvent", "ObservationRecorder", "RecordingError", "RecordingStatus"]
