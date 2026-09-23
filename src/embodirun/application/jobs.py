"""Persistent, idempotent short-job registration for control APIs.

This module owns request identity and lifecycle bookkeeping only.  It does not
open a robot connection or import an SDK.  A caller supplies a short execution
callback and, optionally, a cancellation callback.  Device authority and
physical stop confirmation remain with the injected control layer.

The SQLite primary key is ``(caller_id, session_id, request_id)``.  Parameters
are normalized JSON values and compared by their canonical ordinary values;
they are never reduced to a hash.  An unfinished row found after restart is
marked ``unknown`` and is never replayed.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import TaskRequest, TaskResult
from .job_store import SQLiteJobStore


class JobStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class DispatchStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class InferenceStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PhysicalStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    STOP_UNCONFIRMED = "stop_unconfirmed"
    UNKNOWN = "unknown"


_TERMINAL = frozenset(
    {
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
        JobStatus.UNKNOWN.value,
    }
)
_MAX_EVENT_DETAIL = 512
_MAX_ERROR_DETAIL = 1024


class JobError(RuntimeError):
    """Base error for registration and lifecycle operations."""


class JobConflict(JobError):
    """The request identity already has different normalized parameters."""


class JobCapacityExceeded(JobError):
    """The bounded historical job store cannot accept another identity."""


class JobNotFound(JobError):
    """No job exists for the supplied request identity."""


class JobOwnershipError(PermissionError, JobError):
    """The caller/session does not own the requested identity."""


class JobClosed(JobError):
    """The registry has been closed and accepts no new work."""


class JobTimeout(TimeoutError, JobError):
    """Waiting timed out; the job remains queryable by its original ID."""

    def __init__(self, handle: JobHandle, record: JobRecord):
        self.handle = handle
        self.record = record
        super().__init__(f"job {record.request_id!r} did not finish before the timeout")


class JobUnknownError(JobError):
    """The result cannot be established, usually after a restart or close."""

    def __init__(self, record: JobRecord):
        self.record = record
        super().__init__(f"job {record.request_id!r} has unknown result")


class JobCancelledError(JobError):
    """The job stopped before completion; physical stop is reported separately."""

    def __init__(self, record: JobRecord):
        self.record = record
        super().__init__(f"job {record.request_id!r} was cancelled")


class JobExecutionError(JobError):
    """The injected execution callback failed."""

    def __init__(self, record: JobRecord):
        self.record = record
        detail = record.error or "execution callback failed"
        super().__init__(f"job {record.request_id!r} failed: {detail}")


@dataclass(frozen=True, slots=True)
class JobKey:
    caller_id: str
    session_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class JobEvent:
    name: str
    at_s: float
    duration_s: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class JobRecord:
    caller_id: str
    session_id: str
    request_id: str
    run_id: str
    parameters: Mapping[str, Any]
    status: JobStatus
    dispatch_status: DispatchStatus
    inference_status: InferenceStatus
    physical_status: PhysicalStatus
    cancel_requested: bool
    result: Any
    error: str | None
    events: tuple[JobEvent, ...]
    events_truncated: bool
    created_at_s: float
    updated_at_s: float
    started_at_s: float | None
    finished_at_s: float | None

    @property
    def key(self) -> JobKey:
        return JobKey(self.caller_id, self.session_id, self.request_id)


@dataclass(slots=True)
class _Runtime:
    key: JobKey
    execute: Callable[[threading.Event], Any]
    cancel: Callable[[threading.Event], Any] | None
    cancel_event: threading.Event
    cancel_invoked: bool = False


class JobHandle:
    """Stable handle for a job; all operations retain caller/session scope."""

    def __init__(self, registry: JobRegistry, key: JobKey, run_id: str) -> None:
        self._registry = registry
        self.key = key
        self.run_id = run_id

    @property
    def caller_id(self) -> str:
        return self.key.caller_id

    @property
    def session_id(self) -> str:
        return self.key.session_id

    @property
    def request_id(self) -> str:
        return self.key.request_id

    def inspect(self) -> JobRecord:
        return self._registry.inspect(
            self.key.caller_id,
            self.key.session_id,
            self.key.request_id,
        )

    def cancel(self) -> JobRecord:
        return self._registry.cancel(
            self.key.caller_id,
            self.key.session_id,
            self.key.request_id,
        )

    def wait(self, timeout_s: float | None = None) -> Any:
        return self._registry.wait(self, timeout_s=timeout_s)


class JobRegistry:
    """Bounded SQLite-backed acceptance and execution registry."""

    def __init__(
        self,
        database_path: str | Path = ":memory:",
        *,
        max_jobs: int = 256,
        max_events: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs <= 0:
            raise ValueError("max_jobs must be a positive integer")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self._store = SQLiteJobStore(database_path)
        self.database_path = self._store.database_path
        self.max_jobs = max_jobs
        self.max_events = max_events
        self._clock = clock
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._runtimes: dict[JobKey, _Runtime] = {}
        self._workers: dict[JobKey, threading.Thread] = {}
        self._cancel_workers: set[threading.Thread] = set()
        self._results: dict[JobKey, Any] = {}
        self._closed = False
        self._store_closed = False
        with self._condition:
            self._recover_inflight_locked()

    def submit(
        self,
        caller_id: str,
        session_id: str,
        request_id: str,
        parameters: Mapping[str, Any],
        execute: Callable[[threading.Event], Any],
        cancel: Callable[[threading.Event], Any] | None = None,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
        inference_requested: bool = True,
    ) -> JobHandle | Any:
        """Atomically accept or reuse a request, then run one short callback."""

        key = self._key(caller_id, session_id, request_id)
        normalized = self._normalize_parameters(parameters)
        parameters_json = SQLiteJobStore.canonical_json(normalized)
        if not callable(execute):
            raise TypeError("execute must be callable")
        if cancel is not None and not callable(cancel):
            raise TypeError("cancel must be callable")
        if not isinstance(inference_requested, bool):
            raise TypeError("inference_requested must be a boolean")
        self._validate_timeout(timeout_s)

        with self._condition:
            self._require_open_locked()
            existing = self._fetch_row_locked(key)
            if existing is not None:
                if existing["parameters_json"] != parameters_json:
                    raise JobConflict(f"request_id {request_id!r} already has different parameters")
                handle = JobHandle(self, key, existing["run_id"])
            else:
                self._begin_locked()
                try:
                    # Re-read after BEGIN IMMEDIATE.  Another registry
                    # process may have accepted this identity between the
                    # optimistic read above and this transaction.
                    existing = self._fetch_row_locked(key)
                    if existing is not None:
                        if existing["parameters_json"] != parameters_json:
                            raise JobConflict(f"request_id {request_id!r} already has different parameters")
                        run_id = existing["run_id"]
                    else:
                        count = self._store.count()
                        if count >= self.max_jobs:
                            raise JobCapacityExceeded("job history capacity is full")
                        now = self._clock()
                        run_id = f"run-{uuid.uuid4().hex}"
                        events = self._event_list(
                            JobEvent("accepted", now),
                        )
                        self._store.insert(
                            key,
                            run_id=run_id,
                            parameters_json=parameters_json,
                            status=JobStatus.ACCEPTED.value,
                            dispatch_status=DispatchStatus.ACCEPTED.value,
                            inference_status=(
                                InferenceStatus.NOT_STARTED.value
                                if inference_requested
                                else InferenceStatus.NOT_REQUESTED.value
                            ),
                            physical_status=PhysicalStatus.NOT_REQUESTED.value,
                            events_json=SQLiteJobStore.canonical_json(events),
                            now=now,
                        )
                    self._commit_locked()
                except Exception:
                    self._rollback_locked()
                    raise
                handle = JobHandle(self, key, run_id)
                if existing is None:
                    runtime = _Runtime(key, execute, cancel, threading.Event())
                    self._runtimes[key] = runtime
                    worker = threading.Thread(
                        target=self._run_job,
                        args=(key,),
                        name=f"embodirun-job-{run_id}",
                        daemon=True,
                    )
                    self._workers[key] = worker
                    worker.start()
            self._condition.notify_all()

        if wait:
            return handle.wait(timeout_s=timeout_s)
        return handle

    def submit_task(
        self,
        request: TaskRequest,
        *,
        caller_id: str,
        session_id: str,
        execute: Callable[[threading.Event], TaskResult | Mapping[str, Any]],
        cancel: Callable[[threading.Event], Any] | None = None,
        wait: bool = True,
        timeout_s: float | None = None,
    ) -> JobHandle | TaskResult | Mapping[str, Any]:
        """Submit a legacy TaskRequest while preserving synchronous results."""

        return self.submit(
            caller_id,
            session_id,
            request.request_id,
            request.to_payload(),
            execute,
            cancel,
            wait=wait,
            timeout_s=timeout_s,
        )

    def inspect(self, caller_id: str, session_id: str, request_id: str) -> JobRecord:
        key = self._key(caller_id, session_id, request_id)
        with self._condition:
            self._require_open_locked()
            row = self._owned_row_locked(key)
            return self._record_locked(row)

    def get_handle(self, caller_id: str, session_id: str, request_id: str) -> JobHandle:
        """Return a scoped handle for an existing request without re-dispatching it.

        Applications use this when an idempotent retry arrives.  Reusing the
        original handle keeps a running job waitable while avoiding a second
        execution callback; ownership is checked under the same condition
        lock as :meth:`inspect` and :meth:`cancel`.
        """

        key = self._key(caller_id, session_id, request_id)
        with self._condition:
            self._require_open_locked()
            row = self._owned_row_locked(key)
            return JobHandle(self, key, row["run_id"])

    def cancel(self, caller_id: str, session_id: str, request_id: str) -> JobRecord:
        key = self._key(caller_id, session_id, request_id)
        callback: Callable[[threading.Event], Any] | None = None
        runtime: _Runtime | None = None
        with self._condition:
            self._require_open_locked()
            row = self._owned_row_locked(key)
            if row["status"] in _TERMINAL:
                return self._record_locked(row)
            runtime = self._runtimes.get(key)
            if runtime is None:
                # A process restart cannot have a callback to invoke.  The
                # recovery path already made the result unknown.
                return self._record_locked(row)
            runtime.cancel_event.set()
            if not runtime.cancel_invoked:
                runtime.cancel_invoked = True
                callback = runtime.cancel
            now = self._clock()
            self._begin_locked()
            try:
                events, truncated = self._append_event_locked(
                    row,
                    JobEvent("cancel_requested", now),
                )
                self._store.update(
                    key,
                    status=JobStatus.CANCEL_REQUESTED.value,
                    dispatch_status=DispatchStatus.CANCEL_REQUESTED.value,
                    inference_status=InferenceStatus.CANCEL_REQUESTED.value,
                    physical_status=PhysicalStatus.STOP_REQUESTED.value,
                    cancel_requested=1,
                    events_json=SQLiteJobStore.canonical_json(events),
                    events_truncated=int(truncated),
                    updated_at_s=now,
                )
                self._commit_locked()
            except Exception:
                self._rollback_locked()
                raise
            record = self._record_locked(self._fetch_row_locked(key))
            self._condition.notify_all()
        callback_thread: threading.Thread | None = None
        if callback is not None:
            callback_thread = threading.Thread(
                target=self._run_cancel_callback,
                args=(key, runtime, callback),
                name=f"embodirun-job-cancel-{runtime.key.request_id}",
                daemon=True,
            )
            # Register before starting so close() cannot release the SQLite
            # flock while a cancellation callback still owns the store.
            with self._condition:
                if not self._closed:
                    self._cancel_workers.add(callback_thread)
                else:
                    callback_thread = None
            if callback_thread is not None:
                callback_thread.start()
        return record

    def record_stage(
        self,
        caller_id: str,
        session_id: str,
        request_id: str,
        stage: str,
        duration_s: float,
        *,
        detail: str | None = None,
    ) -> JobRecord:
        """Append one bounded stage event for later diagnostics."""

        key = self._key(caller_id, session_id, request_id)
        if not isinstance(stage, str) or not stage.strip() or len(stage) > 128:
            raise ValueError("stage must be a non-empty name of at most 128 characters")
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or duration_s < 0
        ):
            raise ValueError("duration_s must be finite and non-negative")
        if detail is not None and not isinstance(detail, str):
            raise TypeError("detail must be a string or None")
        if detail is not None and len(detail) > _MAX_EVENT_DETAIL:
            raise ValueError(f"detail must be at most {_MAX_EVENT_DETAIL} characters")
        with self._condition:
            self._require_open_locked()
            row = self._owned_row_locked(key)
            now = self._clock()
            self._begin_locked()
            try:
                events, truncated = self._append_event_locked(
                    row,
                    JobEvent(stage, now, float(duration_s), detail),
                )
                self._store.update(
                    key,
                    events_json=SQLiteJobStore.canonical_json(events),
                    events_truncated=int(truncated),
                    updated_at_s=now,
                )
                self._commit_locked()
            except Exception:
                self._rollback_locked()
                raise
            return self._record_locked(self._fetch_row_locked(key))

    def wait(self, handle: JobHandle, *, timeout_s: float | None = None) -> Any:
        if not isinstance(handle, JobHandle) or handle._registry is not self:
            raise TypeError("wait requires a handle from this registry")
        self._validate_timeout(timeout_s)
        deadline = None if timeout_s is None else self._clock() + timeout_s
        with self._condition:
            while True:
                self._require_open_locked()
                row = self._owned_row_locked(handle.key)
                record = self._record_locked(row)
                if record.status.value in _TERMINAL:
                    return self._result_or_raise(record)
                if deadline is not None:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise JobTimeout(handle, record)
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def close(self, *, wait_s: float = 0.1) -> bool:
        """Request shutdown and close persistence only after workers drain.

        ``False`` means a callback/worker is still running and the SQLite
        connection plus its sidecar flock remain owned.  Callers may retry
        with a larger bounded wait; this prevents a second registry from
        treating a still-live callback as a crashed process.
        """

        self._validate_timeout(wait_s)
        with self._condition:
            if self._store_closed:
                return True
            if not self._closed:
                self._begin_locked()
                try:
                    rows = self._store.rows_not_status(tuple(_TERMINAL))
                    for row in rows:
                        key = JobKey(row["caller_id"], row["session_id"], row["request_id"])
                        events, truncated = self._append_event_locked(
                            row,
                            JobEvent("closed_unknown", self._clock()),
                        )
                        physical = (
                            PhysicalStatus.STOP_UNCONFIRMED.value
                            if row["cancel_requested"]
                            else PhysicalStatus.UNKNOWN.value
                        )
                        inference = (
                            InferenceStatus.NOT_REQUESTED.value
                            if row["inference_status"] == InferenceStatus.NOT_REQUESTED.value
                            else InferenceStatus.UNKNOWN.value
                        )
                        self._store.update(
                            key,
                            status=JobStatus.UNKNOWN.value,
                            dispatch_status=DispatchStatus.UNKNOWN.value,
                            inference_status=inference,
                            physical_status=physical,
                            events_json=SQLiteJobStore.canonical_json(events),
                            events_truncated=int(truncated),
                            updated_at_s=self._clock(),
                        )
                    self._commit_locked()
                except Exception:
                    self._rollback_locked()
                    raise
                self._closed = True
                runtimes = tuple(self._runtimes.values())
                for runtime in runtimes:
                    runtime.cancel_event.set()
                self._condition.notify_all()
            workers = tuple(self._workers.values()) + tuple(self._cancel_workers)
        deadline = time.monotonic() + wait_s
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(remaining)
        if any(worker.is_alive() for worker in workers):
            return False
        with self._condition:
            if not self._store_closed:
                self._store.close()
                self._store_closed = True
            return True

    def _run_job(self, key: JobKey) -> None:
        with self._condition:
            runtime = self._runtimes.get(key)
            if runtime is None or self._closed:
                return
            row = self._fetch_row_locked(key)
            if row is None:
                return
            if row["status"] == JobStatus.CANCEL_REQUESTED.value:
                now = self._clock()
                physical = (
                    PhysicalStatus.STOP_REQUESTED.value
                    if runtime.cancel is not None
                    else PhysicalStatus.STOP_UNCONFIRMED.value
                )
                events, truncated = self._append_event_locked(
                    row,
                    JobEvent("cancelled_before_dispatch", now),
                )
                self._begin_locked()
                try:
                    self._store.update(
                        key,
                        status=JobStatus.CANCELLED.value,
                        dispatch_status=DispatchStatus.CANCELLED.value,
                        inference_status=(
                            InferenceStatus.NOT_REQUESTED.value
                            if row["inference_status"] == InferenceStatus.NOT_REQUESTED.value
                            else InferenceStatus.CANCELLED.value
                        ),
                        physical_status=physical,
                        events_json=SQLiteJobStore.canonical_json(events),
                        events_truncated=int(truncated),
                        finished_at_s=now,
                        updated_at_s=now,
                    )
                    self._commit_locked()
                except Exception:
                    self._rollback_locked()
                    raise
                self._runtimes.pop(key, None)
                self._condition.notify_all()
                return
            if row["status"] != JobStatus.ACCEPTED.value:
                return
            now = self._clock()
            events, truncated = self._append_event_locked(row, JobEvent("running", now))
            self._begin_locked()
            try:
                self._store.update(
                    key,
                    status=JobStatus.RUNNING.value,
                    dispatch_status=DispatchStatus.RUNNING.value,
                    inference_status=(
                        InferenceStatus.RUNNING.value
                        if row["inference_status"] != InferenceStatus.NOT_REQUESTED.value
                        else InferenceStatus.NOT_REQUESTED.value
                    ),
                    events_json=SQLiteJobStore.canonical_json(events),
                    events_truncated=int(truncated),
                    started_at_s=now,
                    updated_at_s=now,
                )
                self._commit_locked()
            except Exception:
                self._rollback_locked()
                raise
            self._condition.notify_all()

        value: Any = None
        callback_error: BaseException | None = None
        try:
            value = runtime.execute(runtime.cancel_event)
        except BaseException as error:  # callback boundary; preserve diagnostic
            callback_error = error

        with self._condition:
            if self._closed:
                self._runtimes.pop(key, None)
                self._condition.notify_all()
                return
            row = self._fetch_row_locked(key)
            if row is None:
                self._runtimes.pop(key, None)
                return
            cancelled = runtime.cancel_event.is_set() or bool(row["cancel_requested"])
            inference_not_requested = row["inference_status"] == InferenceStatus.NOT_REQUESTED.value
            now = self._clock()
            if cancelled:
                status = JobStatus.CANCELLED.value
                dispatch = DispatchStatus.CANCELLED.value
                inference = (
                    InferenceStatus.NOT_REQUESTED.value if inference_not_requested else InferenceStatus.CANCELLED.value
                )
                physical = row["physical_status"]
                if physical == PhysicalStatus.STOP_REQUESTED.value:
                    physical = PhysicalStatus.STOP_UNCONFIRMED.value
                error_text = self._error_text(callback_error)
                event_name = "cancelled"
                stored_result = None
            elif callback_error is not None:
                status = JobStatus.FAILED.value
                dispatch = DispatchStatus.FAILED.value
                inference = (
                    InferenceStatus.NOT_REQUESTED.value if inference_not_requested else InferenceStatus.FAILED.value
                )
                physical = PhysicalStatus.UNKNOWN.value
                error_text = self._error_text(callback_error)
                event_name = "failed"
                stored_result = None
            else:
                status = JobStatus.COMPLETED.value
                dispatch = DispatchStatus.COMPLETED.value
                inference = (
                    InferenceStatus.NOT_REQUESTED.value if inference_not_requested else InferenceStatus.COMPLETED.value
                )
                physical = self._physical_from_value(value)
                error_text = None
                event_name = "completed"
                stored_result = SQLiteJobStore.encode_result(value)
                self._results[key] = value
            events, truncated = self._append_event_locked(row, JobEvent(event_name, now))
            self._begin_locked()
            try:
                self._store.update(
                    key,
                    status=status,
                    dispatch_status=dispatch,
                    inference_status=inference,
                    physical_status=physical,
                    result_json=stored_result,
                    error=error_text,
                    events_json=SQLiteJobStore.canonical_json(events),
                    events_truncated=int(truncated),
                    finished_at_s=now,
                    updated_at_s=now,
                )
                self._commit_locked()
            except Exception:
                self._rollback_locked()
                raise
            self._runtimes.pop(key, None)
            self._condition.notify_all()

    def _run_cancel_callback(
        self,
        key: JobKey,
        runtime: _Runtime,
        callback: Callable[[threading.Event], Any],
    ) -> None:
        current = threading.current_thread()
        try:
            self._run_cancel_callback_impl(key, runtime, callback)
        finally:
            with self._condition:
                self._cancel_workers.discard(current)
                self._condition.notify_all()

    def _run_cancel_callback_impl(
        self,
        key: JobKey,
        runtime: _Runtime,
        callback: Callable[[threading.Event], Any],
    ) -> None:
        value: Any = None
        error: BaseException | None = None
        try:
            value = callback(runtime.cancel_event)
        except BaseException as exc:
            error = exc
        with self._condition:
            if self._closed:
                return
            row = self._fetch_row_locked(key)
            if row is None or row["status"] not in {
                JobStatus.CANCEL_REQUESTED.value,
                JobStatus.CANCELLED.value,
            }:
                return
            physical = self._physical_from_cancel(value) if error is None else PhysicalStatus.STOP_UNCONFIRMED.value
            detail = self._error_text(error)
            events, truncated = self._append_event_locked(
                row,
                JobEvent("cancel_result", self._clock(), detail=detail),
            )
            self._begin_locked()
            try:
                self._store.update(
                    key,
                    physical_status=physical,
                    error=detail if detail is not None else row["error"],
                    events_json=SQLiteJobStore.canonical_json(events),
                    events_truncated=int(truncated),
                    updated_at_s=self._clock(),
                )
                self._commit_locked()
            except Exception:
                self._rollback_locked()
                raise
            self._condition.notify_all()

    def _recover_inflight_locked(self) -> None:
        rows = self._store.rows_with_status(
            (
                JobStatus.ACCEPTED.value,
                JobStatus.RUNNING.value,
                JobStatus.CANCEL_REQUESTED.value,
            )
        )
        if not rows:
            return
        self._begin_locked()
        try:
            for row in rows:
                events, truncated = self._append_event_locked(
                    row,
                    JobEvent("recovered_unknown", self._clock()),
                )
                physical = (
                    PhysicalStatus.STOP_UNCONFIRMED.value if row["cancel_requested"] else PhysicalStatus.UNKNOWN.value
                )
                inference = (
                    InferenceStatus.NOT_REQUESTED.value
                    if row["inference_status"] == InferenceStatus.NOT_REQUESTED.value
                    else InferenceStatus.UNKNOWN.value
                )
                self._store.update(
                    JobKey(row["caller_id"], row["session_id"], row["request_id"]),
                    status=JobStatus.UNKNOWN.value,
                    dispatch_status=DispatchStatus.UNKNOWN.value,
                    inference_status=inference,
                    physical_status=physical,
                    events_json=SQLiteJobStore.canonical_json(events),
                    events_truncated=int(truncated),
                    updated_at_s=self._clock(),
                )
            self._commit_locked()
        except Exception:
            self._rollback_locked()
            raise

    def _owned_row_locked(self, key: JobKey) -> sqlite3.Row:
        row = self._fetch_row_locked(key)
        if row is not None:
            return row
        owners = self._store.owners_for_request(key.request_id)
        if owners:
            raise JobOwnershipError(f"request_id {key.request_id!r} does not belong to this caller/session")
        raise JobNotFound(f"job {key.request_id!r} was not found")

    def _record_locked(self, row: sqlite3.Row | None) -> JobRecord:
        if row is None:
            raise JobNotFound("job row disappeared")
        key = JobKey(row["caller_id"], row["session_id"], row["request_id"])
        result_marker = object()
        value = self._results.get(key, result_marker)
        if value is result_marker:
            value = None if row["result_json"] is None else SQLiteJobStore.decode_result(row["result_json"])
        events_payload = SQLiteJobStore.decode_json(row["events_json"])
        events = tuple(
            JobEvent(
                name=item["name"],
                at_s=float(item["at_s"]),
                duration_s=item.get("duration_s"),
                detail=item.get("detail"),
            )
            for item in events_payload
        )
        return JobRecord(
            caller_id=key.caller_id,
            session_id=key.session_id,
            request_id=key.request_id,
            run_id=row["run_id"],
            parameters=SQLiteJobStore.decode_json(row["parameters_json"]),
            status=JobStatus(row["status"]),
            dispatch_status=DispatchStatus(row["dispatch_status"]),
            inference_status=InferenceStatus(row["inference_status"]),
            physical_status=PhysicalStatus(row["physical_status"]),
            cancel_requested=bool(row["cancel_requested"]),
            result=value,
            error=row["error"],
            events=events,
            events_truncated=bool(row["events_truncated"]),
            created_at_s=float(row["created_at_s"]),
            updated_at_s=float(row["updated_at_s"]),
            started_at_s=(None if row["started_at_s"] is None else float(row["started_at_s"])),
            finished_at_s=(None if row["finished_at_s"] is None else float(row["finished_at_s"])),
        )

    def _result_or_raise(self, record: JobRecord) -> Any:
        if record.status is JobStatus.COMPLETED:
            return record.result
        if record.status is JobStatus.UNKNOWN:
            raise JobUnknownError(record)
        if record.status is JobStatus.CANCELLED:
            raise JobCancelledError(record)
        if record.status is JobStatus.FAILED:
            raise JobExecutionError(record)
        raise RuntimeError(f"job is not terminal: {record.status.value}")

    def _fetch_row_locked(self, key: JobKey) -> sqlite3.Row | None:
        return self._store.fetch(key)

    def _append_event_locked(
        self,
        row: sqlite3.Row,
        event: JobEvent,
    ) -> tuple[list[dict[str, Any]], bool]:
        return SQLiteJobStore.append_event(
            row,
            {
                "name": event.name,
                "at_s": event.at_s,
                "duration_s": event.duration_s,
                "detail": event.detail,
            },
            max_events=self.max_events,
        )

    def _event_list(self, event: JobEvent) -> list[dict[str, Any]]:
        return [
            {
                "name": event.name,
                "at_s": event.at_s,
                "duration_s": event.duration_s,
                "detail": event.detail,
            }
        ]

    @staticmethod
    def _physical_from_cancel(value: Any) -> str:
        if isinstance(value, Mapping):
            status = value.get("physical_status")
            if isinstance(status, str) and status in {item.value for item in PhysicalStatus}:
                return status
            confirmed = value.get("stop_confirmed")
            if confirmed is True:
                return PhysicalStatus.STOPPED.value
            if confirmed is False:
                return PhysicalStatus.STOP_UNCONFIRMED.value
        if value is True:
            return PhysicalStatus.STOPPED.value
        if value is False:
            return PhysicalStatus.STOP_UNCONFIRMED.value
        return PhysicalStatus.STOP_UNCONFIRMED.value

    @staticmethod
    def _physical_from_value(value: Any) -> str:
        if isinstance(value, Mapping):
            status = value.get("physical_status")
            if isinstance(status, str) and status in {item.value for item in PhysicalStatus}:
                return status
        return PhysicalStatus.UNKNOWN.value

    @staticmethod
    def _error_text(error: BaseException | None) -> str | None:
        if error is None:
            return None
        return str(error)[:_MAX_ERROR_DETAIL]

    @staticmethod
    def _normalize_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("parameters must be a mapping")
        normalized = SQLiteJobStore.normalize_json(value)
        if not isinstance(normalized, dict):
            raise TypeError("parameters must normalize to an object")
        return normalized

    @staticmethod
    def _key(caller_id: str, session_id: str, request_id: str) -> JobKey:
        for name, value in (
            ("caller_id", caller_id),
            ("session_id", session_id),
            ("request_id", request_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        return JobKey(caller_id, session_id, request_id)

    @staticmethod
    def _validate_timeout(value: float | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("timeout_s must be finite and non-negative")

    def _require_open_locked(self) -> None:
        if self._closed:
            raise JobClosed("job registry is closed")

    def _begin_locked(self) -> None:
        self._store.begin()

    def _commit_locked(self) -> None:
        self._store.commit()

    def _rollback_locked(self) -> None:
        self._store.rollback()


__all__ = [
    "DispatchStatus",
    "InferenceStatus",
    "JobCancelledError",
    "JobCapacityExceeded",
    "JobClosed",
    "JobConflict",
    "JobError",
    "JobEvent",
    "JobExecutionError",
    "JobHandle",
    "JobKey",
    "JobNotFound",
    "JobOwnershipError",
    "JobRecord",
    "JobRegistry",
    "JobStatus",
    "JobTimeout",
    "JobUnknownError",
    "PhysicalStatus",
]
