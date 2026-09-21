"""SQLite persistence and JSON serialization for :mod:`control.jobs`.

The store intentionally knows nothing about execution callbacks or device
authority.  Its composite primary key and explicit transactions provide the
atomic request-acceptance boundary used by ``JobRegistry``.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import TaskResult


class SQLiteJobStore:
    """Small store with one live local owner per persistent database.

    The sidecar flock separates an active registry from crash recovery.  A
    second live process is rejected, while a lock released by a crashed
    process permits the next registry to mark unfinished rows unknown.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = (
            ":memory:" if str(database_path) == ":memory:" else str(Path(database_path).expanduser().resolve())
        )
        self._lock_fd: int | None = None
        lock_fd: int | None = None
        connection: sqlite3.Connection | None = None
        try:
            if self.database_path != ":memory:":
                lock_path = Path(f"{self.database_path}.lock")
                lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        f"job database is already owned by another live registry: {self.database_path}"
                    ) from error
                self._lock_fd = lock_fd
                lock_fd = None
            connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            if self.database_path != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    caller_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    run_id TEXT NOT NULL UNIQUE,
                    parameters_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dispatch_status TEXT NOT NULL,
                    inference_status TEXT NOT NULL,
                    physical_status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    events_json TEXT NOT NULL,
                    events_truncated INTEGER NOT NULL DEFAULT 0,
                    created_at_s REAL NOT NULL,
                    updated_at_s REAL NOT NULL,
                    started_at_s REAL,
                    finished_at_s REAL,
                    PRIMARY KEY (caller_id, session_id, request_id)
                )
                """
            )
            self.connection = connection
        except BaseException:
            if connection is not None:
                connection.close()
            if lock_fd is not None:
                os.close(lock_fd)
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None
            raise

    def begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def commit(self) -> None:
        self.connection.execute("COMMIT")

    def rollback(self) -> None:
        self.connection.execute("ROLLBACK")

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def fetch(self, key: Any) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM jobs WHERE caller_id=? AND session_id=? AND request_id=?",
            (key.caller_id, key.session_id, key.request_id),
        ).fetchone()

    def owners_for_request(self, request_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT caller_id, session_id FROM jobs WHERE request_id=?",
            (request_id,),
        ).fetchall()

    def rows_with_status(self, statuses: Sequence[str]) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in statuses)
        return self.connection.execute(
            f"SELECT * FROM jobs WHERE status IN ({placeholders})", tuple(statuses)
        ).fetchall()

    def rows_not_status(self, statuses: Sequence[str]) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in statuses)
        return self.connection.execute(
            f"SELECT * FROM jobs WHERE status NOT IN ({placeholders})", tuple(statuses)
        ).fetchall()

    def insert(
        self,
        key: Any,
        *,
        run_id: str,
        parameters_json: str,
        status: str,
        dispatch_status: str,
        inference_status: str,
        physical_status: str,
        events_json: str,
        now: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO jobs (
                caller_id, session_id, request_id, run_id,
                parameters_json, status, dispatch_status,
                inference_status, physical_status,
                cancel_requested, result_json, error,
                events_json, events_truncated,
                created_at_s, updated_at_s, started_at_s, finished_at_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, 0, ?, ?, NULL, NULL)
            """,
            (
                key.caller_id,
                key.session_id,
                key.request_id,
                run_id,
                parameters_json,
                status,
                dispatch_status,
                inference_status,
                physical_status,
                events_json,
                now,
                now,
            ),
        )

    def update(self, key: Any, **fields: Any) -> None:
        allowed = {
            "status",
            "dispatch_status",
            "inference_status",
            "physical_status",
            "cancel_requested",
            "result_json",
            "error",
            "events_json",
            "events_truncated",
            "updated_at_s",
            "started_at_s",
            "finished_at_s",
        }
        if not fields or any(name not in allowed for name in fields):
            raise ValueError("invalid or empty job store update")
        assignments = ", ".join(f"{name}=?" for name in fields)
        values = list(fields.values())
        values.extend((key.caller_id, key.session_id, key.request_id))
        self.connection.execute(
            f"UPDATE jobs SET {assignments} WHERE caller_id=? AND session_id=? AND request_id=?",
            values,
        )

    @staticmethod
    def append_event(
        row: Mapping[str, Any],
        event: Mapping[str, Any],
        *,
        max_events: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        events = json.loads(row["events_json"])
        truncated = bool(row["events_truncated"])
        if len(events) >= max_events:
            return events, True
        events.append(dict(event))
        return events, truncated

    @staticmethod
    def normalize_json(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("JSON numbers must be finite")
            return value
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                normalized[key] = SQLiteJobStore.normalize_json(item)
            return dict(sorted(normalized.items()))
        if isinstance(value, (list, tuple)):
            return [SQLiteJobStore.normalize_json(item) for item in value]
        raise TypeError(f"unsupported JSON parameter type: {type(value).__name__}")

    @staticmethod
    def canonical_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def encode_result(value: Any) -> str:
        if isinstance(value, TaskResult):
            value = value.to_payload()
        try:
            normalized = SQLiteJobStore.normalize_json(value)
        except (TypeError, ValueError):
            normalized = {
                "opaque_type": type(value).__name__,
                "opaque_repr": repr(value),
            }
        return SQLiteJobStore.canonical_json(normalized)

    @staticmethod
    def decode_result(value: str) -> Any:
        decoded = json.loads(value)
        if isinstance(decoded, Mapping):
            try:
                return TaskResult.from_payload(decoded)
            except Exception:
                pass
        return decoded

    @staticmethod
    def decode_json(value: str) -> Any:
        """Decode a stored JSON object without guessing a domain type.

        Parameters and event rows are ordinary JSON.  They must not pass
        through :meth:`decode_result`, whose legacy compatibility behavior
        recognizes a ``TaskResult`` payload.
        """

        return json.loads(value)

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None


__all__ = ["SQLiteJobStore"]
