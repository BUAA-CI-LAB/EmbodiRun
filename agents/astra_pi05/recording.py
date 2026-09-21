"""Small JSONL recorder for upper-layer cooperative session evidence.

The recorder stores detached observations, decisions, control responses, and
errors.  It does not open cameras or inspect a robot.  A caller that wants
media files may first obtain them through ``ControlClient.media`` and pass the
returned bytes to :meth:`SessionRecorder.save_media`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    return str(value)


def _safe_name(value: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("media name must be a non-empty simple name")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("media name must not contain path traversal")
    return value


class SessionRecorder:
    """Append session events and a final detached status summary."""

    def __init__(self, output: str | Path) -> None:
        self.output = Path(output).resolve()
        if self.output.exists() and not self.output.is_dir():
            raise FileExistsError(f"recording output is not a directory: {self.output}")
        self.output.mkdir(parents=True, exist_ok=False)
        self.media_dir = self.output / "media"
        self.media_dir.mkdir()
        self._events_path = self.output / "events.jsonl"
        self._events = self._events_path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._closed = False
        self._finalized = False

    @property
    def events_path(self) -> Path:
        return self._events_path

    def record(self, kind: str, **payload: Any) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("event kind must be a non-empty string")
        event = {
            "kind": kind,
            "wall_time_s": time.time(),
            "monotonic_s": time.monotonic(),
            **{str(key): _json_value(value) for key, value in payload.items()},
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder is closed")
            self._events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self._events.flush()
        return event

    def save_media(
        self,
        name: str,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str | None = None,
        observation_id: str | None = None,
    ) -> Path:
        """Persist bytes returned by a public media route.

        This helper accepts data supplied by the caller; it never captures a
        frame itself.  The optional ``media_type`` is recorded as provenance.
        """

        safe_name = _safe_name(name)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("media data must be bytes-like")
        value = bytes(data)
        if not value:
            raise ValueError("media data must not be empty")
        path = self.media_dir / safe_name
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder is closed")
            if path.exists():
                raise FileExistsError(f"media file already exists: {path.name}")
            path.write_bytes(value)
        self.record(
            "media_saved",
            name=safe_name,
            path=str(path.relative_to(self.output)),
            bytes=len(value),
            media_type=media_type,
            observation_id=observation_id,
        )
        return path

    def finalize(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(summary, Mapping):
            raise TypeError("summary must be a mapping")
        result = _json_value(dict(summary))
        if not isinstance(result, dict):  # pragma: no cover - guarded above
            raise TypeError("summary must be a mapping")
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder is closed")
            if self._finalized:
                raise RuntimeError("recording is already finalized")
            self._events.flush()
            self._finalized = True
            status_path = self.output / "status.json"
            status_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._events.flush()
            self._events.close()
            self._closed = True

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


__all__ = ["SessionRecorder"]
