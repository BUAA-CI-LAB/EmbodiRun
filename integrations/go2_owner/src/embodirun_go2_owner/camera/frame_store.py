"""Thread-safe latest RGB frame store."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional  # noqa: UP035

from .store_validation import validate_source_frame_number, validate_timestamp
from .types import FrameSnapshot, VideoProfile


class FrameStore:
    """One immutable RGB slot plus capture-health metadata."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        device: Optional[str] = None,  # noqa: UP045
    ) -> None:
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._frame: Optional[FrameSnapshot] = None  # noqa: UP045
        self._sequence = 0
        self._capture_running = False
        self._last_error: Optional[str] = None  # noqa: UP045
        self._started_at = self._clock()
        self._device = device
        self._profile: Optional[VideoProfile] = None  # noqa: UP045

    def set_capture_running(self, running: bool) -> None:
        with self._lock:
            self._capture_running = running

    def set_error(self, error: Optional[str]) -> None:  # noqa: UP045
        with self._lock:
            self._last_error = error

    def set_source_profile(self, device: str, profile: VideoProfile) -> None:
        with self._lock:
            self._device = device
            self._profile = profile

    def update(
        self,
        jpeg: bytes,
        sequence: Optional[int] = None,  # noqa: UP045
        captured_at_unix: Optional[float] = None,  # noqa: UP045
        captured_at_monotonic: Optional[float] = None,  # noqa: UP045
        source_timestamp_ms: Optional[float] = None,  # noqa: UP045
        source_frame_number: Optional[int] = None,  # noqa: UP045
    ) -> FrameSnapshot:
        if not jpeg:
            raise ValueError("jpeg must not be empty")
        for name, value in (
            ("captured_at_unix", captured_at_unix),
            ("captured_at_monotonic", captured_at_monotonic),
            ("source_timestamp_ms", source_timestamp_ms),
        ):
            validate_timestamp(name, value)
        validate_source_frame_number(source_frame_number)

        with self._lock:
            next_sequence = self._sequence + 1 if sequence is None else sequence
            if isinstance(next_sequence, bool) or not isinstance(next_sequence, int) or next_sequence <= self._sequence:
                raise ValueError("frame sequence must increase")
            snapshot = FrameSnapshot(
                jpeg=bytes(jpeg),
                sequence=next_sequence,
                captured_at_unix=(self._wall_clock() if captured_at_unix is None else float(captured_at_unix)),
                captured_at_monotonic=(
                    self._clock() if captured_at_monotonic is None else float(captured_at_monotonic)
                ),
                source_timestamp_ms=(None if source_timestamp_ms is None else float(source_timestamp_ms)),
                source_frame_number=source_frame_number,
            )
            self._sequence = next_sequence
            self._frame = snapshot
            self._last_error = None
            return snapshot

    def latest(self) -> Optional[FrameSnapshot]:  # noqa: UP045
        with self._lock:
            return self._frame

    def frame_age(self, frame: FrameSnapshot) -> float:
        return max(0.0, self._clock() - frame.captured_at_monotonic)

    def health(self, max_frame_age: float) -> Dict[str, object]:  # noqa: UP006
        with self._lock:
            frame = self._frame
            running = self._capture_running
            error = self._last_error
            uptime = max(0.0, self._clock() - self._started_at)
            device = self._device
            profile = self._profile

        age = None if frame is None else self.frame_age(frame)
        fresh = frame is not None and age is not None and age <= max_frame_age
        if running and fresh:
            status = "ok"
        elif running and frame is None and error is None:
            status = "starting"
        else:
            status = "degraded"
        return {
            "status": status,
            "capture_running": running,
            "frame_available": frame is not None,
            "frame_fresh": fresh,
            "frame_sequence": 0 if frame is None else frame.sequence,
            "frame_age_s": None if age is None else round(age, 3),
            "max_frame_age_s": max_frame_age,
            "last_error": error,
            "uptime_s": round(uptime, 3),
            "rgb_device": device,
            "rgb_profile": None if profile is None else profile.as_dict(),
            "source_timestamp_ms": None if frame is None else frame.source_timestamp_ms,
            "source_frame_number": None if frame is None else frame.source_frame_number,
        }


__all__ = ["FrameStore"]
