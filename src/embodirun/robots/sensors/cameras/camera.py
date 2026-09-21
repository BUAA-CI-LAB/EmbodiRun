"""Model-independent camera values used by robot runtimes.

``CameraFrame`` deliberately keeps the original three positional fields.  The
optional timing and profile fields describe the source without changing the
encoded bytes.  ``captured_timestamp_ns`` is the time at which the deployment
process read the source frame; it is not an exposure or device-clock claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol


def _freeze(value: Any) -> Any:
    """Make nested metadata safe to share between observation consumers."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, complex, type(None))):
        return value
    raise TypeError(f"camera metadata value {type(value).__name__} is not immutable")


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One encoded frame identified in the robot observation schema.

    ``captured_timestamp_ns`` and ``received_timestamp_ns`` use the clock
    named by ``clock_domain``.  For V4L2 and RealSense implementations the
    captured value is taken immediately after host-side frame read and before
    JPEG encoding.  A legacy source can leave both values as ``None``; callers
    must then treat timing/freshness as unknown rather than fresh.
    """

    name: str
    mime_type: str
    data: bytes
    captured_timestamp_ns: int | None = None
    received_timestamp_ns: int | None = None
    clock_domain: str | None = None
    profile: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera frame name must not be empty")
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("camera frame must be JPEG or PNG")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("camera frame data must not be empty")
        for field_name in ("captured_timestamp_ns", "received_timestamp_ns"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"camera frame {field_name} must be a non-negative integer")
        if (
            self.captured_timestamp_ns is not None
            and self.received_timestamp_ns is not None
            and self.received_timestamp_ns < self.captured_timestamp_ns
        ):
            raise ValueError("camera frame receive time cannot precede capture time")
        if self.clock_domain is not None and (not isinstance(self.clock_domain, str) or not self.clock_domain.strip()):
            raise ValueError("camera frame clock_domain must be a non-empty string")
        if self.profile is not None:
            if not isinstance(self.profile, Mapping):
                raise ValueError("camera frame profile must be a mapping")
            if any(not isinstance(key, str) for key in self.profile):
                raise ValueError("camera frame profile keys must be strings")
            object.__setattr__(self, "profile", _freeze(self.profile))

    @property
    def capture_timestamp_ns(self) -> int | None:
        """Compatibility spelling for the host-side capture timestamp."""

        return self.captured_timestamp_ns

    @property
    def timestamp_ns(self) -> int | None:
        """Return the capture time when known, otherwise ``None``.

        This property intentionally does not fall back to receive time: a
        post-encoding timestamp cannot prove when the physical frame arrived.
        """

        return self.captured_timestamp_ns


@dataclass(frozen=True, slots=True)
class RawCameraFrame:
    """Owned, packed uint8 image bytes for local observation consumers.

    This separate type is not an encoded image accepted by inference clients.
    It skips application JPEG encoding; camera-driver decoding may still occur.
    It is kept for local capture and transport experiments that need a raw
    packed frame rather than an encoded :class:`CameraFrame`.
    """

    name: str
    width: int
    height: int
    data: bytes
    pixel_format: str = "bgr8"
    mime_type: str = "application/x-embodirun-raw-image"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera name must not be empty")
        if any(type(value) is not int or value <= 0 for value in (self.width, self.height)):
            raise ValueError("raw camera dimensions must be positive integers")
        if self.pixel_format not in {"bgr8", "rgb8"}:
            raise ValueError("raw camera pixel format must be bgr8 or rgb8")
        if self.mime_type != "application/x-embodirun-raw-image":
            raise ValueError("unexpected raw camera MIME type")
        if not isinstance(self.data, bytes) or len(self.data) != self.width * self.height * 3:
            raise ValueError("raw camera payload must contain exactly width*height*3 bytes")


class CameraSource(Protocol):
    """Capture and release a configured collection of cameras."""

    def capture(self) -> tuple[CameraFrame, ...]: ...

    def close(self) -> None: ...


class CameraSources:
    """Expose multiple camera implementations as one capture source."""

    def __init__(self, sources: tuple[CameraSource, ...]) -> None:
        if not sources:
            raise ValueError("at least one camera source is required")
        self._sources = sources

    def capture(self) -> tuple[CameraFrame, ...]:
        return tuple(frame for source in self._sources for frame in source.capture())

    def close(self) -> None:
        first_error: Exception | None = None
        for source in reversed(self._sources):
            try:
                source.close()
            except Exception as error:  # pragma: no cover - hardware cleanup failure
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


__all__ = ["CameraFrame", "CameraSource", "CameraSources"]
