"""Exact, watchdog-bounded reads from raw camera pipes."""

from __future__ import annotations

import os
import select
import time
from typing import List  # noqa: UP035

from .types import CameraStreamError


def read_exact_with_timeout(stream: object, size: int, timeout: float) -> bytes:
    """Read exactly ``size`` bytes or fail before the frame watchdog expires."""

    if size <= 0 or timeout <= 0:
        raise ValueError("size and timeout must be positive")
    try:
        file_descriptor = stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError) as error:
        raise CameraStreamError("camera pipe has no usable file descriptor") from error

    deadline = time.monotonic() + timeout
    chunks: List[bytes] = []  # noqa: UP006
    remaining = size
    while remaining:
        wait_time = deadline - time.monotonic()
        if wait_time <= 0:
            raise CameraStreamError(f"camera frame watchdog expired after {timeout:.3f}s")
        try:
            readable, _, _ = select.select([file_descriptor], [], [], wait_time)
        except (OSError, ValueError) as error:
            raise CameraStreamError(f"camera pipe polling failed: {error}") from error
        if not readable:
            raise CameraStreamError(f"camera frame watchdog expired after {timeout:.3f}s")
        try:
            chunk = os.read(file_descriptor, remaining)
        except OSError as error:
            raise CameraStreamError(f"camera pipe read failed: {error}") from error
        if not chunk:
            raise CameraStreamError("camera pipe reached EOF mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = ["read_exact_with_timeout"]
