"""Validation shared by the camera snapshot stores."""

from __future__ import annotations

import math
from typing import Optional


def validate_timestamp(name: str, value: Optional[float]) -> None:  # noqa: UP045
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def validate_source_frame_number(value: Optional[int]) -> None:  # noqa: UP045
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("source_frame_number must be a non-negative integer")


__all__ = ["validate_source_frame_number", "validate_timestamp"]
