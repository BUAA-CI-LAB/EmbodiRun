"""Wire contract for asynchronous-inference hints.

EmbodiRun and EmbodiInfer speak a versioned HTTP/WirelessComm API, so the async
algorithms cannot share Python objects across the boundary — they share a
*payload*.  This module owns that payload: one optional ``async`` block inside
the existing step ``metadata``, carrying everything the server needs to
reproduce the execution side's timing decisions.

Keeping it in ``metadata`` rather than widening the step schema is deliberate:
the step schema is frozen and every existing client keeps working unchanged,
because a request without an ``async`` block is exactly the synchronous request
it always was. RTC is opt-in per request.

Schema (``embodiinfer.async.v1``)::

    {
      "schema": "embodiinfer.async.v1",
      "rtc": {
        "prev_chunk_left_over": [[...], ...],
        "inference_delay": 3,
        "cache_start_index": 10,
        "execution_horizon": 8,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 10.0,
        "hard_prefix": false
      }
    }

Validation mirrors the server's ``embodiinfer.engine.async_inference.contracts``
rules so a malformed block is rejected on the execution side too, instead of
silently downgrading to synchronous behaviour.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rtc import PREFIX_ATTENTION_SCHEDULES

__all__ = [
    "ASYNC_SCHEMA",
    "RTCAsyncRequest",
    "build_async_metadata",
]

ASYNC_SCHEMA = "embodiinfer.async.v1"

# Bounds mirroring the server so a malformed or hostile request cannot make a
# peer allocate without limit.
MAX_GRID_ROWS = 4096
MAX_GRID_COLUMNS = 4096


def _is_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (Sequence, np.ndarray))


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _numeric_grid(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    """Validate a nested sequence of finite numbers into a nested tuple."""

    if not _is_sequence(value):
        raise TypeError(f"{name} must be a list of lists")
    rows_value = list(value)  # type: ignore[arg-type]
    if len(rows_value) > MAX_GRID_ROWS:
        raise ValueError(f"{name} has too many rows (limit {MAX_GRID_ROWS})")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for index, row in enumerate(rows_value):
        if not _is_sequence(row):
            raise TypeError(f"{name}[{index}] must be a list of numbers")
        columns = [_finite_float(item, f"{name}[{index}]") for item in row]
        if not columns:
            raise ValueError(f"{name}[{index}] must not be empty")
        if len(columns) > MAX_GRID_COLUMNS:
            raise ValueError(f"{name}[{index}] has too many columns (limit {MAX_GRID_COLUMNS})")
        if width is None:
            width = len(columns)
        elif len(columns) != width:
            raise ValueError(f"{name} rows must all have the same width")
        rows.append(tuple(columns))
    return tuple(rows)


def _grid_payload(grid: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [list(row) for row in grid]


@dataclass(frozen=True, slots=True)
class RTCAsyncRequest:
    """One validated RTC hint for a single step request.

    Args:
        prev_chunk_left_over: Unconsumed raw model-space actions from the
            previous chunk, or ``None`` on the first inference of an episode.
        inference_delay: Latency in control steps (``Δ``).
        cache_start_index: Index in the server-cached full model chunk at which
            the unexecuted prefix begins. This is the replan stride, not the
            inference delay. Omit it when an explicit prefix is supplied.
        execution_horizon: Number of leading steps eligible to be held.
        prefix_attention_schedule: Weight shape across the horizon.
        max_guidance_weight: Upper clamp on the guidance strength.
        hard_prefix: Use the fixed-prefix clamp instead of soft guidance.
    """

    prev_chunk_left_over: tuple[tuple[float, ...], ...] | None
    inference_delay: int
    execution_horizon: int
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0
    hard_prefix: bool = False
    model_space: bool = False
    cache_start_index: int | None = None

    def __post_init__(self) -> None:
        if self.prev_chunk_left_over is not None:
            grid = _numeric_grid(self.prev_chunk_left_over, "async.rtc.prev_chunk_left_over")
        else:
            grid = None
        object.__setattr__(self, "prev_chunk_left_over", grid)
        object.__setattr__(
            self,
            "inference_delay",
            _non_negative_int(self.inference_delay, "async.rtc.inference_delay"),
        )
        object.__setattr__(
            self,
            "execution_horizon",
            _non_negative_int(self.execution_horizon, "async.rtc.execution_horizon"),
        )
        if self.cache_start_index is not None:
            object.__setattr__(
                self,
                "cache_start_index",
                _non_negative_int(self.cache_start_index, "async.rtc.cache_start_index"),
            )
        if not isinstance(self.prefix_attention_schedule, str) or (
            self.prefix_attention_schedule not in PREFIX_ATTENTION_SCHEDULES
        ):
            raise ValueError(
                "async.rtc.prefix_attention_schedule must be one of "
                f"{', '.join(PREFIX_ATTENTION_SCHEDULES)}, got {self.prefix_attention_schedule!r}"
            )
        object.__setattr__(
            self,
            "max_guidance_weight",
            _positive_float(self.max_guidance_weight, "async.rtc.max_guidance_weight"),
        )
        if not isinstance(self.hard_prefix, bool):
            raise TypeError("async.rtc.hard_prefix must be a boolean")
        if not isinstance(self.model_space, bool):
            raise TypeError("async.rtc.model_space must be a boolean")
        if self.prev_chunk_left_over is not None and not self.model_space:
            raise ValueError(
                "async.rtc.prev_chunk_left_over requires model_space=true; "
                "otherwise omit the prefix and use the session cache"
            )

    @property
    def enabled(self) -> bool:
        """Whether a previous chunk is present and worth conditioning on."""

        return self.prev_chunk_left_over is not None and bool(self.prev_chunk_left_over)

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable ``rtc`` block."""

        payload = {
            "prev_chunk_left_over": (
                None if self.prev_chunk_left_over is None else _grid_payload(self.prev_chunk_left_over)
            ),
            "inference_delay": self.inference_delay,
            "execution_horizon": self.execution_horizon,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "max_guidance_weight": self.max_guidance_weight,
            "hard_prefix": self.hard_prefix,
        }
        if self.cache_start_index is not None:
            payload["cache_start_index"] = self.cache_start_index
        if self.prev_chunk_left_over is not None:
            payload["model_space"] = True
        return payload


def build_async_metadata(
    *,
    rtc: RTCAsyncRequest | None = None,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build step metadata carrying the optional ``async`` block.

    Args:
        rtc: RTC hint, or ``None`` when RTC is disabled for this request.
        base: Existing observation metadata to extend.  It is copied, never
            mutated.

    Returns:
        A metadata mapping. When RTC is ``None`` the result equals
        ``base`` (an empty mapping by default) with no ``async`` key, so the
        request is byte-identical to the synchronous one.
    """

    metadata: dict[str, Any] = dict(base) if base is not None else {}

    block: dict[str, Any] = {"schema": ASYNC_SCHEMA}
    if rtc is not None:
        block["rtc"] = rtc.to_payload()
    if len(block) > 1:
        metadata["async"] = block
    return metadata
