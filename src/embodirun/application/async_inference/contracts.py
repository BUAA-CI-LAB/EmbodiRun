"""Wire contract for asynchronous-inference hints.

EmbodiRun and EmbodiInfer speak a versioned HTTP/WirelessComm API, so the async
algorithms cannot share Python objects across the boundary — they share a
*payload*.  This module owns that payload: one optional ``async`` block inside
the existing step ``metadata``, carrying everything the server needs to
reproduce the execution side's timing decisions.

Keeping it in ``metadata`` rather than widening the step schema is deliberate:
the step schema is frozen and every existing client keeps working unchanged,
because a request without an ``async`` block is exactly the synchronous request
it always was.  Both algorithms are opt-in per request.

Schema (``embodiinfer.async.v1``)::

    {
      "schema": "embodiinfer.async.v1",
      "rtc": {
        "prev_chunk_left_over": [[...], ...],
        "inference_delay": 3,
        "execution_horizon": 8,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 10.0,
        "hard_prefix": false
      },
      "vlash": {
        "state_fields": ["joint_0", "joint_1"],
        "pending_actions": [[...], ...],
        "delay": 4,
        "action_space": "absolute"
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
from .vlash import ACTION_SPACE_SEMANTICS

__all__ = [
    "ASYNC_SCHEMA",
    "RTCAsyncRequest",
    "VlashAsyncRequest",
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

    @property
    def enabled(self) -> bool:
        """Whether a previous chunk is present and worth conditioning on."""

        return self.prev_chunk_left_over is not None and bool(self.prev_chunk_left_over)

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable ``rtc`` block."""

        return {
            "prev_chunk_left_over": (
                None if self.prev_chunk_left_over is None else _grid_payload(self.prev_chunk_left_over)
            ),
            "inference_delay": self.inference_delay,
            "execution_horizon": self.execution_horizon,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "max_guidance_weight": self.max_guidance_weight,
            "hard_prefix": self.hard_prefix,
        }


@dataclass(frozen=True, slots=True)
class VlashAsyncRequest:
    """One validated VLASH hint for a single step request.

    A named field may be a **scalar or a numeric vector**; its scalars are
    concatenated in field order to form the action-vector columns. This mirrors
    the inference server's ``VlashStatePlan`` and is the normal deployment case:
    LIBERO and π0.5 carry proprioception in one ``observation.state`` entry
    holding the whole joint vector, not as named scalars. Field widths are read
    from the state itself on the server, so a request only has to be internally
    consistent — every pending action vector the same width — and the server
    rejects a state whose total width disagrees rather than reinterpreting the
    columns.

    Args:
        state_fields: Observation-state keys to read and overwrite, in action
            column order. Each contributes as many columns as it holds scalars.
        pending_actions: ``[T, D]`` committed actions, in execution order, where
            ``D`` is the total scalar width of the named fields.
        delay: ``Δ`` in control steps.
        action_space: ``absolute`` or ``delta``.
        action_to_state_scale: Optional per-action-column tracking gain: how many
            state units one unit of that action actually produces. Omitted means
            1.0, which is only correct when the action already is the state
            increment. LIBERO's ``OSC_POSE`` controller tracks at ~0.005 for
            position and ~0.047 for rotation, so a unit-scale roll overstates the
            position change by ~200x.
    """

    state_fields: tuple[str, ...]
    pending_actions: tuple[tuple[float, ...], ...]
    delay: int
    action_space: str = "absolute"
    action_to_state: tuple[int, ...] | None = None
    action_to_state_scale: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not _is_sequence(self.state_fields):
            raise TypeError("async.vlash.state_fields must be a non-empty list of strings")
        fields = tuple(self.state_fields)
        if not fields:
            raise ValueError("async.vlash.state_fields must be a non-empty list of strings")
        if any(not isinstance(item, str) or not item for item in fields):
            raise ValueError("async.vlash.state_fields entries must be non-empty strings")
        if len(set(fields)) != len(fields):
            raise ValueError("async.vlash.state_fields must be unique")
        object.__setattr__(self, "state_fields", fields)

        pending = _numeric_grid(self.pending_actions, "async.vlash.pending_actions")
        if pending:
            widths = {len(row) for row in pending}
            if 0 in widths:
                raise ValueError("async.vlash.pending_actions rows must not be empty")
            if len(widths) != 1:
                raise ValueError("async.vlash.pending_actions rows must all have the same width")
        object.__setattr__(self, "pending_actions", pending)

        delay = _non_negative_int(self.delay, "async.vlash.delay")
        if delay > len(pending):
            raise ValueError(f"async.vlash.delay {delay} exceeds the {len(pending)} pending action(s) supplied")
        object.__setattr__(self, "delay", delay)

        if not isinstance(self.action_space, str) or self.action_space not in ACTION_SPACE_SEMANTICS:
            raise ValueError(
                "async.vlash.action_space must be one of "
                f"{', '.join(ACTION_SPACE_SEMANTICS)}, got {self.action_space!r}"
            )

        if self.action_to_state is not None:
            targets = tuple(int(value) for value in self.action_to_state)
            width = len(pending[0]) if pending else 0
            if not targets:
                raise ValueError("async.vlash.action_to_state must not be empty when provided")
            if len(targets) != width:
                raise ValueError(
                    f"async.vlash.action_to_state has {len(targets)} entries but pending_actions rows are {width} wide"
                )
            if any(value < -1 for value in targets):
                raise ValueError("async.vlash.action_to_state entries must be >= -1")
            mapped = [value for value in targets if value >= 0]
            if len(set(mapped)) != len(mapped):
                raise ValueError("async.vlash.action_to_state must not map two action columns onto one state column")
            object.__setattr__(self, "action_to_state", targets)

        if self.action_to_state_scale is not None:
            scales = tuple(float(value) for value in self.action_to_state_scale)
            width = len(pending[0]) if pending else 0
            if len(scales) != width:
                raise ValueError(
                    f"async.vlash.action_to_state_scale has {len(scales)} entries but "
                    f"pending_actions rows are {width} wide"
                )
            if any(not math.isfinite(value) for value in scales):
                raise ValueError("async.vlash.action_to_state_scale entries must be finite")
            if self.action_to_state is None:
                raise ValueError("async.vlash.action_to_state_scale requires action_to_state")
            object.__setattr__(self, "action_to_state_scale", scales)

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable ``vlash`` block."""

        payload: dict[str, Any] = {
            "state_fields": list(self.state_fields),
            "pending_actions": _grid_payload(self.pending_actions),
            "delay": self.delay,
            "action_space": self.action_space,
        }
        if self.action_to_state is not None:
            payload["action_to_state"] = list(self.action_to_state)
        if self.action_to_state_scale is not None:
            payload["action_to_state_scale"] = list(self.action_to_state_scale)
        return payload


def build_async_metadata(
    *,
    rtc: RTCAsyncRequest | None = None,
    vlash: VlashAsyncRequest | None = None,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build step metadata carrying the optional ``async`` block.

    Args:
        rtc: RTC hint, or ``None`` when RTC is disabled for this request.
        vlash: VLASH hint, or ``None`` when VLASH is disabled.
        base: Existing observation metadata to extend.  It is copied, never
            mutated.

    Returns:
        A metadata mapping.  When both algorithms are ``None`` the result equals
        ``base`` (an empty mapping by default) with no ``async`` key, so the
        request is byte-identical to the synchronous one.
    """

    metadata: dict[str, Any] = dict(base) if base is not None else {}

    block: dict[str, Any] = {"schema": ASYNC_SCHEMA}
    if rtc is not None:
        block["rtc"] = rtc.to_payload()
    if vlash is not None:
        block["vlash"] = vlash.to_payload()

    if len(block) > 1:
        metadata["async"] = block
    return metadata
