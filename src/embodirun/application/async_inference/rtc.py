"""Real-Time Chunking (RTC) execution-side bookkeeping.

This module is the execution-side port of LeRobot's
``src/lerobot/policies/rtc/action_queue.py`` and ``latency_tracker.py``.  It is
pure Python plus numpy: EmbodiRun never imports the inference engine, so the
numeric pieces the server also computes (the prefix-weight table) are duplicated
here and pinned against the shared golden vectors.

The central object is :class:`ActionQueue`, a thread-safe queue holding *two*
parallel sequences:

``original_queue``
    The policy's raw (model-space) chunk.  Only the leftover prefix
    ``original_queue[last_index:]`` is sent back to the server as
    ``prev_chunk_left_over``.
``queue``
    The post-processed (robot-space) chunk that is actually executed.

Both are numpy arrays of shape ``[T, D]``.  Clones are returned so a caller
cannot mutate queue state through a returned action.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

__all__ = [
    "PREFIX_ATTENTION_SCHEDULES",
    "ActionQueue",
    "LatencyTracker",
    "prefix_weights",
]

PREFIX_ATTENTION_SCHEDULES = ("zeros", "ones", "linear", "exp")


def _as_action_array(value: object, name: str) -> np.ndarray:
    """Coerce one action chunk into a numeric numpy array, preserving dtype."""

    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of actions, got shape {array.shape}")
    if array.size and array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str = "linear",
) -> np.ndarray:
    """Return the ``[total]`` RTC prefix-firmness weights for one request.

    ``start`` is the inference delay (pushed down to ``end`` when larger, since
    ``end`` takes precedence), ``end`` is the execution horizon (exclusive), and
    ``total`` is the action horizon ``H``.  This is a numpy port of the same
    piecewise leading-ones / ramp / trailing-zeros construction used by the
    inference engine, so both sides of the boundary assert the identical golden
    table and a drift on either side fails loudly.

    Args:
        start: Inference delay in control steps (``Δ``).
        end: Execution horizon, exclusive.
        total: Action horizon ``H``.
        schedule: One of :data:`PREFIX_ATTENTION_SCHEDULES`.

    Returns:
        A ``float64`` array of length ``total``.

    Raises:
        ValueError: for a non-positive ``total``, negative ``start``/``end``, or
            an unknown schedule.
    """

    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if schedule not in PREFIX_ATTENTION_SCHEDULES:
        raise ValueError(f"schedule must be one of {', '.join(PREFIX_ATTENTION_SCHEDULES)}, got {schedule!r}")

    start = min(start, end)

    if schedule == "zeros":
        weights = np.zeros(total, dtype=np.float64)
        weights[:start] = 1.0
        return weights
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float64)
        weights[end:] = 0.0
        return weights

    ramp = _linear_ramp(start, end, total)
    if schedule == "exp":
        ramp = ramp * np.expm1(ramp) / (math.e - 1.0)
    return _add_leading_ones(_add_trailing_zeros(ramp, total, end), start, total)


def _linear_ramp(start: int, end: int, total: int) -> np.ndarray:
    """The interior ``linspace(1, 0)`` segment of the linear/exp schedules."""

    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        return np.array([], dtype=np.float64)
    return np.linspace(1.0, 0.0, linspace_steps + 2)[1:-1]


def _add_trailing_zeros(weights: np.ndarray, total: int, end: int) -> np.ndarray:
    zeros_len = total - end
    if zeros_len <= 0:
        return weights
    return np.concatenate([weights, np.zeros(zeros_len, dtype=np.float64)])


def _add_leading_ones(weights: np.ndarray, start: int, total: int) -> np.ndarray:
    ones_len = min(start, total)
    if ones_len <= 0:
        return weights
    return np.concatenate([np.ones(ones_len, dtype=np.float64), weights])


class ActionQueue:
    """Thread-safe double queue of raw and post-processed action chunks.

    Args:
        rtc_enabled: When true, :meth:`merge` *replaces* the queue (dropping the
            actions consumed during inference); when false it *appends*,
            preserving continuity without replacement.
    """

    def __init__(self, *, rtc_enabled: bool = True) -> None:
        self.rtc_enabled = bool(rtc_enabled)
        self.queue: np.ndarray | None = None
        self.original_queue: np.ndarray | None = None
        self._task_queue: list[str | None] | None = None
        self.lock = threading.Lock()
        self.last_index = 0

    # -- consumption -----------------------------------------------------
    def get(self) -> np.ndarray | None:
        """Return the next processed action (a clone), or ``None`` if empty."""

        queued = self.get_with_task()
        return None if queued is None else queued[0]

    def get_with_task(self) -> tuple[np.ndarray, str | None] | None:
        """Return the next processed action together with its chunk's task."""

        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            self._assert_lockstep()
            action = np.array(self.queue[self.last_index], copy=True)
            task = None if self._task_queue is None else self._task_queue[self.last_index]
            self.last_index += 1
            return action, task

    def qsize(self) -> int:
        """Number of unconsumed processed actions."""

        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Whether no processed actions remain."""

        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Index of the next processed action to consume."""

        with self.lock:
            return self.last_index

    # -- leftovers -------------------------------------------------------
    def get_left_over(self) -> np.ndarray | None:
        """Unconsumed *original* (model-space) actions, the RTC prefix."""

        with self.lock:
            self._assert_lockstep()
            if self.original_queue is None:
                return None
            return np.array(self.original_queue[self.last_index :], copy=True)

    def get_processed_left_over(self) -> np.ndarray | None:
        """Unconsumed *processed* (robot-space) actions."""

        with self.lock:
            self._assert_lockstep()
            if self.queue is None:
                return None
            return np.array(self.queue[self.last_index :], copy=True)

    def clear(self) -> None:
        """Drop all queued actions and reset consumption."""

        with self.lock:
            self.queue = None
            self.original_queue = None
            self._task_queue = None
            self.last_index = 0

    # -- merging ---------------------------------------------------------
    def merge(
        self,
        original_actions: object,
        processed_actions: object,
        real_delay: int,
        action_index_before_inference: int | None = None,
        *,
        task: str | None = None,
    ) -> int:
        """Merge a fresh chunk into the queue and return the resolved delay.

        RTC enabled replaces the queue, dropping the first ``resolved_delay``
        actions; RTC disabled appends, dropping already-consumed entries so the
        new chunk continues where the old one left off.

        Args:
            original_actions: Raw model-space chunk ``[T, D_model]``.
            processed_actions: Robot-space chunk ``[T, D_robot]``.
            real_delay: Delay in control steps, typically latency / period.
            action_index_before_inference: ``last_index`` captured when the
                request was sent; enables the never-over-discard rule.
            task: Instruction that produced the chunk; labels every new action.

        Returns:
            The resolved delay actually applied.
        """

        original = _as_action_array(original_actions, "original_actions")
        processed = _as_action_array(processed_actions, "processed_actions")
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
            if self.rtc_enabled:
                return self._replace_actions_queue(original, processed, delay, task)
            self._append_actions_queue(original, processed, task)
            return delay

    def _replace_actions_queue(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        task: str | None,
    ) -> int:
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = np.array(original_actions[clamped_delay:], copy=True)
        self.queue = np.array(processed_actions[clamped_delay:], copy=True)
        self._task_queue = [task] * len(self.queue)
        self.last_index = 0
        return clamped_delay

    def _append_actions_queue(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        task: str | None,
    ) -> None:
        if self.queue is None:
            self.original_queue = np.array(original_actions, copy=True)
            self.queue = np.array(processed_actions, copy=True)
            self._task_queue = [task] * len(self.queue)
            self.last_index = 0
            return

        self._assert_lockstep()
        existing_tasks = self._task_queue if self._task_queue is not None else [None] * len(self.queue)
        original_queue = self.original_queue if self.original_queue is not None else np.empty((0,))
        self.original_queue = np.concatenate([original_queue, np.array(original_actions, copy=True)])[self.last_index :]
        self.queue = np.concatenate([self.queue, np.array(processed_actions, copy=True)])[self.last_index :]
        self._task_queue = existing_tasks[self.last_index :] + [task] * len(processed_actions)
        self.last_index = 0

    def _check_and_resolve_delays(
        self,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> int:
        """Resolve the delay to apply.

        When the caller captured ``action_index_before_inference``, the delay can
        never exceed the number of actions actually consumed.  Discarding more
        would splice the queue ahead of the physical pose.
        """

        if isinstance(real_delay, bool) or not isinstance(real_delay, int):
            raise TypeError("real_delay must be an integer")
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            if isinstance(action_index_before_inference, bool) or not isinstance(action_index_before_inference, int):
                raise TypeError("action_index_before_inference must be an integer")
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                # Never discard more than was actually consumed.
                return min(real_delay, indexes_diff)

        return effective_delay

    def _assert_lockstep(self) -> None:
        """Raise if the task-label list ever disagrees with the action length."""

        action_count = 0 if self.queue is None else len(self.queue)
        if self._task_queue is not None and len(self._task_queue) != action_count:
            raise RuntimeError(
                "ActionQueue task labels out of sync with actions "
                f"({len(self._task_queue)} labels for {action_count} actions) — "
                "a queue mutation broke the action/task lockstep invariant"
            )


class LatencyTracker:
    """Inference-latency window with an injectable clock.

    Args:
        window: Maximum number of samples kept.  ``None`` keeps all.
        monotonic: Injectable monotonic clock, for deterministic tests.
    """

    def __init__(
        self,
        window: int | None = 100,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("window must be a positive integer or None")
        self._monotonic = monotonic
        self._values: deque[float] = deque(maxlen=window)
        self._started_at: float | None = None
        self.max_latency = 0.0

    def reset(self) -> None:
        """Clear all recorded latencies."""

        self._values.clear()
        self._started_at = None
        self.max_latency = 0.0

    def start(self) -> None:
        """Mark the start of one inference call."""

        self._started_at = self._monotonic()

    def stop(self) -> float:
        """Mark the end of one inference call and record its latency."""

        if self._started_at is None:
            raise RuntimeError("LatencyTracker.stop() called before start()")
        latency = max(0.0, self._monotonic() - self._started_at)
        self._started_at = None
        self.record(latency)
        return latency

    def record(self, latency: float) -> None:
        """Add one latency sample in seconds.  Negative values are ignored."""

        value = float(latency)
        if not math.isfinite(value) or value < 0:
            return
        self._values.append(value)
        self.max_latency = max(self.max_latency, value)

    def __len__(self) -> int:
        return len(self._values)

    def average(self) -> float:
        """Mean latency over the window, or ``0.0`` when empty."""

        if not self._values:
            return 0.0
        return float(sum(self._values) / len(self._values))

    def median(self) -> float:
        """Median latency over the window, or ``0.0`` when empty."""

        if not self._values:
            return 0.0
        values = np.asarray(self._values, dtype=np.float64)
        return float(np.median(values))

    def max(self) -> float:
        """Maximum latency recorded so far."""

        return self.max_latency

    def delay_in_steps(self, latency_s: float, period_s: float) -> int:
        """Convert a latency in seconds into a non-negative integer step count."""

        period = float(period_s)
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period_s must be a finite positive number")
        latency = float(latency_s)
        if not math.isfinite(latency) or latency < 0:
            return 0
        # A tiny epsilon keeps an exact ``n * period`` product from flooring to
        # ``n - 1`` because of binary floating-point representation.
        return max(0, int(math.floor(latency / period + 1e-9)))

    def latest_delay(self, period_s: float) -> int:
        """Delay implied by the most recent sample, or ``0`` when empty."""

        if not self._values:
            return 0
        return self.delay_in_steps(self._values[-1], period_s)
