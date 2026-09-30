"""VLASH execution-side future-state-aware async inference.

VLASH (Tang et al., *VLASH: Real-Time VLAs via Future-State-Aware Asynchronous
Inference*, arXiv 2512.01031) closes the prediction/execution misalignment of
naive asynchronous inference by rolling the robot state forward under the actions
that are already committed::

    s_{t+Δ} = roll(s_t, a_{t : t+Δ-1})

This module is the execution-side port of ``vlash/vlash/run.py`` semantics.  It
is pure Python plus numpy, so it is shared verbatim with the inference engine's
own implementation and pinned by the common golden vectors in
``tests/fixtures/async_inference_golden.json``.

Two action-space semantics are supported:

``absolute``
    The action vector is the commanded target, so the state after ``Δ`` steps is
    ``pending[Δ - 1]``.  This is the LeRobot / π0.5 convention and what the
    upstream deployment loop hardcodes.
``delta``
    The action is an increment, so the state after ``Δ`` steps is
    ``s_t + Σ_{i<Δ} pending[i]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ACTION_SPACE_SEMANTICS",
    "VlashScheduler",
    "estimate_future_state",
    "quantize_actions",
    "roll_state_forward",
]

ACTION_SPACE_SEMANTICS = ("absolute", "delta")


def _semantics(value: str) -> str:
    if value not in ACTION_SPACE_SEMANTICS:
        raise ValueError(f"semantics must be one of {', '.join(ACTION_SPACE_SEMANTICS)}, got {value!r}")
    return value


def roll_state_forward(
    state: object,
    pending_actions: object,
    delay: int,
    semantics: str = "absolute",
) -> np.ndarray:
    """Return the robot state ``delay`` steps ahead, given committed actions.

    Args:
        state: ``[D]`` or ``[B, D]`` current robot state.
        pending_actions: ``[T, D]`` or ``[B, T, D]`` actions already issued and
            still to be executed, in execution order.
        delay: ``Δ``, the number of pending steps that elapse.  ``0`` returns the
            state unchanged; ``delay == T`` consumes the whole buffer.
        semantics: ``absolute`` or ``delta``.

    Returns:
        ``[D]`` or ``[B, D]`` rolled-forward state, matching ``state``'s rank.

    Raises:
        ValueError: if ``delay`` is not an integer, is negative, exceeds the
            pending actions, or the shapes disagree.
    """

    semantics = _semantics(semantics)
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise ValueError("delay must be an integer")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    state_array = np.asarray(state, dtype=np.float64)
    actions = np.asarray(pending_actions, dtype=np.float64)

    squeezed = state_array.ndim == 1
    state_2d = state_array[np.newaxis, :] if squeezed else state_array
    if state_2d.ndim != 2:
        raise ValueError(f"state must be 1D or 2D, got shape {state_array.shape}")

    if actions.ndim == 2:
        actions = actions[np.newaxis, :, :]
    if actions.ndim != 3:
        raise ValueError(f"pending_actions must be 2D or 3D, got shape {np.shape(pending_actions)}")

    horizon = actions.shape[1]
    if delay > horizon:
        raise ValueError(f"delay {delay} exceeds the {horizon} pending action(s) available")
    if actions.shape[2] != state_2d.shape[1]:
        raise ValueError(f"pending action width {actions.shape[2]} does not match state width {state_2d.shape[1]}")
    if actions.shape[0] != state_2d.shape[0]:
        raise ValueError(f"pending actions batch {actions.shape[0]} does not match state batch {state_2d.shape[0]}")

    if delay == 0:
        rolled = state_2d
    elif semantics == "absolute":
        rolled = actions[:, delay - 1, :]
    else:
        rolled = state_2d + actions[:, :delay, :].sum(axis=1)

    return rolled[0] if squeezed else rolled


def quantize_actions(
    actions: object,
    ratio: int,
    semantics: str = "delta",
) -> object:
    """Group ``ratio`` consecutive actions into one macro-action.

    ``delta`` actions are summed (``q`` micro-increments compose into one
    macro-increment); ``absolute`` actions keep the last of each group (the
    commanded target the group settles on).  A trailing group shorter than
    ``ratio`` is kept rather than dropped, so no action is silently lost.

    Args:
        actions: ``[T]`` or ``[T, D]`` action sequence.
        ratio: Number of micro-actions per macro-action.  ``ratio <= 1`` is a
            no-op and returns ``actions`` unchanged.
        semantics: ``absolute`` or ``delta``.

    Returns:
        The quantized array, or the original object when ``ratio <= 1``.
    """

    if isinstance(ratio, bool) or not isinstance(ratio, int):
        raise TypeError("ratio must be an integer")
    if ratio <= 1:
        return actions
    semantics = _semantics(semantics)

    array = np.asarray(actions, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"actions must be 1D or 2D, got shape {array.shape}")
    if len(array) == 0:
        return np.array(array, copy=True)

    groups = [array[index : index + ratio] for index in range(0, len(array), ratio)]
    if semantics == "absolute":
        return np.stack([group[-1] for group in groups])
    return np.stack([group.sum(axis=0) for group in groups])


@dataclass(frozen=True, slots=True)
class VlashScheduler:
    """Decide when to launch inference, switch chunks, and fetch observations.

    ``effective_overlap_steps`` scales the configured overlap by the action
    quantization ratio, because one quantized macro-action spans ``ratio``
    control steps of the raw stream.

    Args:
        chunk_length: Number of policy actions in one chunk.
        overlap_steps: Control steps before chunk end at which to start the next
            inference.
        action_quant_ratio: Micro-actions per macro-action.  ``1`` means no
            quantization.
    """

    chunk_length: int
    overlap_steps: int = 0
    action_quant_ratio: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.chunk_length, bool) or not isinstance(self.chunk_length, int) or self.chunk_length <= 0:
            raise ValueError("chunk_length must be a positive integer")
        if isinstance(self.overlap_steps, bool) or not isinstance(self.overlap_steps, int) or self.overlap_steps < 0:
            raise ValueError("overlap_steps must be a non-negative integer")
        if (
            isinstance(self.action_quant_ratio, bool)
            or not isinstance(self.action_quant_ratio, int)
            or self.action_quant_ratio <= 0
        ):
            raise ValueError("action_quant_ratio must be a positive integer")
        if self.effective_overlap_steps > self.chunk_length:
            raise ValueError(
                f"effective overlap {self.effective_overlap_steps} exceeds chunk_length {self.chunk_length}"
            )

    @property
    def effective_overlap_steps(self) -> int:
        """Overlap in control steps after quantization scaling."""

        return self.overlap_steps * self.action_quant_ratio

    def should_launch_next_inference(self, index: int) -> bool:
        """Whether ``index`` is the trigger point for the next inference."""

        return index == self.chunk_length - self.effective_overlap_steps

    def should_switch_chunk(self, index: int) -> bool:
        """Whether ``index`` is the start of a new chunk cycle."""

        return index == 0

    def should_fetch_observation(self, index: int, running: bool) -> bool:
        """Whether a fresh observation should be captured at ``index``."""

        return (not running) or self.should_launch_next_inference(index)


def estimate_future_state(
    state: object,
    executing_chunk: object,
    chunk_index: int,
    delay: int,
    semantics: str = "absolute",
) -> np.ndarray:
    """Predict the state ``delay`` steps after ``chunk_index`` in a chunk.

    With ``delay`` equal to the remaining steps, ``absolute`` semantics reduces
    to the last action of the chunk — exactly what the upstream VLASH loop
    hardcodes as ``observation["observation.state"] = current_chunk[-1]``.
    """

    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        raise TypeError("chunk_index must be an integer")
    actions = np.asarray(executing_chunk, dtype=np.float64)
    if chunk_index < 0 or chunk_index > len(actions):
        raise ValueError(f"chunk_index {chunk_index} out of range for {len(actions)} action(s)")
    return roll_state_forward(state, actions[chunk_index:], delay, semantics)
