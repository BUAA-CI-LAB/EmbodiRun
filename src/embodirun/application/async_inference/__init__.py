"""Execution-side asynchronous VLA inference (RTC and VLASH).

This subpackage owns the *execution* half of two asynchronous action-chunk
algorithms:

* :mod:`~embodirun.application.async_inference.rtc` — Real-Time Chunking action
  queue, latency tracker, and prefix-weight table.
* :mod:`~embodirun.application.async_inference.vlash` — VLASH state roll-forward,
  action quantization, and chunk scheduler.
* :mod:`~embodirun.application.async_inference.contracts` — the versioned
  ``embodiinfer.async.v1`` metadata block shared with the inference engine.
* :mod:`~embodirun.application.async_inference.runtime` — the
  :class:`AsyncControlRuntime` schedule.

The package is pure Python plus numpy and never imports the inference engine; it
talks to it only through the versioned HTTP/WirelessComm API.
"""

from __future__ import annotations

from .contracts import (
    ASYNC_SCHEMA,
    RTCAsyncRequest,
    VlashAsyncRequest,
    build_async_metadata,
)
from .rtc import (
    PREFIX_ATTENTION_SCHEDULES,
    ActionQueue,
    LatencyTracker,
    prefix_weights,
)
from .runtime import (
    AsyncControlRuntime,
    AsyncRuntimeConfig,
    InferenceEvent,
    default_action_encoder,
    default_chunk_extractor,
)
from .vlash import (
    ACTION_SPACE_SEMANTICS,
    VlashScheduler,
    estimate_future_state,
    quantize_actions,
    roll_state_forward,
)

__all__ = [
    "ACTION_SPACE_SEMANTICS",
    "ASYNC_SCHEMA",
    "PREFIX_ATTENTION_SCHEDULES",
    "ActionQueue",
    "AsyncControlRuntime",
    "AsyncRuntimeConfig",
    "InferenceEvent",
    "LatencyTracker",
    "RTCAsyncRequest",
    "VlashAsyncRequest",
    "VlashScheduler",
    "build_async_metadata",
    "default_action_encoder",
    "default_chunk_extractor",
    "estimate_future_state",
    "prefix_weights",
    "quantize_actions",
    "roll_state_forward",
]
