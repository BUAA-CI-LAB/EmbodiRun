"""Execution-side asynchronous VLA inference with RTC.

This subpackage owns the execution half of RTC action chunking:

* :mod:`~embodirun.application.async_inference.rtc` — Real-Time Chunking action
  queue, latency tracker, and prefix-weight table.
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

__all__ = [
    "ASYNC_SCHEMA",
    "PREFIX_ATTENTION_SCHEDULES",
    "ActionQueue",
    "AsyncControlRuntime",
    "AsyncRuntimeConfig",
    "InferenceEvent",
    "LatencyTracker",
    "RTCAsyncRequest",
    "build_async_metadata",
    "default_action_encoder",
    "default_chunk_extractor",
    "prefix_weights",
]
