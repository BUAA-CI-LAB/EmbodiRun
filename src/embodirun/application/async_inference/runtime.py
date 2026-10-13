"""Asynchronous control loop executing naive overlap and RTC schedules.

:class:`AsyncControlRuntime` mirrors :class:`embodirun.application.model_loop.ControlRuntime`
and reuses its session, cancellation, observation-source, and command-sink
handling.  It adds the async schedule: while a chunk is being executed, the next
inference is launched before that chunk is exhausted, the number of actions
consumed during inference is tracked, and RTC prefix information is attached to
request metadata when enabled.

Inference runs on one per-runtime worker while the control thread dispatches
the current action chunk.  Only the control thread mutates queue and safety
state, so cancellation and takeover cannot apply a late result.

* The inference client remains a blocking request-response API, so a single
  worker is used per session.  The request is never issued concurrently with a
  second request; actions continue on the control thread until the worker
  result is ready.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from embodirun.bindings import BindingMapper
from embodirun.model_services import InferenceClient, PolicyObservation, PolicyResult
from embodirun.robots import RobotAction, RobotAdapter, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

from ..model_loop import CommandSink, ControlRuntime
from .contracts import RTCAsyncRequest, build_async_metadata
from .rtc import ActionQueue, LatencyTracker

__all__ = [
    "AsyncControlRuntime",
    "AsyncRuntimeConfig",
    "InferenceEvent",
]


def _flatten_numbers(value: object, out: list[float]) -> None:
    """Flatten numeric leaves of a nested mapping/sequence in insertion order."""

    if value is None or isinstance(value, (bool, str, bytes)):
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isfinite(number):
            out.append(number)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "type":
                continue
            _flatten_numbers(item, out)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            _flatten_numbers(item, out)
        return


def default_action_encoder(action: RobotAction) -> np.ndarray:
    """Encode one robot action as a flat numeric vector.

    The default flattens ``action.values`` in insertion order, skipping the
    ``type`` discriminator.  A binding whose robot-space vector order does not
    match that convention should inject its own ``action_encoder``.
    """

    numbers: list[float] = []
    _flatten_numbers(action.values, numbers)
    if not numbers:
        raise ValueError("robot action does not expose any numeric values to encode")
    return np.asarray(numbers, dtype=np.float64)


def default_chunk_extractor(result: PolicyResult) -> np.ndarray:
    """Extract the raw model-space chunk from a policy result.

    The default looks for the first ``action_chunk`` action and returns its
    ``values.data`` grid, the π0.5 wire shape.  A different policy payload should
    inject its own ``chunk_extractor``.
    """

    for action in result.actions:
        if action.kind == "action_chunk":
            data = action.values.get("data")
            if data is None:
                continue
            chunk = np.asarray(data, dtype=np.float64)
            if chunk.ndim != 2 or len(chunk) == 0:
                raise ValueError("action_chunk values.data must be a non-empty 2D grid")
            return chunk
    raise ValueError("policy result does not contain an action_chunk action with values.data")


@dataclass(frozen=True, slots=True)
class AsyncRuntimeConfig:
    """Static async-inference settings for :class:`AsyncControlRuntime`.

    Args:
        enabled: Master switch.  When false the runtime defers entirely to the
            synchronous :meth:`ControlRuntime.step`.
        rtc_enabled: Attach the RTC block and use RTC queue replacement.
        overlap_steps: Control steps before chunk end at which to launch the next
            inference.
        execution_horizon: RTC execution horizon; defaults to ``chunk_steps``.
        prefix_attention_schedule: RTC weight schedule.
        max_guidance_weight: RTC guidance clamp.
        hard_prefix: Use the RTC fixed-prefix clamp.
        initial_inference_delay: Delay to advertise before any latency sample
            exists.
        artificial_inference_delay_s: Extra wall-clock delay added after each
            inference response before it becomes available to the controller.
    """

    enabled: bool = False
    rtc_enabled: bool = False
    overlap_steps: int = 0
    execution_horizon: int | None = None
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0
    hard_prefix: bool = False
    initial_inference_delay: int = 0
    artificial_inference_delay_s: float = 0.0
    # Explicit prefixes are only safe when the extractor is guaranteed to
    # return model-space actions.  The default uses the server/session cache.
    send_model_space_prefix: bool = False

    def __post_init__(self) -> None:
        for name in ("enabled", "rtc_enabled", "hard_prefix", "send_model_space_prefix"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if isinstance(self.overlap_steps, bool) or not isinstance(self.overlap_steps, int) or self.overlap_steps < 0:
            raise ValueError("overlap_steps must be a non-negative integer")
        if self.execution_horizon is not None and (
            isinstance(self.execution_horizon, bool)
            or not isinstance(self.execution_horizon, int)
            or self.execution_horizon < 0
        ):
            raise ValueError("execution_horizon must be a non-negative integer or None")
        if not isinstance(self.max_guidance_weight, (int, float)) or isinstance(self.max_guidance_weight, bool):
            raise TypeError("max_guidance_weight must be a number")
        weight = float(self.max_guidance_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("max_guidance_weight must be a finite positive number")
        object.__setattr__(self, "max_guidance_weight", weight)
        if (
            isinstance(self.initial_inference_delay, bool)
            or not isinstance(self.initial_inference_delay, int)
            or self.initial_inference_delay < 0
        ):
            raise ValueError("initial_inference_delay must be a non-negative integer")
        if (
            isinstance(self.artificial_inference_delay_s, bool)
            or not isinstance(self.artificial_inference_delay_s, (int, float))
            or not math.isfinite(self.artificial_inference_delay_s)
            or self.artificial_inference_delay_s < 0
        ):
            raise ValueError("artificial_inference_delay_s must be a finite non-negative number")
        object.__setattr__(self, "artificial_inference_delay_s", float(self.artificial_inference_delay_s))

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> AsyncRuntimeConfig:
        """Parse a runtime ``async`` option block without silent downgrade."""

        raw = dict(options or {})
        if not raw:
            return cls()
        if "async_mode" in raw:
            raw.setdefault("mode", raw.pop("async_mode"))
        if "async" in raw:
            block = raw.pop("async")
            if not isinstance(block, Mapping):
                raise ValueError("runtime_options.async must be an object")
            if raw:
                raise ValueError("runtime_options contains keys outside async: " + ", ".join(sorted(raw)))
            raw = dict(block)
        allowed = {
            "enabled",
            "mode",
            "rtc_enabled",
            "overlap_steps",
            "execution_horizon",
            "prefix_attention_schedule",
            "max_guidance_weight",
            "hard_prefix",
            "initial_inference_delay",
            "artificial_inference_delay_s",
            "send_model_space_prefix",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("unsupported async runtime option(s): " + ", ".join(unknown))
        mode = raw.pop("mode", None)
        if mode is not None:
            if mode not in {"sync", "naive", "rtc"}:
                raise ValueError("async mode must be one of 'sync', 'naive', or 'rtc'")
            raw.setdefault("enabled", mode not in {"sync"})
            raw.setdefault("rtc_enabled", mode == "rtc")
        if "enabled" not in raw:
            raw["enabled"] = bool(raw.get("rtc_enabled", False) or mode == "naive")
        enabled = bool(raw.get("enabled", False))
        if enabled and not raw.get("rtc_enabled", False) and mode != "naive":
            raise ValueError("async runtime enabled but rtc_enabled is false and mode is not naive")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    """Bookkeeping for one inference launched by the async schedule."""

    kind: str
    step_id: int
    index_before_inference: int
    inference_delay: int
    measured_latency_s: float
    latency_delay: int
    resolved_delay: int
    actions_consumed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _InferencePayload:
    """Immutable worker output adopted only by the control thread."""

    result: PolicyResult
    actions: tuple[RobotAction, ...]
    vectors: np.ndarray
    originals: np.ndarray
    latency_s: float
    index_before: int
    delay_for_request: int
    generation: int


@dataclass(frozen=True, slots=True)
class _InferenceJob:
    future: Future[_InferencePayload]
    generation: int
    index_before: int
    delay_for_request: int


class AsyncControlRuntime(ControlRuntime):
    """Execute one naive or RTC asynchronous action-chunk schedule on one robot.

    Accepts the same robot/client/mapper/clock surface as :class:`ControlRuntime`
    plus :class:`AsyncRuntimeConfig`.  When ``config.enabled`` is false,
    :meth:`step` delegates to the untouched synchronous implementation.
    """

    def __init__(
        self,
        robot: RobotAdapter,
        client: InferenceClient,
        *,
        instruction: str,
        mapper: BindingMapper,
        chunk_steps: int,
        control_hz: float = 5.0,
        config: AsyncRuntimeConfig | None = None,
        command_sink: CommandSink | None = None,
        observation_source: Callable[[], RobotObservation] | None = None,
        cancel_event: threading.Event | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        latency_tracker: LatencyTracker | None = None,
        action_encoder: Callable[[RobotAction], np.ndarray] | None = None,
        chunk_extractor: Callable[[PolicyResult], np.ndarray] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if monotonic is not None:
            kwargs["monotonic"] = monotonic
        if sleep is not None:
            kwargs["sleep"] = sleep
        super().__init__(
            robot,
            client,
            instruction=instruction,
            mapper=mapper,
            chunk_steps=chunk_steps,
            control_hz=control_hz,
            inference_delay_s=(config or AsyncRuntimeConfig()).artificial_inference_delay_s,
            command_sink=command_sink,
            observation_source=observation_source,
            cancel_event=cancel_event,
            **kwargs,
        )
        self.config = config or AsyncRuntimeConfig()
        self.queue = ActionQueue(rtc_enabled=self.config.rtc_enabled)
        self.latency = latency_tracker if latency_tracker is not None else LatencyTracker(monotonic=self.monotonic)
        self.action_encoder = action_encoder or default_action_encoder
        self.chunk_extractor = chunk_extractor or default_chunk_extractor

        self.events: list[InferenceEvent] = []
        self._chunk: tuple[RobotAction, ...] | None = None
        self._chunk_vectors: np.ndarray | None = None
        self._chunk_index = 0
        # Offset of ``_chunk[0]`` within the server's cached full model chunk.
        # RTC adopts ``actions[delay:]``, so the local index alone no longer
        # identifies the server-cache prefix on later replans.
        self._chunk_model_start_index = 0
        self._pending: _InferencePayload | None = None
        # An RTC response may arrive before the actions advertised in its
        # ``inference_delay`` prefix have actually been sent to the robot. Keep
        # it ready, but do not splice it in until that committed prefix has run.
        self._rtc_pending: _InferencePayload | None = None
        # One worker per runtime is sufficient: requests on a session must stay
        # ordered, while this worker lets the control thread dispatch the old
        # chunk during the network/model wait.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embodirun-infer")
        self._job: _InferenceJob | None = None
        self._generation = 0
        self._closed_async = False
        self._join_timeout_s = 1.0
        self._last_result: PolicyResult | None = None
        self._next_action_deadline_s: float | None = None
        # Without a fixed playback period, wall time cannot be converted into
        # steps.  Keep the most recent number of actions actually dispatched
        # while inference was in flight instead.
        self._latest_unpaced_delay: int | None = None

    # -- public API ------------------------------------------------------
    def step(
        self,
        frames: Sequence[CameraFrame],
        *,
        reset: bool = False,
    ) -> PolicyResult | None:
        """Advance one control step of the async schedule.

        Executes exactly one action and, at the trigger index, launches the next
        inference.  Returns the :class:`PolicyResult` produced this step, or
        ``None`` when no inference ran.
        """

        if not self.config.enabled:
            return super().step(frames, reset=reset)

        if reset:
            self.reset()
        self._raise_if_cancelled()
        frames = tuple(frames)
        self._last_result = None

        # A completed worker is always adopted before a new action is sent.
        # This is the stale-result boundary used by cancellation and takeover.
        self._collect_job(block=False)

        self._ensure_chunk(frames)
        assert self._chunk is not None

        if self._job is None and self._pending is None and self._rtc_pending is None and self._should_launch():
            self._start_job(frames)
            self._collect_job(block=False)
            if self._chunk is None or self._chunk_index >= len(self._chunk):
                self._ensure_chunk(frames)

        action = self._chunk[self._chunk_index]
        self._pace_action()
        self._raise_if_cancelled()
        self._execute(action)
        self.queue.get()
        self._chunk_index += 1
        return self._last_result

    def reset(self) -> None:
        """Reset the session and all async schedule state."""

        self._cancel_job(wait=True)
        self._generation += 1
        super().reset()
        self.queue.clear()
        self._chunk = None
        self._chunk_vectors = None
        self._chunk_index = 0
        self._chunk_model_start_index = 0
        self._pending = None
        self._rtc_pending = None
        self._last_result = None
        self._next_action_deadline_s = None
        self._latest_unpaced_delay = None
        self.events.clear()
        self.latency.reset()

    def close(self) -> None:
        if self._closed_async:
            return
        self._closed_async = True
        self._generation += 1
        self._pending = None
        self._rtc_pending = None
        self._cancel_job()
        # A running HTTP call may not be cancellable.  Wait briefly for a
        # cooperative client, then detach the executor so close never hangs a
        # robot shutdown or an emergency-stop path.
        self._shutdown_executor()
        super().close()

    # -- schedule helpers ------------------------------------------------
    @property
    def effective_overlap_steps(self) -> int:
        """Configured overlap in control steps."""

        return self.config.overlap_steps

    def _trigger_index(self) -> int:
        if self._chunk is None:
            return self.chunk_steps
        length = len(self._chunk)
        return max(0, length - self.effective_overlap_steps)

    def _should_launch(self) -> bool:
        if self._chunk is None:
            return False
        return self._chunk_index == self._trigger_index()

    def _ensure_chunk(self, frames: Sequence[CameraFrame]) -> None:
        if self._chunk is not None and self._chunk_index < len(self._chunk):
            return
        if self._job is not None:
            self._collect_job(block=True)
        if self._rtc_pending is not None:
            if not self._adopt_rtc_pending():
                raise RuntimeError(
                    "RTC action chunk was exhausted before the advertised inference-delay prefix was consumed"
                )
            return
        if self._pending is not None:
            self._adopt_pending()
            return
        self._bootstrap(frames)

    def _adopt(self, actions: Sequence[RobotAction], vectors: np.ndarray, *, skip: int) -> None:
        self._chunk = tuple(actions)[skip:]
        self._chunk_vectors = np.asarray(vectors, dtype=np.float64)[skip:]
        self._chunk_index = 0
        self._chunk_model_start_index = skip

    def _bootstrap(self, frames: Sequence[CameraFrame]) -> None:
        self._start_job(frames, delay_for_request=self.config.initial_inference_delay)
        self._collect_job(block=True)
        if self._pending is not None:
            self._adopt_pending()
        if self._chunk is None:
            raise RuntimeError("bootstrap inference did not produce an action chunk")

    def _start_job(self, frames: Sequence[CameraFrame], *, delay_for_request: int | None = None) -> None:
        if self._job is not None:
            return
        if self._closed_async:
            raise RuntimeError("async runtime is closed")
        index_before = self.queue.get_action_index()
        delay = self._predict_delay() if delay_for_request is None else delay_for_request
        observation = self.robot.observe() if self.observation_source is None else self.observation_source()
        cache_start_index = self._chunk_model_start_index + self._chunk_index
        request = self._build_request(observation, frames, delay, cache_start_index=cache_start_index)
        generation = self._generation
        future = self._executor.submit(self._infer_request, request, index_before, delay, generation)
        self._job = _InferenceJob(future, generation, index_before, delay)
        self.step_id += 1
        # Give the worker a scheduling opportunity before the control thread
        # executes the first overlapped action.  This matters for very fast
        # local clients and deterministic test clocks: dispatching the request
        # is part of the overlap boundary even when the request itself returns
        # immediately.
        time.sleep(0)

    def _collect_job(self, *, block: bool) -> None:
        if self._rtc_pending is not None:
            self._adopt_rtc_pending()
            if self._rtc_pending is not None:
                return
        job = self._job
        if job is None or not (block or job.future.done()):
            return
        try:
            payload = job.future.result(timeout=self._join_timeout_s if block else 0)
        except TimeoutError as error:
            if block:
                raise RuntimeError("inference worker did not finish before the action chunk was exhausted") from error
            return
        except CancelledError:
            self._job = None
            return
        except BaseException:
            self._job = None
            raise
        self._job = None
        if payload.generation != self._generation or self._closed_async:
            return
        self.latency.record(payload.latency_s)
        if self.config.rtc_enabled:
            consumed = max(0, self.queue.get_action_index() - payload.index_before)
            # Bootstrap has no old action chunk to honor. Every subsequent RTC
            # request promises that ``delay_for_request`` old actions will run;
            # a faster-than-predicted response must wait at this handover.
            if self._chunk is not None and consumed < payload.delay_for_request:
                self._rtc_pending = payload
            else:
                self._adopt_rtc_payload(payload)
        else:
            # Naive async adopts at the old-chunk handover.
            self._pending = payload

    def _adopt_rtc_pending(self) -> bool:
        """Adopt a ready RTC result after its promised old-action prefix ran."""

        payload = self._rtc_pending
        if payload is None:
            return False
        consumed = max(0, self.queue.get_action_index() - payload.index_before)
        if consumed < payload.delay_for_request:
            return False
        self._rtc_pending = None
        self._adopt_rtc_payload(payload)
        return True

    def _adopt_rtc_payload(self, payload: _InferencePayload) -> None:
        consumed = max(0, self.queue.get_action_index() - payload.index_before)
        latency_delay = self._latency_delay(payload.latency_s, consumed)
        # Queue alignment follows physical actions actually sent since capture.
        # Wall-time rounding remains a reported/predicted latency signal, but it
        # must not make us replay a prefix action after the robot already ran it.
        resolved = self.queue.merge(payload.originals, payload.vectors, consumed, payload.index_before)
        self._adopt(payload.actions, payload.vectors, skip=resolved)
        self._last_result = payload.result
        self._record_event(
            kind="inference",
            index_before=payload.index_before,
            inference_delay=payload.delay_for_request,
            latency=payload.latency_s,
            latency_delay=latency_delay,
            resolved=resolved,
            consumed=consumed,
            result=payload.result,
        )

    def _adopt_pending(self) -> None:
        assert self._pending is not None
        payload = self._pending
        consumed = max(0, self.queue.get_action_index() - payload.index_before)
        latency_delay = self._latency_delay(payload.latency_s, consumed)
        resolved = self.queue.merge(payload.originals, payload.vectors, latency_delay, payload.index_before)
        self._pending = None
        self._adopt(payload.actions, payload.vectors, skip=0)
        self._last_result = payload.result
        self._record_event(
            kind="inference",
            index_before=payload.index_before,
            inference_delay=payload.delay_for_request,
            latency=payload.latency_s,
            latency_delay=latency_delay,
            resolved=resolved,
            consumed=consumed,
            result=payload.result,
        )

    def _predict_delay(self) -> int:
        if self.action_period_s == 0:
            if not self._latest_unpaced_delay:
                return self.config.initial_inference_delay
            return self._latest_unpaced_delay
        latest = self.latency.latest_delay(self.action_period_s)
        if latest == 0:
            return self.config.initial_inference_delay
        return latest

    def _pace_action(self) -> None:
        """Dispatch asynchronous actions at the configured control rate."""

        if self.action_period_s == 0:
            return

        now = self.monotonic()
        if self._next_action_deadline_s is None:
            self._next_action_deadline_s = now
            return
        deadline = self._next_action_deadline_s + self.action_period_s
        remaining = deadline - now
        if remaining > 0:
            if self.cancel_event is None:
                self.sleep(remaining)
            else:
                self.cancel_event.wait(remaining)
                self._raise_if_cancelled()
            now = self.monotonic()
        self._next_action_deadline_s = max(deadline, now)

    def _latency_delay(self, latency_s: float, consumed: int) -> int:
        """Express latency in steps using the active playback semantics."""

        if self.action_period_s == 0:
            self._latest_unpaced_delay = consumed
            return consumed
        return self.latency.delay_in_steps(latency_s, self.action_period_s)

    # -- inference -------------------------------------------------------
    def _infer_request(
        self,
        request: PolicyObservation,
        index_before: int,
        delay_for_request: int,
        generation: int,
    ) -> _InferencePayload:
        """Run one network request in the worker without mutating schedule state."""

        started = self.monotonic()
        result = self.client.step(request)
        self._wait_for_inference_delay()
        latency = max(0.0, self.monotonic() - started)

        actions = tuple(self.mapper.map_result(result))
        if not actions:
            raise RuntimeError("binding returned an empty action chunk")
        if any(not isinstance(action, RobotAction) for action in actions):
            raise TypeError("binding action chunk must contain RobotAction values")
        if len(actions) < self.chunk_steps:
            raise RuntimeError(
                f"binding returned {len(actions)} action(s), fewer than requested chunk_steps={self.chunk_steps}"
            )
        actions = actions[: self.chunk_steps]
        vectors = np.asarray([self.action_encoder(action) for action in actions], dtype=np.float64)
        originals = np.asarray(self.chunk_extractor(result), dtype=np.float64)
        return _InferencePayload(
            result=result,
            actions=actions,
            vectors=vectors,
            originals=originals,
            latency_s=latency,
            index_before=index_before,
            delay_for_request=delay_for_request,
            generation=generation,
        )

    def _cancel_job(self, *, wait: bool = False) -> None:
        job = self._job
        self._job = None
        if job is not None:
            job.future.cancel()
            if wait:
                with suppress(TimeoutError, CancelledError, BaseException):
                    job.future.result(timeout=self._join_timeout_s)

    def _shutdown_executor(self) -> None:
        # ``Future.cancel`` cannot stop an in-flight socket read.  Bounded join
        # keeps emergency-stop and manual takeover responsive; the worker has no
        # access to mutable schedule state, so detaching it is safe.
        job = self._job
        if job is not None:
            with suppress(TimeoutError, CancelledError, BaseException):
                job.future.result(timeout=self._join_timeout_s)
            self._job = None
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - Python 3.8 compatibility
            self._executor.shutdown(wait=False)

    def _build_request(
        self,
        observation: RobotObservation,
        frames: Sequence[CameraFrame],
        delay_for_request: int,
        *,
        cache_start_index: int,
    ) -> PolicyObservation:
        request = self.mapper.map_observation(
            observation,
            session_id=self.session.session_id,
            request_id=f"step-{self.step_id}-{uuid.uuid4().hex}",
            step_id=self.step_id,
            instruction=self.instruction,
            frames=tuple(frames),
        )
        observation_metadata = getattr(observation, "metadata", {})
        observation_id = (
            observation_metadata.get("observation_id") if isinstance(observation_metadata, Mapping) else None
        )
        metadata: dict[str, Any] = {}
        request_metadata = getattr(request, "metadata", {})
        if isinstance(request_metadata, Mapping):
            metadata.update(request_metadata)
        if isinstance(observation_id, str) and observation_id:
            metadata.setdefault("observation_id", observation_id)
            metadata.setdefault("snapshot_id", observation_id)

        metadata = build_async_metadata(
            rtc=self._rtc_request(delay_for_request, cache_start_index=cache_start_index),
            base=metadata,
        )
        if hasattr(request, "__dataclass_fields__"):
            return replace(request, metadata=metadata)
        try:
            request.metadata = metadata
        except AttributeError as error:  # pragma: no cover - defensive parity with ControlRuntime
            raise TypeError("mapper request must support metadata propagation") from error
        return request

    def _rtc_request(
        self,
        delay_for_request: int,
        *,
        cache_start_index: int,
    ) -> RTCAsyncRequest | None:
        if not self.config.rtc_enabled:
            return None
        left_over = self.queue.get_left_over() if self.config.send_model_space_prefix else None
        return RTCAsyncRequest(
            prev_chunk_left_over=None if left_over is None else left_over.tolist(),
            inference_delay=delay_for_request,
            execution_horizon=(
                self.chunk_steps if self.config.execution_horizon is None else self.config.execution_horizon
            ),
            prefix_attention_schedule=self.config.prefix_attention_schedule,
            max_guidance_weight=self.config.max_guidance_weight,
            hard_prefix=self.config.hard_prefix,
            model_space=self.config.send_model_space_prefix,
            # Explicit prefixes are already sliced locally. Session-cache mode
            # instead needs the executed offset in the server's full chunk; it
            # cannot be reconstructed from network/model latency.
            cache_start_index=None if left_over is not None else cache_start_index,
        )

    def _record_event(
        self,
        *,
        kind: str,
        index_before: int,
        inference_delay: int,
        latency: float,
        latency_delay: int,
        resolved: int,
        consumed: int,
        result: PolicyResult | None,
    ) -> None:
        metadata = getattr(result, "request_id", None)
        self.events.append(
            InferenceEvent(
                kind=kind,
                step_id=self.step_id,
                index_before_inference=index_before,
                inference_delay=inference_delay,
                measured_latency_s=latency,
                latency_delay=latency_delay,
                resolved_delay=resolved,
                actions_consumed=consumed,
                metadata={"request_id": metadata} if metadata is not None else {},
            )
        )
