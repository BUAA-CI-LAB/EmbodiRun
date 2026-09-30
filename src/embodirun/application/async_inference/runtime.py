"""Asynchronous control loop executing RTC and VLASH schedules.

:class:`AsyncControlRuntime` mirrors :class:`embodirun.application.model_loop.ControlRuntime`
and reuses its session, cancellation, observation-source, and command-sink
handling.  It adds the async schedule: while a chunk is being executed, the next
inference is launched before that chunk is exhausted, the number of actions
consumed during inference is tracked, the RTC leftover and the VLASH
future-state ingredients are attached to the request metadata, and action
quantization is applied when the request is built.

The class is scheduling and bookkeeping only.  It requires no robot hardware and
is deterministic under an injected clock and sleep function.

Judgement calls (documented because the repository has no absolute/delta action
space notion, and because EmbodiRun's inference client is blocking):

* ``action_space`` is an explicit configuration parameter defaulting to
  ``"absolute"``, the π0.5 / LeRobot convention.  Robot action spaces such as
  ``arx.x5.eef_xyzrpy_gripper.absolute.v1`` encode ``absolute`` in their name,
  but no binding exposes a machine-readable absolute/delta flag, so it cannot be
  inferred from the binding.
* The inference call is synchronous, so the runtime models the overlap window
  explicitly: the actions that would have executed while the call was in flight
  are executed immediately after it returns (RTC), or the remainder of the
  current chunk is executed through to handover (VLASH).  The bookkeeping — the
  index captured before inference, the actions consumed during it, and the delay
  resolved by :meth:`ActionQueue.merge` — is identical to a threaded
  implementation.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from embodirun.bindings import BindingMapper
from embodirun.model_services import InferenceClient, PolicyObservation, PolicyResult
from embodirun.robots import RobotAction, RobotAdapter, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

from ..model_loop import CommandSink, ControlRuntime
from .contracts import RTCAsyncRequest, VlashAsyncRequest, build_async_metadata
from .rtc import ActionQueue, LatencyTracker
from .vlash import (
    ACTION_SPACE_SEMANTICS,
    VlashScheduler,
    estimate_future_state,
    quantize_actions,
)

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
        vlash_enabled: Attach the VLASH block.
        overlap_steps: Control steps before chunk end at which to launch the next
            inference.
        action_quant_ratio: Micro-actions per macro-action.  ``1`` disables
            quantization.
        action_space: ``absolute`` or ``delta``.  Defaults to ``"absolute"``.
        execution_horizon: RTC execution horizon; defaults to ``chunk_steps``.
        prefix_attention_schedule: RTC weight schedule.
        max_guidance_weight: RTC guidance clamp.
        hard_prefix: Use the RTC fixed-prefix clamp.
        state_fields: Observation keys the VLASH state plan targets.
        initial_inference_delay: Delay to advertise before any latency sample
            exists.
    """

    enabled: bool = False
    rtc_enabled: bool = False
    vlash_enabled: bool = False
    overlap_steps: int = 0
    action_quant_ratio: int = 1
    action_space: str = "absolute"
    execution_horizon: int | None = None
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0
    hard_prefix: bool = False
    state_fields: tuple[str, ...] = ()
    initial_inference_delay: int = 0

    def __post_init__(self) -> None:
        for name in ("enabled", "rtc_enabled", "vlash_enabled", "hard_prefix"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if isinstance(self.overlap_steps, bool) or not isinstance(self.overlap_steps, int) or self.overlap_steps < 0:
            raise ValueError("overlap_steps must be a non-negative integer")
        if (
            isinstance(self.action_quant_ratio, bool)
            or not isinstance(self.action_quant_ratio, int)
            or self.action_quant_ratio <= 0
        ):
            raise ValueError("action_quant_ratio must be a positive integer")
        if self.action_space not in ACTION_SPACE_SEMANTICS:
            raise ValueError(
                f"action_space must be one of {', '.join(ACTION_SPACE_SEMANTICS)}, got {self.action_space!r}"
            )
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
        if isinstance(self.state_fields, str) or not isinstance(self.state_fields, Sequence):
            raise TypeError("state_fields must be a sequence of strings")
        fields = tuple(self.state_fields)
        if any(not isinstance(item, str) or not item for item in fields):
            raise ValueError("state_fields entries must be non-empty strings")
        if len(set(fields)) != len(fields):
            raise ValueError("state_fields must be unique")
        object.__setattr__(self, "state_fields", fields)
        if (
            isinstance(self.initial_inference_delay, bool)
            or not isinstance(self.initial_inference_delay, int)
            or self.initial_inference_delay < 0
        ):
            raise ValueError("initial_inference_delay must be a non-negative integer")
        if self.vlash_enabled and not fields:
            raise ValueError("state_fields must be non-empty when VLASH is enabled")


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


class AsyncControlRuntime(ControlRuntime):
    """Execute one RTC/VLASH asynchronous action-chunk schedule on one robot.

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
            command_sink=command_sink,
            observation_source=observation_source,
            cancel_event=cancel_event,
            **kwargs,
        )
        self.config = config or AsyncRuntimeConfig()
        self.scheduler = VlashScheduler(
            chunk_length=self.chunk_steps,
            overlap_steps=self.config.overlap_steps,
            action_quant_ratio=self.config.action_quant_ratio,
        )
        self.queue = ActionQueue(rtc_enabled=self.config.rtc_enabled)
        self.latency = latency_tracker if latency_tracker is not None else LatencyTracker(monotonic=self.monotonic)
        self.action_encoder = action_encoder or default_action_encoder
        self.chunk_extractor = chunk_extractor or default_chunk_extractor

        self.events: list[InferenceEvent] = []
        self.future_state: np.ndarray | None = None
        self._chunk: tuple[RobotAction, ...] | None = None
        self._chunk_vectors: np.ndarray | None = None
        self._chunk_index = 0
        self._pending: tuple[tuple[RobotAction, ...], np.ndarray, np.ndarray, int, int, int, float] | None = None
        self._last_result: PolicyResult | None = None

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

        self._ensure_chunk(frames)
        assert self._chunk is not None

        if self._pending is None and self._should_launch():
            self._launch(frames)
            if self._chunk is None or self._chunk_index >= len(self._chunk):
                self._ensure_chunk(frames)

        action = self._chunk[self._chunk_index]
        self._raise_if_cancelled()
        self._execute(action)
        self.queue.get()
        self._chunk_index += 1
        return self._last_result

    def reset(self) -> None:
        """Reset the session and all async schedule state."""

        super().reset()
        self.queue.clear()
        self._chunk = None
        self._chunk_vectors = None
        self._chunk_index = 0
        self._pending = None
        self._last_result = None
        self.future_state = None
        self.events.clear()
        self.latency.reset()

    def close(self) -> None:
        self._pending = None
        super().close()

    # -- schedule helpers ------------------------------------------------
    @property
    def effective_overlap_steps(self) -> int:
        """Quantization-scaled overlap in control steps."""

        return self.scheduler.effective_overlap_steps

    def _trigger_index(self) -> int:
        if self._chunk is None:
            return self.chunk_steps
        length = len(self._chunk)
        if length == self.scheduler.chunk_length:
            return self.scheduler.chunk_length - self.scheduler.effective_overlap_steps
        return max(0, length - self.scheduler.effective_overlap_steps)

    def _should_launch(self) -> bool:
        if self._chunk is None:
            return False
        return self._chunk_index == self._trigger_index()

    def _ensure_chunk(self, frames: Sequence[CameraFrame]) -> None:
        if self._chunk is not None and self._chunk_index < len(self._chunk):
            return
        if self._pending is not None:
            self._adopt_pending()
            return
        self._bootstrap(frames)

    def _adopt(self, actions: Sequence[RobotAction], vectors: np.ndarray, *, skip: int) -> None:
        self._chunk = tuple(actions)[skip:]
        self._chunk_vectors = np.asarray(vectors, dtype=np.float64)[skip:]
        self._chunk_index = 0

    def _bootstrap(self, frames: Sequence[CameraFrame]) -> None:
        result, actions, vectors, originals, _latency, _index_before = self._infer(
            frames,
            delay_for_request=self.config.initial_inference_delay,
        )
        self.queue.merge(originals, vectors, 0, None)
        self._adopt(actions, vectors, skip=0)
        self._last_result = result

    def _launch(self, frames: Sequence[CameraFrame]) -> None:
        index_before = self.queue.get_action_index()
        delay_for_request = self._predict_delay()
        result, actions, vectors, originals, latency, index_before = self._infer(
            frames,
            delay_for_request=delay_for_request,
            index_before=index_before,
        )
        latency_delay = self.latency.delay_in_steps(latency, self.action_period_s)
        self._last_result = result

        if self.config.rtc_enabled:
            consumed = self._drain_overlap(latency_delay)
            resolved = self.queue.merge(originals, vectors, latency_delay, index_before)
            self._adopt(actions, vectors, skip=resolved)
            self._record_event(
                kind="inference",
                index_before=index_before,
                inference_delay=delay_for_request,
                latency=latency,
                latency_delay=latency_delay,
                resolved=resolved,
                consumed=consumed,
                result=result,
            )
            return

        # VLASH keeps executing the current chunk through to handover; the queue
        # (and the event) are resolved when the pending chunk is adopted.
        self._pending = (
            tuple(actions),
            np.asarray(vectors, dtype=np.float64),
            originals,
            latency_delay,
            index_before,
            delay_for_request,
            latency,
        )

    def _adopt_pending(self) -> None:
        assert self._pending is not None
        actions, vectors, originals, latency_delay, index_before, delay_for_request, latency = self._pending
        consumed = max(0, self.queue.get_action_index() - index_before)
        resolved = self.queue.merge(originals, vectors, latency_delay, index_before)
        self._pending = None
        self._adopt(actions, vectors, skip=0)
        self._record_event(
            kind="inference",
            index_before=index_before,
            inference_delay=delay_for_request,
            latency=latency,
            latency_delay=latency_delay,
            resolved=resolved,
            consumed=consumed,
            result=self._last_result,
        )

    def _drain_overlap(self, count: int) -> int:
        """Execute up to ``count`` actions that ran while inference was in flight."""

        executed = 0
        if self._chunk is None:
            return 0
        while executed < count and self._chunk_index < len(self._chunk):
            self._raise_if_cancelled()
            self._execute(self._chunk[self._chunk_index])
            self.queue.get()
            self._chunk_index += 1
            executed += 1
        return executed

    def _predict_delay(self) -> int:
        latest = self.latency.latest_delay(self.action_period_s)
        if latest == 0:
            return self.config.initial_inference_delay
        return latest

    # -- inference -------------------------------------------------------
    def _infer(
        self,
        frames: Sequence[CameraFrame],
        *,
        delay_for_request: int,
        index_before: int | None = None,
    ) -> tuple[PolicyResult, tuple[RobotAction, ...], np.ndarray, np.ndarray, float, int]:
        observation = self.robot.observe() if self.observation_source is None else self.observation_source()
        if index_before is None:
            index_before = self.queue.get_action_index()
        request = self._build_request(observation, frames, delay_for_request)
        started = self.monotonic()
        result = self.client.step(request)
        latency = max(0.0, self.monotonic() - started)
        self.latency.record(latency)
        self._raise_if_cancelled()

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
        self.step_id += 1
        return result, actions, vectors, originals, latency, index_before

    def _build_request(
        self,
        observation: RobotObservation,
        frames: Sequence[CameraFrame],
        delay_for_request: int,
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
            rtc=self._rtc_request(delay_for_request),
            vlash=self._vlash_request(observation),
            base=metadata,
        )
        if hasattr(request, "__dataclass_fields__"):
            return replace(request, metadata=metadata)
        try:
            request.metadata = metadata
        except AttributeError as error:  # pragma: no cover - defensive parity with ControlRuntime
            raise TypeError("mapper request must support metadata propagation") from error
        return request

    def _rtc_request(self, delay_for_request: int) -> RTCAsyncRequest | None:
        if not self.config.rtc_enabled:
            return None
        left_over = self.queue.get_left_over()
        return RTCAsyncRequest(
            prev_chunk_left_over=None if left_over is None else left_over.tolist(),
            inference_delay=delay_for_request,
            execution_horizon=(
                self.chunk_steps if self.config.execution_horizon is None else self.config.execution_horizon
            ),
            prefix_attention_schedule=self.config.prefix_attention_schedule,
            max_guidance_weight=self.config.max_guidance_weight,
            hard_prefix=self.config.hard_prefix,
        )

    def _vlash_request(self, observation: RobotObservation) -> VlashAsyncRequest | None:
        if not self.config.vlash_enabled or self._chunk is None or self._chunk_vectors is None:
            return None
        remaining = self._chunk_vectors[self._chunk_index :]
        if len(remaining) == 0:
            return None
        macro = np.asarray(
            quantize_actions(remaining, self.config.action_quant_ratio, self.config.action_space),
            dtype=np.float64,
        )
        self.future_state = self._future_state(observation, macro)
        return VlashAsyncRequest(
            state_fields=self.config.state_fields,
            pending_actions=macro.tolist(),
            delay=len(macro),
            action_space=self.config.action_space,
        )

    def _future_state(self, observation: RobotObservation, pending: np.ndarray) -> np.ndarray:
        state = self._state_vector(observation)
        return estimate_future_state(state, pending, 0, len(pending), self.config.action_space)

    def _state_vector(self, observation: RobotObservation) -> np.ndarray:
        values = observation.values
        if not isinstance(values, Mapping):
            raise ValueError("observation.values must be a mapping for VLASH state_fields")
        missing = [name for name in self.config.state_fields if name not in values]
        if missing:
            raise ValueError(f"observation is missing VLASH state field(s): {', '.join(missing)}")
        numbers: list[float] = []
        for name in self.config.state_fields:
            item = values[name]
            if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
                raise ValueError(f"observation state field {name!r} must be numeric")
            numbers.append(float(item))
        return np.asarray(numbers, dtype=np.float64)

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
