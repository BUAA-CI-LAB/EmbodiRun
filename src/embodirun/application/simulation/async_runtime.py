"""Asynchronous chunk execution for one simulator episode.

:class:`embodirun.application.simulation.runtime.SimulationRuntime` runs a
simulator episode *synchronously*: it blocks on inference, then executes the whole
chunk, then blocks again.  A real robot does not have that luxury — inference takes
wall-clock time during which the body keeps moving — so RTC and VLASH exist to keep
the policy aligned with a robot whose state has advanced since the observation it
was conditioned on.

A simulator has no wall clock either: its ``step`` returns immediately.  So this
runtime models the inference window with an explicit, **validated configuration
field**, :attr:`AsyncSimulationConfig.inference_delay_steps` (``Δ``): the number of
simulator steps that elapse while the next chunk is being computed.

    ``Δ`` is a *modeled* delay, not a measured one.  It is the standard way the RTC
    and VLASH papers express the execution/inference overlap in simulation
    (arXiv:2506.07339, arXiv:2512.01031).  The runtime additionally measures the
    real wall-clock time each blocking client call takes and reports it as a
    diagnostic (:class:`AsyncInferenceRecord`), but it deliberately never
    substitutes that measurement for ``Δ`` — the loop is driven by ``Δ`` alone.

The loop reproduces the physical ordering of an asynchronous deployment:

1. Execute the current chunk step by step.
2. At the trigger point ``chunk_index == chunk_steps - overlap_steps`` capture the
   observation and record the current chunk index.  The captured observation is
   what the server will see, so it is intentionally stale by ``Δ`` steps.
3. Keep executing the current chunk for ``Δ`` more simulator steps (the inference
   window).  Those are the actions the server must reason about.
4. Only *now* issue the request, built from the observation captured in step 2 and
   carrying the async metadata for the configured mode.
5. Adopt the returned chunk with the mode's handover rule.

Modes
-----

``sync``
    No overlap and no modeled delay: the next request happens only after the
    current chunk is fully consumed.  No ``async`` block is attached, so the
    request is byte-identical to the synchronous runtime's.
``naive``
    The unstable asynchronous baseline: the request is issued at the trigger from
    the stale observation with *no* conditioning block, and the returned chunk is
    adopted at handover.
``rtc``
    Real-Time Chunking (Black et al., arXiv:2506.07339).  The request carries
    ``inference_delay = Δ`` plus the configured execution horizon and attention
    schedule.  The committed prefix is the model-space leftover of the previously
    issued chunk; this client does not hold model space, so it sends
    ``prev_chunk_left_over = None`` and lets the server slice its own cache.  The
    new chunk is adopted immediately, skipping its first ``Δ`` actions (they
    correspond to time already executed during the window).
``vlash``
    Future-state-aware asynchronous inference (Tang et al., arXiv:2512.01031).  The
    request carries ``delay = Δ``, ``state_fields`` (default
    ``("observation.state",)``) and the ``Δ`` action vectors consumed during the
    window, so the server can roll the captured state forward itself; the new chunk
    is therefore aligned to the current simulator time and is adopted without a
    skip.  The shared ``VlashAsyncRequest`` contract requires one scalar column per
    named state field, so ``state_fields`` must have as many entries as the encoded
    action vector is wide.

Bookkeeping mirrors :class:`~embodirun.application.async_inference.runtime.AsyncControlRuntime`:
an :class:`~embodirun.application.async_inference.rtc.ActionQueue` is kept in
lockstep with the executing chunk (``get`` per executed action, ``merge`` per
adopted chunk) and a :class:`~embodirun.application.async_inference.rtc.LatencyTracker`
collects the measured call latencies.

The module is pure Python plus numpy, imports no simulator/inference package, and
is deterministic under an injected clock.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from embodirun.bindings import BindingMapper
from embodirun.model_services import InferenceClient, PolicyObservation, PolicyResult
from embodirun.robots import RobotAction
from embodirun.simulators import SimulatorAdapter, SimulatorObservation

from ..async_inference.contracts import RTCAsyncRequest, VlashAsyncRequest, build_async_metadata
from ..async_inference.rtc import PREFIX_ATTENTION_SCHEDULES, ActionQueue, LatencyTracker
from ..async_inference.runtime import default_action_encoder, default_chunk_extractor
from ..async_inference.vlash import ACTION_SPACE_SEMANTICS
from .runtime import EpisodeOutcome

__all__ = [
    "ASYNC_SIMULATION_MODES",
    "AsyncEpisodeOutcome",
    "AsyncInferenceRecord",
    "AsyncSimulationConfig",
    "AsyncSimulationRuntime",
    "latency_steps",
]

ASYNC_SIMULATION_MODES = ("sync", "naive", "rtc", "vlash")
"""Every execution mode the runtime understands."""


def latency_steps(latency_s: float, *, control_hz: float) -> int:
    """Convert a measured latency in seconds to whole control steps.

    This is the reported conversion only; it is never used as ``Δ``.  It mirrors
    :meth:`embodirun.application.async_inference.rtc.LatencyTracker.delay_in_steps`
    (including the epsilon that keeps an exact ``n * period`` product from flooring
    to ``n - 1``).
    """

    if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)):
        raise TypeError("control_hz must be a number")
    hertz = float(control_hz)
    if not math.isfinite(hertz) or hertz <= 0:
        raise ValueError("control_hz must be a finite positive number")
    return LatencyTracker().delay_in_steps(latency_s, 1.0 / hertz)


@dataclass(frozen=True, slots=True)
class AsyncSimulationConfig:
    """Static configuration for one asynchronous simulator episode.

    Args:
        mode: One of :data:`ASYNC_SIMULATION_MODES`.
        chunk_steps: Actions per returned chunk (``H``).
        overlap_steps: Steps before chunk end at which the next inference is
            triggered, i.e. the number of old-chunk steps remaining at the trigger.
            Must be ``0`` for ``sync``.
        inference_delay_steps: ``Δ``, the **modeled** number of simulator steps that
            elapse while the next chunk is computed.  It is a simulation parameter,
            not a measured wall-clock latency, and it must not exceed
            ``overlap_steps`` (the chunk remaining at the trigger).
        control_hz: Control rate used to express the *measured* latency in steps.
        execution_horizon: RTC execution horizon; defaults to ``chunk_steps``.
        prefix_attention_schedule: RTC weight schedule.
        max_guidance_weight: RTC guidance clamp.
        hard_prefix: Use the RTC fixed-prefix clamp.
        state_fields: Observation keys the VLASH state plan targets.  Defaults to
            the single LIBERO key ``("observation.state",)``.
        action_space: ``absolute`` or ``delta`` for VLASH.  ``delta`` is the LIBERO
            default.
        action_to_state: Optional action-column -> state-column projection for
            VLASH, with ``-1`` marking an action column that drives no state
            column.  Required when the action and state vectors differ in width
            (LIBERO: ``(0, 1, 2, 3, 4, 5, -1)`` — the single gripper command does
            not map onto the two-finger gripper position).  ``None`` means the
            identity, which requires equal widths.
    """

    mode: str = "sync"
    chunk_steps: int = 50
    overlap_steps: int = 0
    inference_delay_steps: int = 0
    control_hz: float = 20.0
    execution_horizon: int | None = None
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0
    hard_prefix: bool = False
    state_fields: tuple[str, ...] = ("observation.state",)
    action_space: str = "delta"
    action_to_state: tuple[int, ...] | None = None
    action_to_state_scale: tuple[float, ...] | None = None
    vlash_state_source: str = "roll"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in ASYNC_SIMULATION_MODES:
            raise ValueError(f"mode must be one of {', '.join(ASYNC_SIMULATION_MODES)}, got {self.mode!r}")

        if isinstance(self.chunk_steps, bool) or not isinstance(self.chunk_steps, int) or self.chunk_steps <= 0:
            raise ValueError("chunk_steps must be a positive integer")
        if isinstance(self.overlap_steps, bool) or not isinstance(self.overlap_steps, int) or self.overlap_steps < 0:
            raise ValueError("overlap_steps must be a non-negative integer")
        if self.overlap_steps > self.chunk_steps:
            raise ValueError(f"overlap_steps {self.overlap_steps} exceeds chunk_steps {self.chunk_steps}")
        if (
            isinstance(self.inference_delay_steps, bool)
            or not isinstance(self.inference_delay_steps, int)
            or self.inference_delay_steps < 0
        ):
            raise ValueError("inference_delay_steps must be a non-negative integer")

        if self.mode == "sync":
            if self.overlap_steps != 0 or self.inference_delay_steps != 0:
                raise ValueError(
                    "sync mode executes the whole chunk before the next request; "
                    "overlap_steps and inference_delay_steps must both be 0"
                )
        elif self.inference_delay_steps > self.overlap_steps:
            raise ValueError(
                f"inference_delay_steps {self.inference_delay_steps} exceeds the {self.overlap_steps} step(s) "
                "of the current chunk still remaining at the trigger"
            )
        if self.mode == "rtc" and 2 * self.inference_delay_steps > self.chunk_steps:
            # RTC adopts each chunk with its first Δ actions skipped, so every chunk
            # after the bootstrap is only ``chunk_steps - Δ`` long.  The window must
            # still fit before that shortened chunk runs out.
            raise ValueError(
                f"rtc inference_delay_steps {self.inference_delay_steps} leaves only "
                f"{self.chunk_steps - self.inference_delay_steps} action(s) in each adopted chunk; "
                "the inference window must still fit, so 2 * inference_delay_steps must not exceed chunk_steps"
            )

        if (
            isinstance(self.control_hz, bool)
            or not isinstance(self.control_hz, (int, float))
            or not math.isfinite(float(self.control_hz))
            or float(self.control_hz) <= 0
        ):
            raise ValueError("control_hz must be a finite positive number")
        object.__setattr__(self, "control_hz", float(self.control_hz))

        if self.execution_horizon is not None and (
            isinstance(self.execution_horizon, bool)
            or not isinstance(self.execution_horizon, int)
            or self.execution_horizon < 0
        ):
            raise ValueError("execution_horizon must be a non-negative integer or None")
        if self.prefix_attention_schedule not in PREFIX_ATTENTION_SCHEDULES:
            raise ValueError(
                "prefix_attention_schedule must be one of "
                f"{', '.join(PREFIX_ATTENTION_SCHEDULES)}, got {self.prefix_attention_schedule!r}"
            )
        if (
            isinstance(self.max_guidance_weight, bool)
            or not isinstance(self.max_guidance_weight, (int, float))
            or not math.isfinite(float(self.max_guidance_weight))
            or float(self.max_guidance_weight) <= 0
        ):
            raise ValueError("max_guidance_weight must be a finite positive number")
        object.__setattr__(self, "max_guidance_weight", float(self.max_guidance_weight))
        if not isinstance(self.hard_prefix, bool):
            raise TypeError("hard_prefix must be a boolean")

        if isinstance(self.state_fields, str) or not isinstance(self.state_fields, Sequence):
            raise TypeError("state_fields must be a sequence of strings")
        fields = tuple(self.state_fields)
        if self.mode == "vlash" and not fields:
            raise ValueError("state_fields must be non-empty when mode is 'vlash'")
        if self.vlash_state_source not in ("roll", "oracle"):
            raise ValueError("vlash_state_source must be 'roll' or 'oracle'")
        if self.vlash_state_source == "oracle" and self.mode != "vlash":
            raise ValueError("vlash_state_source='oracle' only applies to mode 'vlash'")
        if any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("state_fields entries must be non-empty strings")
        if len(set(fields)) != len(fields):
            raise ValueError("state_fields must be unique")
        object.__setattr__(self, "state_fields", fields)

        if self.action_space not in ACTION_SPACE_SEMANTICS:
            raise ValueError(
                f"action_space must be one of {', '.join(ACTION_SPACE_SEMANTICS)}, got {self.action_space!r}"
            )

    def trigger_index_for(self, chunk_length: int) -> int | None:
        """Chunk index that triggers the next inference, or ``None`` when there is none.

        The trigger is ``chunk_length - overlap_steps`` (clamped at zero), exactly
        as :meth:`embodirun.application.async_inference.vlash.VlashScheduler.should_launch_next_inference`
        does for the full chunk.  Using the *actual* chunk length matters for RTC,
        whose adopted chunks are ``chunk_steps - Δ`` long, so that ``overlap_steps``
        always means "actions still remaining when the next inference starts".

        ``sync`` and every zero-overlap configuration only request a new chunk once
        the current one is exhausted, which is what makes ``Δ = 0`` degenerate to
        the synchronous behaviour.
        """

        if self.mode == "sync" or self.overlap_steps == 0:
            return None
        return max(0, chunk_length - self.overlap_steps)

    @property
    def resolved_execution_horizon(self) -> int:
        """RTC execution horizon, defaulting to ``chunk_steps``."""

        return self.chunk_steps if self.execution_horizon is None else self.execution_horizon


@dataclass(frozen=True, slots=True)
class AsyncInferenceRecord:
    """Diagnostics for one inference request issued by the runtime.

    ``inference_delay_steps`` is the modeled ``Δ`` actually attached to the request,
    while ``measured_latency_s``/``latency_steps`` observe how long the blocking
    client call really took.  They are reported side by side and the measured value
    never drives the loop.
    """

    request_index: int
    mode: str
    trigger: bool
    trigger_chunk_index: int | None
    captured_observation_id: str | None
    environment_steps_at_capture: int
    environment_steps_at_send: int
    inference_delay_steps: int
    measured_latency_s: float
    latency_steps: int
    pending_actions: tuple[tuple[float, ...], ...]
    async_block: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class AsyncEpisodeOutcome:
    """An :class:`EpisodeOutcome` plus the asynchronous diagnostics.

    The five episode fields are duplicated at the top level so the value is a
    drop-in replacement for :class:`EpisodeOutcome`; :meth:`to_episode_outcome`
    returns the plain summary.
    """

    policy_steps: int
    environment_steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    requests: int
    inference_delay_steps: tuple[int, ...]
    measured_latency_s: tuple[float, ...]
    latency_steps: tuple[int, ...]
    records: tuple[AsyncInferenceRecord, ...]

    def to_episode_outcome(self) -> EpisodeOutcome:
        """Return the plain synchronous-compatible summary."""

        return EpisodeOutcome(
            policy_steps=self.policy_steps,
            environment_steps=self.environment_steps,
            total_reward=self.total_reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )


@dataclass(frozen=True, slots=True)
class _Inference:
    """One mapped, validated inference result before it is adopted."""

    actions: tuple[RobotAction, ...]
    vectors: np.ndarray
    originals: np.ndarray
    record: AsyncInferenceRecord


class AsyncSimulationRuntime:
    """Run one simulator episode under asynchronous chunk execution.

    The runtime is a sibling of
    :class:`~embodirun.application.simulation.runtime.SimulationRuntime`, not a
    replacement, and never touches the wall clock to advance the simulator.
    """

    def __init__(
        self,
        simulator: SimulatorAdapter,
        client: InferenceClient,
        *,
        mapper: BindingMapper,
        config: AsyncSimulationConfig | None = None,
        monotonic: Callable[[], float] | None = None,
        latency_tracker: LatencyTracker | None = None,
        action_encoder: Callable[[RobotAction], np.ndarray] | None = None,
        chunk_extractor: Callable[[PolicyResult], np.ndarray] | None = None,
    ) -> None:
        self.simulator = simulator
        self.client = client
        self.mapper = mapper
        self.config = config or AsyncSimulationConfig()
        self._monotonic = monotonic or time.monotonic
        self.latency = latency_tracker if latency_tracker is not None else LatencyTracker(monotonic=self._monotonic)
        self.action_encoder = action_encoder or default_action_encoder
        self.chunk_extractor = chunk_extractor or default_chunk_extractor
        self.session = None

        self.queue = ActionQueue(rtc_enabled=self.config.mode == "rtc")
        self.records: list[AsyncInferenceRecord] = []
        self._prompt = ""
        self._control_period_s = 1.0 / self.config.control_hz

    @property
    def control_period_s(self) -> float:
        """Seconds per control step implied by ``control_hz``."""

        return self._control_period_s

    def run(
        self,
        *,
        instruction: str | None,
        max_policy_steps: int,
        task: str | None = None,
        seed: int | None = None,
    ) -> AsyncEpisodeOutcome:
        """Run one bounded episode and return its outcome plus async diagnostics."""

        if instruction is not None and not instruction.strip():
            raise ValueError("instruction must be non-empty when provided")
        if isinstance(max_policy_steps, bool) or not isinstance(max_policy_steps, int) or max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be a positive integer")

        config = self.config
        current = self.simulator.reset(task=task, seed=seed)
        self._prompt = instruction.strip() if instruction is not None else _instruction(current)
        self.session = self.client.open_session(
            robot_id=self.simulator.simulator_id,
            action_space=self.mapper.policy_action_space,
            metadata={"source": "simulator", "async_mode": config.mode},
        )
        self.queue = ActionQueue(rtc_enabled=config.mode == "rtc")
        self.records = []
        self.latency.reset()

        total_reward = 0.0
        environment_steps = 0
        policy_steps = 0
        terminated = False
        truncated = False

        chunk: tuple[RobotAction, ...] = ()
        vectors = np.empty((0, 0), dtype=np.float64)
        chunk_index = 0
        pending_adopt: _Inference | None = None

        def execute(action: RobotAction) -> None:
            nonlocal current, total_reward, environment_steps, terminated, truncated
            transition = self.simulator.step(action)
            current = transition.observation
            total_reward += transition.reward
            environment_steps += 1
            terminated = transition.terminated
            truncated = transition.truncated
            self.queue.get()

        # Bootstrap: the first chunk is computed from the fresh reset observation.
        # There is no stale observation and no committed prefix yet, so it is not a
        # trigger request and carries no conditioning payload.
        bootstrap = self._infer(
            current,
            request_index=0,
            delay=0,
            pending=(),
            trigger=False,
            environment_steps_at_capture=0,
            environment_steps_at_send=0,
        )
        policy_steps += 1
        self.queue.merge(bootstrap.originals, bootstrap.vectors, 0, None)
        chunk, vectors = bootstrap.actions, bootstrap.vectors
        chunk_index = 0

        while not (terminated or truncated):
            if chunk_index >= len(chunk):
                if pending_adopt is not None:
                    # Naive handover: the old chunk is now fully consumed, so the
                    # append merge collapses to the new chunk from its first action.
                    adopted = pending_adopt
                    pending_adopt = None
                    self.queue.merge(adopted.originals, adopted.vectors, 0, None)
                    chunk, vectors = adopted.actions, adopted.vectors
                    chunk_index = 0
                    continue
                if policy_steps >= max_policy_steps:
                    break
                # Sync-style handover: no overlap remains to start inference early,
                # so the next chunk is requested from the observation reached here.
                fresh = self._infer(
                    current,
                    request_index=policy_steps,
                    delay=0,
                    pending=(),
                    trigger=False,
                    environment_steps_at_capture=environment_steps,
                    environment_steps_at_send=environment_steps,
                )
                policy_steps += 1
                self.queue.merge(fresh.originals, fresh.vectors, 0, None)
                chunk, vectors = fresh.actions, fresh.vectors
                chunk_index = 0
                continue

            trigger_index = config.trigger_index_for(len(chunk))
            if trigger_index is not None and chunk_index == trigger_index:
                if policy_steps >= max_policy_steps:
                    break
                remaining = len(chunk) - chunk_index
                if remaining < config.inference_delay_steps:
                    raise ValueError(
                        f"inference window Δ={config.inference_delay_steps} does not fit in the "
                        f"{remaining} remaining action(s) of the current chunk"
                    )
                captured = current
                captured_index = chunk_index
                captured_environment_steps = environment_steps
                pending: list[tuple[float, ...]] = []
                for _ in range(config.inference_delay_steps):
                    if terminated or truncated:
                        break
                    pending.append(tuple(float(value) for value in vectors[chunk_index]))
                    execute(chunk[chunk_index])
                    chunk_index += 1
                if terminated or truncated:
                    break

                outcome = self._infer(
                    captured,
                    request_index=policy_steps,
                    delay=config.inference_delay_steps,
                    pending=tuple(pending),
                    trigger=True,
                    trigger_chunk_index=captured_index,
                    environment_steps_at_capture=captured_environment_steps,
                    environment_steps_at_send=environment_steps,
                    oracle_state=current,
                )
                policy_steps += 1

                if config.mode == "rtc":
                    resolved = self.queue.merge(
                        outcome.originals,
                        outcome.vectors,
                        config.inference_delay_steps,
                        captured_index,
                    )
                    chunk = outcome.actions[resolved:]
                    vectors = outcome.vectors[resolved:]
                    chunk_index = 0
                elif config.mode == "vlash":
                    # The server rolled the captured state forward by Δ, so the
                    # returned chunk is aligned to the current simulator time.  The
                    # old chunk's unexecuted tail is dropped and the queue is
                    # replaced with the new chunk from its first action.
                    self.queue.clear()
                    self.queue.merge(outcome.originals, outcome.vectors, 0, None)
                    chunk, vectors = outcome.actions, outcome.vectors
                    chunk_index = 0
                else:
                    pending_adopt = outcome
                continue

            execute(chunk[chunk_index])
            chunk_index += 1

        return self._outcome(
            policy_steps=policy_steps,
            environment_steps=environment_steps,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        """Close the policy session, mirroring :class:`SimulationRuntime`."""

        if self.session is not None:
            self.client.close(self.session.session_id)
            self.session = None

    # -- inference -------------------------------------------------------
    def _infer(
        self,
        captured: SimulatorObservation,
        *,
        request_index: int,
        delay: int,
        pending: tuple[tuple[float, ...], ...],
        trigger: bool,
        environment_steps_at_capture: int,
        environment_steps_at_send: int,
        trigger_chunk_index: int | None = None,
        oracle_state: SimulatorObservation | None = None,
    ) -> _Inference:
        if self.session is None:
            raise RuntimeError("async simulation runtime must open a session before inference")
        request = self.mapper.map_observation(
            captured.robot,
            session_id=self.session.session_id,
            request_id=f"async-{request_index}-{uuid.uuid4().hex}",
            step_id=request_index,
            instruction=self._prompt,
            frames=captured.frames,
        )
        metadata: dict[str, Any] = {}
        request_metadata = getattr(request, "metadata", {})
        if isinstance(request_metadata, Mapping):
            metadata.update(request_metadata)
        observation_id = _observation_identity(captured)
        if observation_id is not None:
            metadata.setdefault("observation_id", observation_id)
            metadata.setdefault("snapshot_id", observation_id)

        # VLASH's mechanism is "condition on the execution-time state". The async
        # block exists so the *server* can derive that state from ingredients; when
        # the caller already knows it -- a simulator does, exactly -- the state is
        # simply sent, and no block is needed. That is the faithful reproduction of
        # what VLASH's own LIBERO recipe requires (`use_state_ground_truth=True`),
        # because a state 8 wide and an action 7 wide cannot be related by the
        # action proxy their code refuses.
        oracle = self.config.mode == "vlash" and self.config.vlash_state_source == "oracle" and oracle_state is not None
        if oracle:
            request = _replace_state(request, dict(oracle_state.robot.values))
        metadata = build_async_metadata(
            rtc=self._rtc_request(delay) if self.config.mode == "rtc" else None,
            vlash=(
                self._vlash_request(delay, pending) if self.config.mode == "vlash" and trigger and not oracle else None
            ),
            base=metadata,
        )
        request = _replace_metadata(request, metadata)

        started = self._monotonic()
        result = self.client.step(request)
        measured = max(0.0, self._monotonic() - started)
        self.latency.record(measured)
        measured_steps = self.latency.delay_in_steps(measured, self._control_period_s)

        actions = tuple(self.mapper.map_result(result))
        if not actions:
            raise RuntimeError("binding returned an empty action chunk")
        if any(not isinstance(action, RobotAction) for action in actions):
            raise TypeError("binding action chunk must contain RobotAction values")
        if len(actions) < self.config.chunk_steps:
            raise RuntimeError(
                f"binding returned {len(actions)} action(s), fewer than requested chunk_steps={self.config.chunk_steps}"
            )
        actions = actions[: self.config.chunk_steps]
        vectors = np.asarray([self.action_encoder(action) for action in actions], dtype=np.float64)
        originals = np.asarray(self.chunk_extractor(result), dtype=np.float64)

        async_block = metadata.get("async")
        self.records.append(
            AsyncInferenceRecord(
                request_index=request_index,
                mode=self.config.mode,
                trigger=trigger,
                trigger_chunk_index=trigger_chunk_index,
                captured_observation_id=observation_id,
                environment_steps_at_capture=environment_steps_at_capture,
                environment_steps_at_send=environment_steps_at_send,
                inference_delay_steps=delay,
                measured_latency_s=measured,
                latency_steps=measured_steps,
                pending_actions=tuple(pending),
                async_block=dict(async_block) if isinstance(async_block, Mapping) else None,
            )
        )
        return _Inference(actions=actions, vectors=vectors, originals=originals, record=self.records[-1])

    def _rtc_request(self, delay: int) -> RTCAsyncRequest:
        config = self.config
        # The committed prefix lives in the server's cache: this client holds
        # robot-space actions, not model space, so it sends the delay only.
        return RTCAsyncRequest(
            prev_chunk_left_over=None,
            inference_delay=delay,
            execution_horizon=config.resolved_execution_horizon,
            prefix_attention_schedule=config.prefix_attention_schedule,
            max_guidance_weight=config.max_guidance_weight,
            hard_prefix=config.hard_prefix,
        )

    def _vlash_request(
        self,
        delay: int,
        pending: tuple[tuple[float, ...], ...],
    ) -> VlashAsyncRequest:
        # A named state field may hold a whole vector, so ``state_fields`` does not
        # have to be as wide as the action vector.  When the two widths still differ
        # — LIBERO's 8-wide proprioception against its 7-wide OSC delta action — the
        # plan must say which action column drives which state column; the server
        # refuses to guess, because guessing rolls the wrong joints.
        return VlashAsyncRequest(
            state_fields=self.config.state_fields,
            pending_actions=pending,
            delay=delay,
            action_space=self.config.action_space,
            action_to_state=self.config.action_to_state,
            action_to_state_scale=self.config.action_to_state_scale,
        )

    def _outcome(
        self,
        *,
        policy_steps: int,
        environment_steps: int,
        total_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> AsyncEpisodeOutcome:
        records = tuple(self.records)
        return AsyncEpisodeOutcome(
            policy_steps=policy_steps,
            environment_steps=environment_steps,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            requests=policy_steps,
            inference_delay_steps=tuple(record.inference_delay_steps for record in records),
            measured_latency_s=tuple(record.measured_latency_s for record in records),
            latency_steps=tuple(record.latency_steps for record in records),
            records=records,
        )


def _replace_state(request, state):
    """Return ``request`` with its proprioceptive state replaced, frames kept."""
    if hasattr(request, "__dataclass_fields__"):
        return replace(request, state=state)
    request.state = state
    return request


def _replace_metadata(request: PolicyObservation, metadata: Mapping[str, Any]) -> PolicyObservation:
    """Attach async metadata, using ``replace`` when the request is a dataclass."""

    if hasattr(request, "__dataclass_fields__"):
        return replace(request, metadata=metadata)
    try:
        request.metadata = metadata  # type: ignore[misc]
    except AttributeError as error:  # pragma: no cover - defensive parity with ControlRuntime
        raise TypeError("mapper request must support metadata propagation") from error
    return request


def _observation_identity(observation: SimulatorObservation) -> str | None:
    """Best-effort identity of a captured observation for diagnostics and tests."""

    metadata = observation.robot.metadata
    if not isinstance(metadata, Mapping):
        return None
    for key in ("observation_id", "snapshot_id", "step_index"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def _instruction(observation: SimulatorObservation) -> str:
    instruction = observation.robot.metadata.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("simulator reset did not provide an instruction; pass one explicitly")
    return instruction.strip()
