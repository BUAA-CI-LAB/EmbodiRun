"""Overlap a committed motion tail with one camera/inference continuation.

Only the calling thread can write control commands. The single worker owns
capture and sequential model steps; rejected results terminate that history.
"""

from __future__ import annotations

import math
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from embodirun.robots.mobile_base.discrete import NavigationCommand, NavigationCommandKind
from embodirun.robots.mobile_base.interfaces import MobileBase, MobileBaseError, ReleasableMobileBase
from embodirun.robots.mobile_base.motion import MobileBaseState, PlanarVelocityCommand
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.robots.mobile_base.request import NavigationRequest

from .chunk_session import ChunkNavigationSession, ChunkPolicy, ChunkSessionConfig, NavigationChunk, velocity_segments


class ContinuationPolicy(ChunkPolicy, Protocol):
    """A policy which conditions its next chunk on an immutable execution tail."""

    def plan_after(self, request: NavigationRequest, remaining: tuple[NavigationCommand, ...]) -> NavigationChunk:
        """Return only the suffix to execute after all remaining commands finish."""
        ...


class RenewableBase(MobileBase, Protocol):
    """A heartbeat base with bounded deadline renewal, preserving lease identity."""

    def preflight(self, *, require_renewal: bool = False) -> MobileBaseState:
        """Require an idle armed base and, optionally, renewable stream support."""
        ...

    def update_velocity_lease(
        self, action_id: str, command: PlanarVelocityCommand, *, remaining_s: float | None = None
    ) -> None:
        """Update velocity and optionally replace the remaining lease duration."""
        ...


@dataclass(frozen=True)
class OverlapConfig:
    """Bound prefetch lead and capture-time uncertainty, in local monotonic seconds."""

    lead_s: float = 1.5
    max_capture_uncertainty_s: float = 0.15

    def __post_init__(self) -> None:
        for value in (self.lead_s, self.max_capture_uncertainty_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("overlap timing bounds must be finite and positive")


def remaining_commands(
    chunk: NavigationChunk, elapsed_s: float, config: ChunkSessionConfig
) -> tuple[NavigationCommand, ...]:
    """Estimate the unexecuted tail at capture, retaining partial metric actions.

    This uses the commanded timeline, not measured displacement. The application
    must abort on missed control ticks instead of continuing a divergent history.
    """
    remaining = []
    elapsed = max(0.0, elapsed_s)
    for action in chunk.commands:
        if action.kind is NavigationCommandKind.STOP:
            break
        forward = action.kind is NavigationCommandKind.MOVE_FORWARD
        duration = (
            action.distance_m / config.linear_speed_mps
            if forward
            else math.radians(action.angle_deg) / config.yaw_rate_rps
        )
        if elapsed >= duration:
            elapsed -= duration
            continue
        ratio = 1.0 - elapsed / duration
        remaining.append(
            replace(action, distance_m=action.distance_m * ratio)
            if forward
            else replace(action, angle_deg=action.angle_deg * ratio)
        )
        elapsed = 0.0
    return tuple(remaining)


@dataclass(frozen=True)
class _Plan:
    chunk: NavigationChunk
    started_s: float
    captured_s: float
    ready_s: float
    capture_s: float
    inference_s: float


DEFAULT_OVERLAP_CONFIG = OverlapConfig()


class _RuntimeLimit(TimeoutError):
    """A configured idle wait ended, rather than an inference/control failure."""


class AsyncChunkNavigationSession(ChunkNavigationSession):
    """Prefetch one continuation while retaining the full committed action tail.

    If inference misses the motion boundary the base stops and waits. A late,
    invalid or cancelled result cannot extend motion. Shutdown stops first,
    drains the finite-timeout worker, then permits its owner to close the model
    session. No worker can race that stop with another motion command.
    """

    policy: ContinuationPolicy
    base: RenewableBase

    def __init__(self, *args: object, overlap: OverlapConfig = DEFAULT_OVERLAP_CONFIG, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.overlap = overlap
        self._cancelled = threading.Event()
        self._lease_active = False
        self._last_motion: tuple[float, float] | None = None

    def _frame_time(self, observation: NavigationObservation) -> float:
        timestamp = observation.metadata.get("captured_at_monotonic_s")
        uncertainty = observation.metadata.get("capture_uncertainty_s")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (timestamp, uncertainty)
        ):
            raise ValueError("async capture needs frame_age_s from the updated camera agent")
        if not 0 <= uncertainty <= self.overlap.max_capture_uncertainty_s:
            raise ValueError("camera timing uncertainty exceeds overlap bound")
        if timestamp > self.clock() or self.clock() - timestamp > self.config.max_plan_age_s:
            raise TimeoutError("camera frame is too old for overlap")
        return timestamp

    def _infer(
        self, instruction: str, episode: str, previous: NavigationChunk | None, motion_start: float = 0.0
    ) -> _Plan:
        started = self.clock()
        if previous is None:
            observation = self._capture(episode, True, -1)
        else:
            while True:
                if self._cancelled.is_set():
                    raise RuntimeError("navigation cancelled before capture")
                observation = self.observations.capture(episode_id=episode, reset=False, robot_state={})
                if (
                    not isinstance(observation, NavigationObservation)
                    or observation.episode_id != episode
                    or observation.reset
                ):
                    raise ValueError("camera changed episode identity")
                if observation.sequence < previous.observation_sequence:
                    raise ValueError("camera sequence moved backwards")
                if observation.sequence > previous.observation_sequence:
                    break
                if self.clock() - started >= self.config.capture_timeout_s:
                    raise TimeoutError("camera did not publish a fresh overlap frame")
                self._cancelled.wait(0.02)
        captured = self._frame_time(observation)
        infer_start = self.clock()
        if self._cancelled.is_set():
            raise RuntimeError("navigation cancelled before inference")
        request = NavigationRequest(instruction, observation)
        chunk = (
            self.policy.plan(request)
            if previous is None
            else self.policy.plan_after(request, remaining_commands(previous, captured - motion_start, self.config))
        )
        if not isinstance(chunk, NavigationChunk) or chunk.observation_sequence != observation.sequence:
            raise ValueError("policy chunk does not match the captured observation")
        velocity_segments(chunk, self.config)  # Reject invalid plans before publishing.
        if self._cancelled.is_set():
            raise RuntimeError("navigation cancelled during inference")
        return _Plan(chunk, started, captured, self.clock(), infer_start - started, self.clock() - infer_start)

    def _wait(self, future: Future[_Plan], deadline: float) -> _Plan:
        while not future.done():
            if self.clock() >= deadline:
                raise _RuntimeLimit("navigation runtime limit reached waiting for inference")
            self.sleep(0.01)
        return future.result()

    def _idle(self, deadline: float) -> None:
        if not self._lease_active:
            return
        self.base.update_velocity_lease(self._lease_id, PlanarVelocityCommand.stopped(), remaining_s=0.05)
        end = min(deadline, self.clock() + 1.0)
        while self.clock() < end:
            active = self.base.state().active_velocity_lease_id
            if active is None:
                self._lease_active = False
                return
            if active != self._lease_id:
                raise MobileBaseError("motion ownership changed while stopping for inference")
            self.sleep(0.02)
        raise TimeoutError("control lease did not finish")

    def _move(
        self,
        plan: _Plan,
        pool: ThreadPoolExecutor,
        instruction: str,
        episode: str,
        deadline: float,
        prefetch: bool,
    ) -> Future[_Plan] | None:
        segments = velocity_segments(plan.chunk, self.config)
        if not segments:
            return None
        total = sum(segment.duration_s for segment in segments)
        if not self._lease_active:
            self.base.preflight(require_renewal=True)
        started = self.clock()
        if started - plan.captured_s > self.config.max_plan_age_s:
            raise TimeoutError("inference result is too old to execute")
        if not self._lease_active:
            self._motion_attempted = True
            self._lease_id = None  # An unacknowledged POST may have started motion.
            self._lease_id = self.base.start_velocity_lease(segments[0].command, duration_s=total + 0.75)
            if not isinstance(self._lease_id, str) or not self._lease_id:
                raise MobileBaseError("control service returned an invalid lease id")
            self._lease_active = True
        else:
            self.base.update_velocity_lease(self._lease_id, segments[0].command, remaining_s=total + 0.75)
        future = None
        index, boundary, last_tick = 0, segments[0].duration_s, started
        while True:
            now = self.clock()
            if now >= deadline or now - last_tick > self.config.max_control_gap_s:
                raise TimeoutError("runtime limit or delayed control heartbeat during overlap")
            if future is not None and future.done():
                future.result()  # Model/capture failure stops current motion promptly.
            if now >= started + total:
                if index != len(segments) - 1:
                    raise TimeoutError("control delay skipped an action")
                break
            while now >= started + boundary and index < len(segments) - 1:
                if now >= started + boundary + segments[index + 1].duration_s:
                    raise TimeoutError("control delay skipped an action")
                index += 1
                boundary += segments[index].duration_s
            if future is None and prefetch and started + total - now <= self.overlap.lead_s:
                future = pool.submit(self._infer, instruction, episode, plan.chunk, started)
                self._emit("prefetch_started", remaining_motion_s=started + total - now)
            self.base.update_velocity_lease(
                self._lease_id, segments[index].command, remaining_s=started + total - now + 0.75
            )
            last_tick = now
            self.sleep(max(0.0, min(now + 1 / self.config.control_hz, started + boundary) - self.clock()))
        completed = self.clock()
        self._last_motion = (started, completed)
        self._emit(
            "chunk_executed",
            observation_sequence=plan.chunk.observation_sequence,
            planned_s=total,
            elapsed_s=completed - started,
        )
        if future is None or not future.done():
            self._idle(deadline)
        if future is None and prefetch:
            future = pool.submit(self._infer, instruction, episode, plan.chunk, started)
        self._emit("overlap_boundary", ready=future is not None and future.done(), motion_finished_s=completed)
        return future

    def run(
        self,
        instruction: str,
        *,
        execute: bool = False,
        episode_id: str | None = None,
        keep_motion_ready: bool = False,
    ) -> dict[str, object]:
        """Run serialized asynchronous continuations, with one bounded worker."""
        if not execute:
            return super().run(instruction, execute=execute, episode_id=episode_id, keep_motion_ready=keep_motion_ready)
        if self._running:
            raise RuntimeError("navigation session is already running")
        if execute is not True or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("execute must be boolean and instruction must be non-empty")
        if not isinstance(keep_motion_ready, bool):
            raise ValueError("keep_motion_ready must be boolean")
        if keep_motion_ready and not isinstance(self.base, ReleasableMobileBase):
            raise MobileBaseError("base does not support scoped lease release")
        self.base.preflight(require_renewal=True)
        self._running, self._motion_attempted = True, False
        self._lease_id = None
        self._lease_active = False
        self._last_motion = None
        self._cancelled.clear()
        episode = episode_id or f"go2-{uuid.uuid4().hex}"
        started = self.clock()
        deadline = started + self.config.max_runtime_s
        chunks = actions = 0
        gap = 0.0
        boundary_wait = hidden_latency = 0.0
        reason = "chunk_limit"
        release = False
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="go2-observe-infer")
        try:
            future = pool.submit(self._infer, instruction, episode, None)
            for index in range(self.config.max_chunks):
                waited = self.clock()
                plan = self._wait(future, deadline)
                gap += self.clock() - waited
                age = self.clock() - plan.captured_s
                if age > self.config.max_plan_age_s:
                    raise TimeoutError("inference result is too old to execute")
                total = sum(s.duration_s for s in velocity_segments(plan.chunk, self.config))
                if self.clock() + total + 1.0 >= deadline:
                    reason = "max_runtime"
                    break
                chunks += 1
                boundary_gap = overlap_s = 0.0
                if self._last_motion is not None:
                    motion_start, motion_end = self._last_motion
                    boundary_gap = max(0.0, self.clock() - motion_end)
                    overlap_s = max(0.0, min(motion_end, plan.ready_s) - max(motion_start, plan.started_s))
                    boundary_wait += boundary_gap
                    hidden_latency += overlap_s
                self._emit(
                    "chunk_planned",
                    observation_sequence=plan.chunk.observation_sequence,
                    capture_s=plan.capture_s,
                    inference_roundtrip_s=plan.inference_s,
                    observation_age_s=age,
                    ready_s=plan.ready_s,
                    boundary_wait_s=boundary_gap,
                    capture_inference_overlap_s=overlap_s,
                    commands=[c.to_action_values() for c in plan.chunk.commands],
                    planned_motion_s=total,
                )
                future = self._move(
                    plan,
                    pool,
                    instruction,
                    episode,
                    deadline,
                    prefetch=not plan.chunk.terminal and index + 1 < self.config.max_chunks,
                )
                actions += len(plan.chunk.commands)
                if plan.chunk.terminal:
                    reason = "stop"
                    break
            release = keep_motion_ready
        except _RuntimeLimit:
            reason = "max_runtime"
            release = keep_motion_ready
        except KeyboardInterrupt:
            release = keep_motion_ready
            raise
        finally:
            self._cancelled.set()
            try:
                self._stop_after_motion(keep_motion_ready=release)
            finally:
                # Stop happens BEFORE draining. CLI closes the recurrent session
                # only after this joins, so close cannot race an in-flight step.
                pool.shutdown(wait=True, cancel_futures=True)
                self._running = False
        result = {
            "episode_id": episode,
            "reason": reason,
            "execute": True,
            "chunks": chunks,
            "executed_actions": actions,
            "wall_seconds": self.clock() - started,
            "inference_wait_s": gap,
            "boundary_wait_s": boundary_wait,
            "capture_inference_overlap_s": hidden_latency,
            "mode": "overlap",
        }
        self._emit("navigation_finished", **result)
        return result
