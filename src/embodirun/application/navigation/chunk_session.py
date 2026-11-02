"""Run complete discrete chunks with one heartbeat lease per chunk.

Timed metric commands do not require translational odometry. Capture and
inference happen at chunk boundaries, after all previously committed actions.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from embodirun.robots.mobile_base.discrete import NavigationCommand, NavigationCommandKind
from embodirun.robots.mobile_base.interfaces import (
    MobileBase,
    MobileBaseError,
    ObservationSource,
    ReleasableMobileBase,
)
from embodirun.robots.mobile_base.motion import PlanarVelocityCommand
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.robots.mobile_base.request import NavigationRequest


@dataclass(frozen=True)
class NavigationChunk:
    """Metric commands tied to the observation that produced them."""

    observation_sequence: int
    commands: tuple[NavigationCommand, ...]

    def __post_init__(self) -> None:
        if type(self.observation_sequence) is not int or self.observation_sequence < 0:
            raise ValueError("observation_sequence must be a non-negative integer")
        if not self.commands or any(not isinstance(c, NavigationCommand) for c in self.commands):
            raise ValueError("chunk must contain navigation commands")
        object.__setattr__(self, "commands", tuple(self.commands))
        for command in self.commands[:-1]:
            if command.kind is NavigationCommandKind.STOP:
                raise ValueError("STOP must be the last command")

    @property
    def terminal(self) -> bool:
        """Whether completing this chunk terminates the task."""
        return self.commands[-1].kind is NavigationCommandKind.STOP


class ChunkPolicy(Protocol):
    """Synchronous policy boundary, called only while the base is idle."""

    def plan(self, request: NavigationRequest) -> NavigationChunk:
        """Return the complete committed chunk; callers must not truncate it."""
        ...


@dataclass(frozen=True)
class ChunkSessionConfig:
    """Timing bounds for open-loop chunks; distances need physical calibration."""

    linear_speed_mps: float = 0.25
    yaw_rate_rps: float = 0.50
    control_hz: float = 10.0
    max_chunk_s: float = 9.0
    max_runtime_s: float = 120.0
    max_chunks: int = 100
    capture_timeout_s: float = 3.0
    max_plan_age_s: float = 5.0
    max_control_gap_s: float = 0.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_chunks) is not int:
            raise ValueError("max_chunks must be an integer")
        # Below the deployed agent's 10 s lease and 2 s heartbeat bounds.
        if self.max_chunk_s > 9.0 or not 5 <= self.control_hz <= 20:
            raise ValueError("max_chunk_s must be <= 9 and control_hz must be in [5,20]")
        if not 1 / self.control_hz < self.max_control_gap_s <= 1.0:
            raise ValueError("max_control_gap_s must exceed one tick and be <= 1")
        PlanarVelocityCommand(self.linear_speed_mps, 0.0, self.yaw_rate_rps)


@dataclass(frozen=True)
class VelocitySegment:
    """A constant body-frame velocity and its intended duration."""

    command: PlanarVelocityCommand
    duration_s: float


DEFAULT_CHUNK_SESSION_CONFIG = ChunkSessionConfig()


def velocity_segments(chunk: NavigationChunk, config: ChunkSessionConfig) -> tuple[VelocitySegment, ...]:
    """Convert every action and merge adjacent equal velocities without truncation."""
    segments: list[VelocitySegment] = []
    for action in chunk.commands:
        if action.kind is NavigationCommandKind.STOP:
            break
        if action.kind is NavigationCommandKind.MOVE_FORWARD:
            command = PlanarVelocityCommand(config.linear_speed_mps, 0.0, 0.0)
            duration = action.distance_m / config.linear_speed_mps
        else:
            sign = 1 if action.kind is NavigationCommandKind.TURN_LEFT else -1
            command = PlanarVelocityCommand(0.0, 0.0, sign * config.yaw_rate_rps)
            duration = math.radians(action.angle_deg) / config.yaw_rate_rps
        if segments and segments[-1].command == command:
            duration += segments.pop().duration_s
        segments.append(VelocitySegment(command, duration))
    if sum(s.duration_s for s in segments) > config.max_chunk_s + 1e-9:
        raise ValueError("complete chunk exceeds max_chunk_s; refusing to truncate recurrent history")
    return tuple(segments)


class ChunkNavigationSession:
    """Run full chunks; dry-run captures and infers once with no control writes.

    One calling thread owns control requests, so cancellation cannot leave a
    background command racing a final stop. Failed model steps are never retried.
    """

    def __init__(
        self,
        policy: ChunkPolicy,
        observations: ObservationSource,
        base: MobileBase,
        *,
        config: ChunkSessionConfig = DEFAULT_CHUNK_SESSION_CONFIG,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy, self.observations, self.base = policy, observations, base
        self.config, self.event_sink = config, event_sink
        self.clock, self.sleep = clock, sleep
        self._motion_attempted = False
        self._lease_id: str | None = None
        self._running = False

    def _emit(self, event: str, **values: object) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, "monotonic_s": self.clock(), **values})

    def _capture(self, episode_id: str, reset: bool, previous: int) -> NavigationObservation:
        state = self.base.state()
        if state.active_velocity_lease_id is not None:
            raise MobileBaseError("capture requires an idle base")
        started = self.clock()
        baseline = None
        while self.clock() - started < self.config.capture_timeout_s:
            observation = self.observations.capture(
                episode_id=episode_id,
                reset=reset,
                robot_state=state.as_observation_metadata(),
            )
            if not isinstance(observation, NavigationObservation):
                raise TypeError("camera must return NavigationObservation")
            if observation.episode_id != episode_id or observation.reset is not reset:
                raise ValueError("camera changed episode identity or reset flag")
            if observation.sequence < previous:
                raise ValueError("camera sequence moved backwards")
            if baseline is None:
                baseline = observation.latest_rgb
            elif observation.sequence > max(previous, baseline.sequence):
                if observation.latest_rgb.captured_at_s <= baseline.captured_at_s:
                    raise ValueError("camera capture time did not increase")
                return observation
            # Await a frame newer than the first read after stopping, without
            # assuming clock synchronization between the camera computer and NX.
            self.sleep(0.02)
        raise TimeoutError("camera did not publish a new post-motion frame")

    def _execute(self, segments: tuple[VelocitySegment, ...], deadline: float) -> None:
        total = sum(s.duration_s for s in segments)
        if not segments:
            return
        self.base.preflight()
        self._motion_attempted = True  # A failed POST may still have been accepted.
        self._lease_id = None
        started = self.clock()
        lease = self.base.start_velocity_lease(segments[0].command, duration_s=total + 0.25)
        if not isinstance(lease, str) or not lease:
            raise MobileBaseError("control service returned an invalid lease id")
        self._lease_id = lease
        segment_index, boundary = 0, segments[0].duration_s
        last_tick = started
        while True:
            now = self.clock()
            if now >= deadline:
                raise TimeoutError("navigation runtime limit reached during motion")
            if now - last_tick > self.config.max_control_gap_s:
                raise TimeoutError("control heartbeat delayed; aborting chunk")
            if now >= started + total:
                if segment_index != len(segments) - 1:
                    raise TimeoutError("control delay skipped an action; aborting chunk")
                break
            while now >= started + boundary and segment_index < len(segments) - 1:
                if now >= started + boundary + segments[segment_index + 1].duration_s:
                    raise TimeoutError("control delay skipped an action; aborting chunk")
                segment_index += 1
                boundary += segments[segment_index].duration_s
                self._emit("segment_started", index=segment_index)
            self.base.update_velocity_lease(lease, segments[segment_index].command)
            last_tick = now
            self.sleep(max(0.0, min(now + 1 / self.config.control_hz, started + boundary) - self.clock()))
        self.base.update_velocity_lease(lease, PlanarVelocityCommand.stopped())
        # /v1/stop also disarms the agent: reserve it for task termination.
        while self.clock() < started + total + 1.0:
            state = self.base.state()
            if state.active_velocity_lease_id is None:
                self._emit("chunk_executed", planned_s=total, elapsed_s=self.clock() - started)
                return
            if state.active_velocity_lease_id != lease:
                raise MobileBaseError("motion ownership changed during chunk completion")
            self.sleep(0.02)
        raise TimeoutError("control lease did not finish")

    def _stop_after_motion(self, *, keep_motion_ready: bool = False) -> None:
        if not self._motion_attempted:
            return
        if keep_motion_ready and self._lease_id is not None:
            # The owner checks identity atomically with cancellation. Never
            # substitute an unscoped stop if another controller took ownership.
            self.base.release_velocity_lease(self._lease_id)
            return
        if self._lease_id is not None:
            try:
                active = self.base.state().active_velocity_lease_id
            except MobileBaseError:
                # If state is unavailable, still attempt the final stop.
                active = self._lease_id
            if active is not None and active != self._lease_id:
                self._emit("ownership_lost", message="another action is active; not stopping its owner")
                return
        self.base.stop()

    def run(
        self,
        instruction: str,
        *,
        execute: bool = False,
        episode_id: str | None = None,
        keep_motion_ready: bool = False,
    ) -> dict[str, object]:
        """Run until STOP/bounds; optionally release a healthy task's own lease.

        Interactive callers can preserve existing readiness on normal completion
        or KeyboardInterrupt. Errors and process termination still disarm.
        """
        if self._running:
            raise RuntimeError("navigation session is already running")
        if not isinstance(execute, bool) or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("execute must be boolean and instruction must be non-empty")
        if not isinstance(keep_motion_ready, bool):
            raise ValueError("keep_motion_ready must be boolean")
        if execute and keep_motion_ready and not isinstance(self.base, ReleasableMobileBase):
            raise MobileBaseError("base does not support scoped lease release")
        episode_id = episode_id or f"go2-{uuid.uuid4().hex}"
        started = self.clock()
        deadline = started + self.config.max_runtime_s
        self._running, self._motion_attempted = True, False
        self._lease_id = None
        release = False
        previous, chunks, actions = -1, 0, 0
        reason = "chunk_limit"
        try:
            if execute:
                self.base.preflight()
            for index in range(self.config.max_chunks):
                if self.clock() >= deadline:
                    reason = "max_runtime"
                    break
                observation = self._capture(episode_id, index == 0, previous)
                previous = observation.sequence
                captured = self.clock()
                chunk = self.policy.plan(NavigationRequest(instruction, observation))
                latency = self.clock() - captured
                if not isinstance(chunk, NavigationChunk) or chunk.observation_sequence != previous:
                    raise ValueError("policy chunk does not match the captured observation")
                segments = velocity_segments(chunk, self.config)
                chunks += 1
                self._emit(
                    "chunk_planned",
                    observation_sequence=previous,
                    inference_roundtrip_s=latency,
                    commands=[c.to_action_values() for c in chunk.commands],
                    segments=len(segments),
                    planned_motion_s=sum(s.duration_s for s in segments),
                )
                if not execute:
                    reason = "dry_run"
                    break
                if latency > self.config.max_plan_age_s:
                    raise TimeoutError("inference result is too old to execute")
                if self.clock() + sum(s.duration_s for s in segments) + 1.0 >= deadline:
                    reason = "max_runtime"
                    break
                self._execute(segments, deadline)
                actions += len(chunk.commands)
                if chunk.terminal:
                    reason = "stop"
                    break
            release = keep_motion_ready
        except KeyboardInterrupt:
            release = keep_motion_ready
            raise
        finally:
            try:
                self._stop_after_motion(keep_motion_ready=release)
            finally:
                self._running = False
        result = {
            "episode_id": episode_id,
            "reason": reason,
            "execute": execute,
            "chunks": chunks,
            "executed_actions": actions,
            "wall_seconds": self.clock() - started,
        }
        self._emit("navigation_finished", **result)
        return result
