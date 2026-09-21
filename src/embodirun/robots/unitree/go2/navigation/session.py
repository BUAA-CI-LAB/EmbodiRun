"""Continuous language-conditioned navigation task session."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable

from embodirun.utils import Pose2D

from .config import DEFAULT_NAVIGATION_SESSION_CONFIG, NavigationSessionConfig
from .controllers.time_sync import capture_pose
from .controllers.velocity_lease import VelocityLease
from .controllers.waypoint_follower import WorldWaypointFollower
from .events import (
    NavigationEndReason,
    NavigationSessionError,
    NavigationSessionEvent,
    NavigationSessionResult,
)
from .interfaces import MobileBase, NavigationPolicy, ObservationSource
from .motion import MobileBaseState
from .observation import NavigationObservation
from .plan import WaypointPlan
from .request import NavigationRequest
from .telemetry import NavigationTelemetry


class NavigationSession:
    """Track accepted waypoint plans while a slower policy replans."""

    def __init__(
        self,
        policy: NavigationPolicy,
        observations: ObservationSource,
        base: MobileBase,
        *,
        follower: WorldWaypointFollower | None = None,
        config: NavigationSessionConfig = DEFAULT_NAVIGATION_SESSION_CONFIG,
        event_sink: Callable[[NavigationSessionEvent], None] | None = None,
    ) -> None:
        if not isinstance(config, NavigationSessionConfig):
            raise TypeError("config must be a NavigationSessionConfig")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self.policy = policy
        self.observations = observations
        self.base = base
        self.follower = WorldWaypointFollower() if follower is None else follower
        self.config = config
        self._lease = VelocityLease(base, duration_s=config.lease_duration_s)
        self._telemetry = NavigationTelemetry(
            max_events=config.max_events,
            event_sink=event_sink,
        )
        self._running = False
        self._reason: NavigationEndReason | None = None
        self._motion_authorized = False

    @property
    def events(self) -> tuple[NavigationSessionEvent, ...]:
        return self._telemetry.events

    @property
    def event_sink(self) -> Callable[[NavigationSessionEvent], None] | None:
        return self._telemetry.event_sink

    @event_sink.setter
    def event_sink(self, value: Callable[[NavigationSessionEvent], None] | None) -> None:
        self._telemetry.event_sink = value

    @staticmethod
    async def _wait_until_stopped(stop_event: asyncio.Event, delay_s: float) -> None:
        if delay_s <= 0:
            await asyncio.sleep(0)
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay_s)

    async def _capture(
        self,
        episode_id: str,
        *,
        reset: bool,
    ) -> tuple[NavigationObservation, Pose2D]:
        before = await asyncio.to_thread(self.base.state)
        if not isinstance(before, MobileBaseState):
            raise NavigationSessionError("mobile base must return MobileBaseState")
        observation = await asyncio.to_thread(
            self.observations.capture,
            episode_id=episode_id,
            reset=reset,
            robot_state=before.as_observation_metadata(),
        )
        after = await asyncio.to_thread(self.base.state)
        if not isinstance(after, MobileBaseState):
            raise NavigationSessionError("mobile base must return MobileBaseState")
        if not isinstance(observation, NavigationObservation):
            raise NavigationSessionError("observation source must return NavigationObservation")
        if observation.episode_id != episode_id:
            raise NavigationSessionError("observation source changed the episode_id")
        if observation.reset is not reset:
            raise NavigationSessionError("observation source changed the requested reset flag")
        previous_sequence = self._telemetry.last_observation_sequence
        if previous_sequence is not None and observation.sequence <= previous_sequence:
            raise NavigationSessionError("observation sequence must strictly increase")
        self._telemetry.last_observation_sequence = observation.sequence
        anchor = capture_pose(before, after, observation.latest_rgb.captured_at_s)
        self._telemetry.emit("observation_captured", observation_sequence=observation.sequence)
        return observation, anchor

    async def _inference_loop(
        self,
        instruction: str,
        episode_id: str,
        stop_event: asyncio.Event,
    ) -> None:
        reset = True
        while not stop_event.is_set():
            observation, anchor = await self._capture(episode_id, reset=reset)
            reset = False
            self._telemetry.emit("inference_started", observation_sequence=observation.sequence)
            plan = await self.policy.plan(NavigationRequest(instruction, observation))
            self._telemetry.inference_count += 1
            if not isinstance(plan, WaypointPlan):
                raise NavigationSessionError("policy must return a WaypointPlan")
            if plan.observation_sequence != observation.sequence:
                raise NavigationSessionError("waypoint plan sequence must match its navigation observation")
            self.follower.replace(plan, anchor, time.monotonic())
            self._telemetry.plans_accepted += 1
            self._telemetry.emit(
                "plan_accepted",
                observation_sequence=plan.observation_sequence,
                message=f"waypoints={len(plan.waypoints)} terminal={plan.terminal}",
            )
            if plan.terminal:
                await stop_event.wait()
                return

    async def _control_loop(self, stop_event: asyncio.Event) -> None:
        period_s = 1.0 / self.config.control_hz
        next_tick_s = time.monotonic()
        last_waypoint: tuple[int | None, int] | None = None
        while not stop_event.is_set():
            state = await asyncio.to_thread(self.base.state)
            sampled_at_s = time.monotonic()
            sample = self.follower.sample(state.pose, sampled_at_s)
            self._telemetry.control_ticks += 1
            if self.config.execute:
                await asyncio.to_thread(self._lease.send, sample.command)
                self._telemetry.motion_commands += 1
            progress = (self.follower.sequence, sample.waypoint_index)
            if progress != last_waypoint:
                self._telemetry.emit(
                    "control_progress",
                    observation_sequence=self.follower.sequence,
                    waypoint_index=sample.waypoint_index,
                )
                last_waypoint = progress
            if sample.terminal_reached:
                self._reason = "terminal"
                self._telemetry.emit(
                    "terminal_reached",
                    observation_sequence=self.follower.sequence,
                    waypoint_index=sample.waypoint_index,
                )
                stop_event.set()
                return
            next_tick_s += period_s
            now_s = time.monotonic()
            if next_tick_s < now_s - period_s:
                next_tick_s = now_s
            await self._wait_until_stopped(stop_event, next_tick_s - now_s)

    async def _runtime_limit(self, stop_event: asyncio.Event) -> None:
        await self._wait_until_stopped(stop_event, self.config.max_runtime_s)
        if not stop_event.is_set():
            self._reason = "max_runtime"
            self._telemetry.emit("max_runtime")
            stop_event.set()
            await self._force_stop()

    async def _force_stop(self) -> None:
        if not self._motion_authorized:
            return
        await asyncio.shield(asyncio.to_thread(self._lease.stop))
        self._motion_authorized = False

    async def run(
        self,
        instruction: str,
        *,
        episode_id: str | None = None,
    ) -> NavigationSessionResult:
        if self._running:
            raise RuntimeError("this navigation session is already running")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if episode_id is None:
            episode_id = f"navigation-{uuid.uuid4().hex}"
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")

        self._running = True
        self._telemetry.reset()
        self._reason = None
        self._motion_authorized = False
        self.follower.reset_episode()
        stop_event = asyncio.Event()

        self._telemetry.emit("session_started", message=f"execute={self.config.execute}")
        if self.config.execute:
            try:
                await asyncio.to_thread(self.base.preflight)
            except BaseException as error:
                self._telemetry.emit("preflight_failed", message=repr(error))
                self._running = False
                raise
            self._motion_authorized = True
            self._telemetry.emit("preflight_passed")

        tasks = (
            asyncio.create_task(
                self._inference_loop(instruction.strip(), episode_id, stop_event),
                name="navigation-inference",
            ),
            asyncio.create_task(self._control_loop(stop_event), name="navigation-control"),
            asyncio.create_task(self._runtime_limit(stop_event), name="navigation-timeout"),
        )
        try:
            await asyncio.gather(*tasks)
        except BaseException as error:
            stop_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self._force_stop()
            except Exception as stop_error:  # noqa: BLE001 - preserve the primary failure
                if hasattr(error, "add_note"):
                    error.add_note(f"forced stop also failed: {stop_error!r}")
            self._telemetry.emit("session_failed", message=repr(error))
            raise
        else:
            await self._force_stop()
        finally:
            self._running = False

        if self._reason is None:
            raise NavigationSessionError("navigation ended without a terminal reason")
        self._telemetry.emit("session_stopped", message=self._reason)
        return self._telemetry.result(
            episode_id=episode_id,
            reason=self._reason,
        )


__all__ = [
    "DEFAULT_NAVIGATION_SESSION_CONFIG",
    "NavigationSession",
    "NavigationSessionConfig",
]
