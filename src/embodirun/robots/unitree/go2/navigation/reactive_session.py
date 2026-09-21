"""Image-reactive navigation using bounded open-loop velocity pulses."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Callable

from .config import (
    DEFAULT_REACTIVE_NAVIGATION_SESSION_CONFIG,
    ReactiveNavigationSessionConfig,
)
from .controllers.pulse import VelocityPulseConfig, waypoint_to_velocity_pulse
from .events import (
    NavigationEndReason,
    NavigationSessionError,
    NavigationSessionEvent,
    NavigationSessionResult,
)
from .interfaces import MobileBase, MobileBaseError, NavigationPolicy, ObservationSource
from .motion import MobileBaseState
from .observation import NavigationObservation
from .plan import Waypoint, WaypointPlan
from .request import NavigationRequest
from .telemetry import NavigationTelemetry


def _relative_base_link_waypoint(previous: Waypoint | None, current: Waypoint) -> Waypoint:
    """Express a cumulative capture-frame waypoint in the preceding target frame."""

    if previous is None:
        return current
    dx = current.x_m - previous.x_m
    dy = current.y_m - previous.y_m
    cosine = math.cos(previous.yaw_rad)
    sine = math.sin(previous.yaw_rad)
    return Waypoint(
        x_m=cosine * dx + sine * dy,
        y_m=-sine * dx + cosine * dy,
        yaw_rad=math.atan2(
            math.sin(current.yaw_rad - previous.yaw_rad),
            math.cos(current.yaw_rad - previous.yaw_rad),
        ),
    )


class ReactiveNavigationSession:
    """Capture, plan, execute bounded waypoint pulses, settle, and capture again."""

    def __init__(
        self,
        policy: NavigationPolicy,
        observations: ObservationSource,
        base: MobileBase,
        *,
        config: ReactiveNavigationSessionConfig = DEFAULT_REACTIVE_NAVIGATION_SESSION_CONFIG,
        event_sink: Callable[[NavigationSessionEvent], None] | None = None,
    ) -> None:
        if not isinstance(config, ReactiveNavigationSessionConfig):
            raise TypeError("config must be a ReactiveNavigationSessionConfig")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self.policy = policy
        self.observations = observations
        self.base = base
        self.config = config
        self._telemetry = NavigationTelemetry(
            max_events=config.max_events,
            event_sink=event_sink,
        )
        self._running = False
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

    async def _read_state(self) -> MobileBaseState:
        last_error: MobileBaseError | None = None
        for attempt in range(1, self.config.state_attempts + 1):
            try:
                state = await asyncio.to_thread(self.base.state)
                if not isinstance(state, MobileBaseState):
                    raise NavigationSessionError("mobile base must return MobileBaseState")
                return state
            except MobileBaseError as error:
                last_error = error
                if attempt >= self.config.state_attempts:
                    raise
                self._telemetry.emit("state_retry", message=f"attempt={attempt} error={error}")
                await asyncio.sleep(self.config.state_retry_delay_s)
        assert last_error is not None
        raise last_error

    async def _capture(self, episode_id: str, *, reset: bool) -> NavigationObservation:
        state = await self._read_state()
        observation = await asyncio.to_thread(
            self.observations.capture,
            episode_id=episode_id,
            reset=reset,
            robot_state=state.as_observation_metadata(),
        )
        if not isinstance(observation, NavigationObservation):
            raise NavigationSessionError("observation source must return NavigationObservation")
        if observation.episode_id != episode_id or observation.reset is not reset:
            raise NavigationSessionError("observation source changed episode identity or reset")
        last_sequence = self._telemetry.last_observation_sequence
        if last_sequence is not None and observation.sequence <= last_sequence:
            raise NavigationSessionError("observation sequence must strictly increase")
        self._telemetry.last_observation_sequence = observation.sequence
        self._telemetry.emit("observation_captured", observation_sequence=observation.sequence)
        return observation

    async def _plan(
        self,
        instruction: str,
        observation: NavigationObservation,
    ) -> WaypointPlan:
        self._telemetry.emit("inference_started", observation_sequence=observation.sequence)
        plan = await self.policy.plan(NavigationRequest(instruction, observation))
        self._telemetry.inference_count += 1
        if not isinstance(plan, WaypointPlan):
            raise NavigationSessionError("policy must return a WaypointPlan")
        if plan.observation_sequence != observation.sequence:
            raise NavigationSessionError("waypoint plan sequence must match its navigation observation")
        self._telemetry.plans_accepted += 1
        self._telemetry.emit(
            "plan_accepted",
            observation_sequence=observation.sequence,
            message=f"waypoints={len(plan.waypoints)} terminal={plan.terminal}",
        )
        return plan

    async def _cycle(self, instruction: str, episode_id: str) -> NavigationEndReason:
        reset = True
        while True:
            observation = await self._capture(episode_id, reset=reset)
            reset = False
            plan = await self._plan(instruction, observation)
            if not plan.waypoints:
                if not plan.terminal:
                    raise NavigationSessionError("empty non-terminal waypoint plan")
                self._telemetry.emit(
                    "terminal_reached",
                    observation_sequence=observation.sequence,
                )
                return "terminal"

            pulse_config = VelocityPulseConfig(
                linear_speed_mps=self.config.linear_speed_mps,
                yaw_rate_rps=self.config.yaw_rate_rps,
                min_duration_s=self.config.min_pulse_s,
                max_duration_s=self.config.max_pulse_s,
                limits=self.config.limits,
            )
            selected = plan.waypoints[: self.config.max_waypoints_per_observation]
            previous: Waypoint | None = None
            for waypoint_index, waypoint in enumerate(selected):
                relative = _relative_base_link_waypoint(previous, waypoint)
                previous = waypoint
                command, duration_s = waypoint_to_velocity_pulse(relative, pulse_config)
                self._telemetry.emit(
                    "pulse_planned",
                    observation_sequence=observation.sequence,
                    waypoint_index=waypoint_index,
                    message=(
                        f"vx={command.vx_mps:.3f} vy={command.vy_mps:.3f} "
                        f"yaw_rate={command.yaw_rate_rps:.3f} duration_s={duration_s:.3f}"
                    ),
                )
                if self.config.execute and command.moving:
                    await asyncio.to_thread(
                        self.base.start_velocity_lease,
                        command,
                        duration_s=duration_s,
                    )
                    self._telemetry.motion_commands += 1
                    self._telemetry.emit(
                        "pulse_started",
                        observation_sequence=observation.sequence,
                        waypoint_index=waypoint_index,
                    )
                    # Each bounded lease expires itself. Calling stop between
                    # pulses would break bases whose stop endpoint also revokes readiness.
                    await asyncio.sleep(duration_s + self.config.settle_s)
                elif self.config.execute:
                    await asyncio.sleep(self.config.settle_s)

            if plan.terminal and self.config.terminal_after_waypoints and len(selected) == len(plan.waypoints):
                self._telemetry.emit(
                    "terminal_reached",
                    observation_sequence=observation.sequence,
                    message="terminal waypoint chunk completed",
                )
                return "terminal"

    async def _force_stop(self) -> None:
        if not self._motion_authorized:
            return
        await asyncio.shield(asyncio.to_thread(self.base.stop))
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
        self._motion_authorized = False
        reason: NavigationEndReason | None = None
        self._telemetry.emit(
            "session_started",
            message=f"execute={self.config.execute} reactive=true",
        )
        try:
            if self.config.execute:
                try:
                    await asyncio.to_thread(self.base.preflight)
                except BaseException as error:
                    self._telemetry.emit("preflight_failed", message=repr(error))
                    raise
                self._motion_authorized = True
                self._telemetry.emit("preflight_passed")
            try:
                reason = await asyncio.wait_for(
                    self._cycle(instruction.strip(), episode_id),
                    timeout=self.config.max_runtime_s,
                )
            except asyncio.TimeoutError:
                reason = "max_runtime"
                self._telemetry.emit("max_runtime")
        except BaseException as error:
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

        if reason is None:
            raise NavigationSessionError("navigation ended without a terminal reason")
        self._telemetry.emit("session_stopped", message=reason)
        return self._telemetry.result(
            episode_id=episode_id,
            reason=reason,
        )


__all__ = [
    "DEFAULT_REACTIVE_NAVIGATION_SESSION_CONFIG",
    "ReactiveNavigationSession",
    "ReactiveNavigationSessionConfig",
    "waypoint_to_velocity_pulse",
]
