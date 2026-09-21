"""Synchronous adapter for the inspected XLeRobot/teleop command API.

The existing XLeRobot teleop client accepts a mapping containing ``x.vel`` in
metres per second and ``theta.vel`` in degrees per second.  It is a two-wheel
base, so lateral velocity is not a supported command.  The client also does
not provide metric x/y/yaw pose or odometry.  ``TeleopLocalization`` is
therefore required from the caller and is the only source of pose and
collision evidence here; this adapter never integrates accepted commands into
synthetic odometry.

Construction is side-effect free.  The caller supplies an already configured
transport object (for example the existing ``RemoteRobot``), and connection,
arming, ownership, watchdog, and stop interlocks remain in that object.
When used through ``LightNav0Controller``, feedback or command failures trigger
the controller's best-effort stop path before the original exception is
re-raised.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .local_segment import LocalSegmentFeedback, LocalSegmentLeaseLost
from .tracker import BodyVelocity, Feedback, Pose2D, XLeRobotBackend


class XLeRobotTeleopError(RuntimeError):
    """Base error raised when the concrete teleop boundary cannot be used."""


class LocalizationUnavailable(XLeRobotTeleopError):
    """The injected metric pose provider returned no usable sample."""


class LocalizationStale(XLeRobotTeleopError):
    """The injected metric pose sample is outside the freshness bound."""


class XLeRobotTeleopCommandError(ValueError):
    """A command cannot be represented by the inspected two-wheel API."""


class XLeRobotFeedbackCached(XLeRobotTeleopError):
    """The source returned its bounded cached state sample."""


class XLeRobotStopUnconfirmed(XLeRobotTeleopError):
    """The concrete robot did not confirm that its stop completed."""


@runtime_checkable
class XLeRobotTeleopRobot(Protocol):
    """Duck-typed subset of the existing synchronous teleop robot."""

    def read(self) -> Any: ...

    def command(self, action: Mapping[str, float]) -> Any: ...

    def stop(self) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TeleopLocalization:
    """One fresh metric localization sample supplied by the caller."""

    pose: Pose2D
    observed_at_s: float
    collision: bool = False
    contact_events: tuple[Mapping[str, Any], ...] = ()
    final_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose2D):
            raise TypeError("localization.pose must be a Pose2D")
        if isinstance(self.observed_at_s, bool) or not isinstance(self.observed_at_s, (int, float)):
            raise TypeError("localization.observed_at_s must be a finite number")
        if not math.isfinite(float(self.observed_at_s)):
            raise ValueError("localization.observed_at_s must be finite")
        if not isinstance(self.collision, bool):
            raise TypeError("localization.collision must be boolean")
        if not isinstance(self.contact_events, tuple):
            raise TypeError("localization.contact_events must be a tuple")
        if not isinstance(self.final_state, Mapping):
            raise TypeError("localization.final_state must be a mapping")


@runtime_checkable
class TeleopLocalizationProvider(Protocol):
    """Produce metric pose feedback from each freshly read robot observation."""

    def __call__(self, observation: Any) -> TeleopLocalization: ...


def _observation_state(observation: Any) -> Mapping[str, Any]:
    """Extract the state mapping returned by RemoteRobot.read()."""

    if isinstance(observation, tuple):
        if not observation:
            raise XLeRobotTeleopError("XLeRobot read returned an empty tuple")
        observation = observation[0]
    if not isinstance(observation, Mapping):
        raise XLeRobotTeleopError("XLeRobot read must return a mapping or (mapping, images)")
    state = observation.get("state", observation)
    if not isinstance(state, Mapping):
        raise XLeRobotTeleopError("XLeRobot observation state must be a mapping")
    return state


def _number(state: Mapping[str, Any], name: str) -> float:
    value = state.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise XLeRobotTeleopError(f"XLeRobot observation {name!r} must be finite")
    return float(value)


class XLeRobotTeleopBackend(XLeRobotBackend):
    """Adapt the existing teleop robot while requiring external localization.

    ``localize`` receives the freshly read observation (the first element of
    ``RemoteRobot.read()`` when images are enabled) and must return a
    ``TeleopLocalization`` with timestamps in the same monotonic domain as
    ``clock``.  The adapter reads velocity feedback from the robot's
    ``x.vel``/``theta.vel`` fields and converts the latter from degrees per
    second to radians per second.  It never calls ``arm`` or ``connect``.
    """

    def __init__(
        self,
        robot: XLeRobotTeleopRobot,
        localize: TeleopLocalizationProvider | Callable[[Any], TeleopLocalization],
        *,
        max_linear_velocity_m_s: float = 0.3,
        max_angular_velocity_rad_s: float = 0.6,
        localization_timeout_s: float = 0.5,
        lateral_tolerance_m_s: float = 1e-9,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(robot, XLeRobotTeleopRobot):
            raise TypeError("robot must implement the existing synchronous teleop API")
        if not callable(localize):
            raise TypeError("localize must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        for name, value in (
            ("max_linear_velocity_m_s", max_linear_velocity_m_s),
            ("max_angular_velocity_rad_s", max_angular_velocity_rad_s),
            ("localization_timeout_s", localization_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(lateral_tolerance_m_s, bool)
            or not isinstance(lateral_tolerance_m_s, (int, float))
            or not math.isfinite(float(lateral_tolerance_m_s))
            or float(lateral_tolerance_m_s) < 0
        ):
            raise ValueError("lateral_tolerance_m_s must be finite and non-negative")
        self.robot = robot
        self.localize = localize
        self.max_linear_velocity_m_s = float(max_linear_velocity_m_s)
        self.max_angular_velocity_rad_s = float(max_angular_velocity_rad_s)
        self.localization_timeout_s = float(localization_timeout_s)
        self.lateral_tolerance_m_s = float(lateral_tolerance_m_s)
        self._clock = clock
        self.clock = clock
        self.closed = False
        self.last_stop_report: Mapping[str, Any] | None = None

    def feedback(self) -> Feedback:
        if self.closed:
            raise XLeRobotTeleopError("XLeRobotTeleopBackend is closed")
        raw = self.robot.read()
        state = _observation_state(raw)
        localization = self.localize(raw[0] if isinstance(raw, tuple) else raw)
        if not isinstance(localization, TeleopLocalization):
            raise LocalizationUnavailable("localization provider must return TeleopLocalization")
        now = float(self._clock())
        age = now - float(localization.observed_at_s)
        if not math.isfinite(age) or age < -self.localization_timeout_s or age > self.localization_timeout_s:
            raise LocalizationStale(f"localization sample age {age:.3f}s exceeds {self.localization_timeout_s:.3f}s")
        velocity = BodyVelocity(
            _number(state, "x.vel"),
            0.0,
            math.radians(_number(state, "theta.vel")),
        )
        final_state = dict(localization.final_state)
        raw_observation = raw[0] if isinstance(raw, tuple) else raw
        if isinstance(raw_observation, Mapping) and "state_timestamp_ns" in raw_observation:
            # This source timestamp belongs to the wheel/state response.  It
            # is retained separately from the localization timestamp and is
            # never used as pose evidence or relabelled into the monotonic
            # timestamp domain required by the controller.
            final_state["velocity_source_timestamp_ns"] = raw_observation["state_timestamp_ns"]
        final_state["velocity_feedback_received_at_s"] = now
        return Feedback(
            pose=localization.pose,
            velocity=velocity,
            observed_at_s=float(localization.observed_at_s),
            collision=localization.collision,
            contact_events=localization.contact_events,
            final_state=final_state,
        )

    def set_body_velocity(self, velocity: BodyVelocity) -> None:
        if self.closed:
            raise XLeRobotTeleopError("XLeRobotTeleopBackend is closed")
        if not isinstance(velocity, BodyVelocity):
            raise TypeError("velocity must be a BodyVelocity")
        if abs(float(velocity.vy)) > self.lateral_tolerance_m_s:
            raise XLeRobotTeleopCommandError("the inspected two-wheel XLeRobot API has no lateral velocity command")
        if abs(float(velocity.vx)) > self.max_linear_velocity_m_s:
            raise XLeRobotTeleopCommandError("x.vel exceeds the configured linear bound")
        if abs(float(velocity.omega)) > self.max_angular_velocity_rad_s:
            raise XLeRobotTeleopCommandError("theta.vel exceeds the configured angular bound")
        action = {
            "x.vel": float(velocity.vx),
            "theta.vel": math.degrees(float(velocity.omega)),
        }
        try:
            result = self.robot.command(action)
        except BaseException as command_error:
            try:
                self.stop(reason="command-exception")
            except Exception as stop_error:
                raise command_error from stop_error
            raise
        if not isinstance(result, Mapping):
            error = XLeRobotTeleopCommandError("XLeRobot command returned no acceptance record")
            try:
                self.stop(reason="command-unknown")
            except Exception as stop_error:
                raise error from stop_error
            raise error
        errors = result.get("errors", ())
        accepted = (
            result.get("accepted") is True
            and result.get("command_accepted") is True
            and isinstance(errors, (list, tuple))
            and not errors
        )
        if not accepted:
            detail = "; ".join(str(item) for item in errors) if isinstance(errors, (list, tuple)) else str(errors)
            error = XLeRobotTeleopCommandError("XLeRobot command was not accepted" + (f": {detail}" if detail else ""))
            try:
                self.stop(reason="command-rejected")
            except Exception as stop_error:
                raise error from stop_error
            raise error

    def stop(self, *, reason: str = "stop") -> Any:
        del reason
        if self.closed:
            return self.last_stop_report
        # Existing RemoteRobot.stop() is the ownership/watchdog-aware stop
        # route.  Do not bypass it by writing a synthetic motor command.
        try:
            result = self.robot.stop()
        except BaseException:
            self.last_stop_report = {
                "stop_confirmed": False,
                "stationary_confirmed": False,
                "physical_outcome": "unknown",
            }
            raise
        if isinstance(result, Mapping):
            self.last_stop_report = dict(result)
        else:
            self.last_stop_report = {
                "stop_confirmed": False,
                "stationary_confirmed": False,
                "physical_outcome": "unknown",
                "raw_result": result,
            }
        if not (
            self.last_stop_report.get("stop_confirmed") is True
            and self.last_stop_report.get("stationary_confirmed") is True
            and self.last_stop_report.get("physical_outcome") == "stopped"
        ):
            errors = self.last_stop_report.get("errors", ())
            detail = "; ".join(str(item) for item in errors) if isinstance(errors, (list, tuple)) else str(errors)
            raise XLeRobotStopUnconfirmed("XLeRobot stop was not confirmed" + (f": {detail}" if detail else ""))
        return self.last_stop_report

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.stop(reason="close")
        finally:
            self.closed = True
            self.robot.close()


def _local_segment_localizer_unavailable(_observation: Any) -> TeleopLocalization:
    raise LocalizationUnavailable(
        "local-segment mode does not provide metric localization; use its wheel feedback boundary"
    )


class XLeRobotTeleopLocalSegmentBackend:
    """Command boundary for local-segment mode without a world pose.

    The command and stop methods delegate to :class:`XLeRobotTeleopBackend`,
    so command acceptance, control ownership, watchdog behaviour, and stop
    confirmation remain the inspected RemoteRobot route.  Feedback is only
    wheel state: the source must explicitly report ``state_timestamp_ns`` and
    ``state_cached=False``.  A local receive timestamp is kept separately and
    is never presented as a camera capture timestamp or metric pose.
    """

    def __init__(
        self,
        robot: XLeRobotTeleopRobot,
        *,
        max_linear_velocity_m_s: float = 0.3,
        max_angular_velocity_rad_s: float = 0.6,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Reuse the existing concrete command/stop validation without invoking
        # its localization callback during construction.
        self._command_boundary = XLeRobotTeleopBackend(
            robot,
            _local_segment_localizer_unavailable,
            max_linear_velocity_m_s=max_linear_velocity_m_s,
            max_angular_velocity_rad_s=max_angular_velocity_rad_s,
            clock=clock,
        )
        self.robot = robot
        self._clock = clock
        self.clock = clock
        self.closed = False
        self.last_stop_report: Mapping[str, Any] | None = None

    def feedback(self) -> LocalSegmentFeedback:
        if self.closed:
            raise XLeRobotTeleopError("XLeRobotTeleopLocalSegmentBackend is closed")
        raw = self.robot.read()
        raw_observation = raw[0] if isinstance(raw, tuple) else raw
        state = _observation_state(raw)
        if not isinstance(raw_observation, Mapping):
            raise XLeRobotTeleopError("XLeRobot observation must be a mapping")
        if raw_observation.get("state_cached") is not False:
            raise XLeRobotFeedbackCached("wheel feedback freshness is unknown or cached; require state_cached=False")
        if raw_observation.get("armed") is not True or raw_observation.get("control_owned") is not True:
            raise LocalSegmentLeaseLost("robot no longer reports an active armed control lease")
        timestamp = raw_observation.get("state_timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            raise XLeRobotTeleopError("wheel feedback requires a positive state_timestamp_ns")
        errors = raw_observation.get("errors")
        if errors:
            raise XLeRobotTeleopError(f"wheel feedback reports errors: {errors!r}")
        collision = raw_observation.get("collision")
        if collision is not None and not isinstance(collision, bool):
            raise XLeRobotTeleopError("collision feedback must be boolean when supplied")
        now = float(self._clock())
        final_state = {
            "state_timestamp_ns": timestamp,
            "state_cached": False,
            "feedback_received_at_s": now,
            "feedback_time_basis": "local-receive",
            "metric_pose_available": False,
            "camera_freshness_checked_here": False,
        }
        metadata = raw_observation.get("metadata")
        if isinstance(metadata, Mapping):
            final_state["robot_metadata"] = dict(metadata)
        return LocalSegmentFeedback(
            velocity=BodyVelocity(
                _number(state, "x.vel"),
                0.0,
                math.radians(_number(state, "theta.vel")),
            ),
            received_at_s=now,
            state_timestamp_ns=timestamp,
            collision=collision,
            final_state=final_state,
        )

    def hold_zero(
        self,
        *,
        timeout_s: float = 0.5,
        poll_s: float = 0.025,
        linear_tolerance_m_s: float = 0.01,
        angular_tolerance_rad_s: float = 0.02,
    ) -> LocalSegmentFeedback:
        """Keep the control lease while commanding zero and polling fresh state."""

        for name, value in (
            ("timeout_s", timeout_s),
            ("poll_s", poll_s),
            ("linear_tolerance_m_s", linear_tolerance_m_s),
            ("angular_tolerance_rad_s", angular_tolerance_rad_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or (name == "timeout_s" and float(value) <= 0.0)
                or (name == "poll_s" and float(value) <= 0.0)
            ):
                raise ValueError(f"{name} must be finite and non-negative (positive for timeouts/poll)")
        self._command_boundary.set_body_velocity(BodyVelocity.zero())
        deadline = time.monotonic() + float(timeout_s)
        last_feedback: LocalSegmentFeedback | None = None
        while True:
            last_feedback = self.fresh_feedback(timeout_s=timeout_s, poll_s=poll_s)
            if abs(float(last_feedback.velocity.vx)) <= float(linear_tolerance_m_s) and abs(
                float(last_feedback.velocity.omega)
            ) <= float(angular_tolerance_rad_s):
                return last_feedback
            if time.monotonic() >= deadline:
                raise XLeRobotTeleopError("zero command did not receive fresh measured stationary feedback")
            time.sleep(float(poll_s))

    def fresh_feedback(
        self,
        *,
        timeout_s: float = 0.5,
        poll_s: float = 0.025,
    ) -> LocalSegmentFeedback:
        """Wait briefly through cached reads for one fresh owned state sample."""

        deadline = time.monotonic() + float(timeout_s)
        while True:
            try:
                return self.feedback()
            except XLeRobotFeedbackCached:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(float(poll_s))

    def set_body_velocity(self, velocity: BodyVelocity) -> None:
        self._command_boundary.set_body_velocity(velocity)

    def stop(self, *, reason: str = "stop") -> Any:
        result = self._command_boundary.stop(reason=reason)
        self.last_stop_report = self._command_boundary.last_stop_report
        return result

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._command_boundary.close()
        finally:
            self.last_stop_report = self._command_boundary.last_stop_report
            self.closed = True
