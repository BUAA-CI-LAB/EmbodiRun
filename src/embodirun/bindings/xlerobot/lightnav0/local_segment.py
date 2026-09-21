"""Short local waypoint segments for a two-wheel XLeRobot base.

This module is intentionally independent from :mod:`lightnav0` pose tracking.
It consumes the first cumulative local waypoint that has a usable forward or
yaw component, turns it into one bounded ``BodyVelocity`` command for a short
explicit interval, and relies on the next RGB observation for the next model
decision.  It never creates a world pose or integrates odometry.

The feedback contract is deliberately smaller than ``Feedback``: a local
segment needs fresh wheel feedback and a receive timestamp, not metric pose.
The transport adapter must still provide a real state timestamp and must reject
cached or duplicate state before returning this value.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .tracker import BodyVelocity, LightNav0Waypoint, _finite, _sequence

_LOGGER = logging.getLogger(__name__)


class LocalSegmentError(RuntimeError):
    """Base error for local segment execution."""


class LocalSegmentOutputError(ValueError):
    """The model output cannot be used as a local segment."""


class LocalSegmentFeedbackError(LocalSegmentError):
    """Wheel feedback is absent, stale, cached, or otherwise unknown."""


class LocalSegmentFeedbackStale(LocalSegmentFeedbackError):
    """Wheel feedback is outside the receive-time freshness bound."""


class LocalSegmentLeaseLost(LocalSegmentFeedbackError):
    """The robot no longer reports this client's active control ownership."""


class LocalSegmentNotStationary(LocalSegmentFeedbackError):
    """A zero command did not produce fresh measured stationary feedback."""


class LocalSegmentCollision(LocalSegmentError):
    """The injected feedback reports a collision during local execution."""


@dataclass(frozen=True, slots=True)
class LocalSegmentTrajectory:
    """Validated cumulative local waypoints without a world-pose anchor."""

    waypoints: tuple[LightNav0Waypoint, ...]
    stop: bool

    def __post_init__(self) -> None:
        if not self.waypoints or len(self.waypoints) > 10:
            raise LocalSegmentOutputError("waypoints must contain one to ten rows")
        if not isinstance(self.stop, bool):
            raise LocalSegmentOutputError("stop must be boolean")

    @classmethod
    def from_output(cls, output: Mapping[str, Any]) -> LocalSegmentTrajectory:
        if not isinstance(output, Mapping):
            raise LocalSegmentOutputError("LightNav-0 output must be an object")
        if "waypoints" not in output:
            raise LocalSegmentOutputError("LightNav-0 output is missing waypoints")
        if "stop" not in output:
            raise LocalSegmentOutputError("LightNav-0 output is missing stop")
        if not isinstance(output["stop"], bool):
            raise LocalSegmentOutputError("stop must be boolean")
        try:
            rows = _sequence(output["waypoints"], "waypoints")
        except ValueError as exc:
            raise LocalSegmentOutputError(str(exc)) from exc
        if not rows or len(rows) > 10:
            raise LocalSegmentOutputError("waypoints must contain one to ten rows")
        waypoints: list[LightNav0Waypoint] = []
        for index, row in enumerate(rows):
            try:
                waypoints.append(LightNav0Waypoint.from_value(row, name=f"waypoints[{index}]"))
            except (TypeError, ValueError) as exc:
                raise LocalSegmentOutputError(str(exc)) from exc
        return cls(tuple(waypoints), output["stop"])


@dataclass(frozen=True, slots=True)
class LocalSegmentFeedback:
    """Fresh wheel feedback; no metric pose is implied by this type."""

    velocity: BodyVelocity
    received_at_s: float
    state_timestamp_ns: int
    collision: bool | None = None
    contact_events: tuple[Mapping[str, Any], ...] = ()
    final_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.velocity, BodyVelocity):
            raise TypeError("velocity must be BodyVelocity")
        _finite(self.received_at_s, "received_at_s")
        if isinstance(self.state_timestamp_ns, bool) or not isinstance(self.state_timestamp_ns, int):
            raise TypeError("state_timestamp_ns must be an integer")
        if self.state_timestamp_ns <= 0:
            raise ValueError("state_timestamp_ns must be positive")
        if self.collision is not None and not isinstance(self.collision, bool):
            raise TypeError("collision must be boolean or None")
        if not isinstance(self.contact_events, tuple):
            raise TypeError("contact_events must be a tuple")
        if not isinstance(self.final_state, Mapping):
            raise TypeError("final_state must be a mapping")


@runtime_checkable
class LocalSegmentBackend(Protocol):
    """Injectable command and wheel-feedback boundary."""

    def feedback(self) -> LocalSegmentFeedback: ...

    def set_body_velocity(self, velocity: BodyVelocity) -> None: ...

    def stop(self, *, reason: str = "stop") -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalSegmentConfig:
    """Conservative local segment and freshness limits, in SI units."""

    duration_s: float = 0.25
    max_linear_velocity_m_s: float = 0.3
    max_angular_velocity_rad_s: float = 0.6
    waypoint_zero_tolerance: float = 1e-6
    feedback_timeout_s: float = 0.5
    stationary_linear_tolerance_m_s: float = 0.01
    stationary_angular_tolerance_rad_s: float = 0.02
    settle_timeout_s: float = 0.5
    settle_poll_s: float = 0.025

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "feedback_timeout_s",
            "stationary_linear_tolerance_m_s",
            "stationary_angular_tolerance_rad_s",
            "settle_timeout_s",
            "settle_poll_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if isinstance(self.waypoint_zero_tolerance, bool) or not isinstance(self.waypoint_zero_tolerance, (int, float)):
            raise TypeError("waypoint_zero_tolerance must be finite and non-negative")
        if not math.isfinite(float(self.waypoint_zero_tolerance)) or float(self.waypoint_zero_tolerance) < 0.0:
            raise ValueError("waypoint_zero_tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LocalSegmentStep:
    """Result of one local segment command."""

    feedback: LocalSegmentFeedback
    trajectory: LocalSegmentTrajectory
    waypoint_index: int
    command: BodyVelocity
    reason: str


def select_unicycle_waypoint(
    waypoints: Sequence[LightNav0Waypoint], *, tolerance: float = 1e-6
) -> tuple[int, LightNav0Waypoint]:
    """Select the first waypoint that can drive a unicycle.

    A waypoint with only lateral displacement cannot be represented by the
    configured two-wheel base, so leading such rows are skipped.  If every row
    is lateral-only or zero, row zero is returned and produces a zero command.
    """

    if not waypoints:
        raise LocalSegmentOutputError("waypoints must not be empty")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, LightNav0Waypoint):
            raise TypeError("waypoints must contain LightNav0Waypoint values")
        if abs(waypoint.forward_m) > tolerance or abs(waypoint.yaw_rad) > tolerance:
            return index, waypoint
    return 0, waypoints[0]


def waypoint_to_body_velocity(
    waypoint: LightNav0Waypoint,
    *,
    duration_s: float,
    max_linear_velocity_m_s: float,
    max_angular_velocity_rad_s: float,
) -> BodyVelocity:
    """Convert one local waypoint to bounded forward/yaw velocity.

    The waypoint-derived velocities are scaled together when either component
    exceeds its XLeRobot limit.  A per-component clamp would change the
    curvature of a mixed translation/rotation command, so the common scale
    preserves the waypoint's direction in ``(vx, omega)`` space while still
    enforcing both bounds.
    """

    if not isinstance(waypoint, LightNav0Waypoint):
        raise TypeError("waypoint must be LightNav0Waypoint")
    for name, value in (
        ("duration_s", duration_s),
        ("max_linear_velocity_m_s", max_linear_velocity_m_s),
        ("max_angular_velocity_rad_s", max_angular_velocity_rad_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    # Work in displacement/time units without first forming ``waypoint /
    # duration``.  A valid finite waypoint paired with a subnormal positive
    # duration can overflow that raw division even though the bounded command
    # is representable.  The common effective duration preserves the same
    # velocity ratio as a common scale factor.
    linear_duration = abs(waypoint.forward_m) / float(max_linear_velocity_m_s)
    angular_duration = abs(waypoint.yaw_rad) / float(max_angular_velocity_rad_s)
    effective_duration = max(float(duration_s), linear_duration, angular_duration)
    if not math.isfinite(effective_duration):
        raise ValueError("waypoint requires a non-finite effective duration for bounded velocity")
    vx = waypoint.forward_m / effective_duration
    omega = waypoint.yaw_rad / effective_duration
    if not math.isfinite(vx) or not math.isfinite(omega):
        raise ValueError("bounded waypoint velocity is not finite")
    return BodyVelocity(vx, 0.0, omega)


class LocalSegmentExecutor:
    """Execute one bounded command between two RGB model decisions."""

    def __init__(
        self,
        backend: LocalSegmentBackend,
        config: LocalSegmentConfig | None = None,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        if not isinstance(backend, LocalSegmentBackend):
            raise TypeError("backend must implement LocalSegmentBackend")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.backend = backend
        self.config = config or LocalSegmentConfig()
        self._clock = clock
        self._closed = False
        self._last_state_timestamp_ns: int | None = None

    def execute(
        self,
        output: Mapping[str, Any] | LocalSegmentTrajectory,
        *,
        now_s: float | None = None,
        feedback: LocalSegmentFeedback | None = None,
    ) -> LocalSegmentStep:
        if self._closed:
            raise LocalSegmentError("LocalSegmentExecutor is closed")
        try:
            trajectory = (
                output if isinstance(output, LocalSegmentTrajectory) else LocalSegmentTrajectory.from_output(output)
            )
            reused_feedback = feedback is not None
            if feedback is None:
                feedback = self.backend.feedback()
            now = float(self._clock() if now_s is None else now_s)
            self._check_feedback(feedback, now, allow_same_timestamp=reused_feedback)
            if feedback.collision is True:
                self._safe_stop(reason="collision")
                raise LocalSegmentCollision("local segment feedback reports collision")
            if trajectory.stop:
                self.backend.stop(reason="explicit-stop")
                return LocalSegmentStep(feedback, trajectory, 0, BodyVelocity.zero(), "explicit-stop")
            index, waypoint = select_unicycle_waypoint(
                trajectory.waypoints, tolerance=self.config.waypoint_zero_tolerance
            )
            command = waypoint_to_body_velocity(
                waypoint,
                duration_s=self.config.duration_s,
                max_linear_velocity_m_s=self.config.max_linear_velocity_m_s,
                max_angular_velocity_rad_s=self.config.max_angular_velocity_rad_s,
            )
            self.backend.set_body_velocity(command)
            return LocalSegmentStep(feedback, trajectory, index, command, "tracking")
        except BaseException:
            self._safe_stop(reason="local-segment-exception")
            raise

    def check_fresh_feedback(self, *, now_s: float | None = None) -> LocalSegmentFeedback:
        """Check lease, collision and fresh feedback without issuing motion."""

        if self._closed:
            raise LocalSegmentError("LocalSegmentExecutor is closed")
        try:
            fresh_feedback = getattr(self.backend, "fresh_feedback", None)
            feedback = (
                fresh_feedback(
                    timeout_s=self.config.settle_timeout_s,
                    poll_s=self.config.settle_poll_s,
                )
                if callable(fresh_feedback)
                else self.backend.feedback()
            )
            now = float(self._clock() if now_s is None else now_s)
            self._check_feedback(feedback, now)
            if feedback.collision is True:
                raise LocalSegmentCollision("local segment feedback reports collision")
            return feedback
        except BaseException:
            self._safe_stop(reason="feedback-exception")
            raise

    def hold_zero(self, *, now_s: float | None = None) -> LocalSegmentFeedback:
        """Command zero through the active lease and verify measured stationarity."""

        if self._closed:
            raise LocalSegmentError("LocalSegmentExecutor is closed")
        try:
            hold_zero = getattr(self.backend, "hold_zero", None)
            if callable(hold_zero):
                feedback = hold_zero(
                    timeout_s=self.config.settle_timeout_s,
                    poll_s=self.config.settle_poll_s,
                    linear_tolerance_m_s=self.config.stationary_linear_tolerance_m_s,
                    angular_tolerance_rad_s=self.config.stationary_angular_tolerance_rad_s,
                )
            else:
                self.backend.set_body_velocity(BodyVelocity.zero())
                feedback = self.backend.feedback()
            now = float(self._clock() if now_s is None else now_s)
            self._check_feedback(feedback, now)
            if feedback.collision is True:
                raise LocalSegmentCollision("local segment feedback reports collision")
            if (
                abs(float(feedback.velocity.vx)) > self.config.stationary_linear_tolerance_m_s
                or abs(float(feedback.velocity.omega)) > self.config.stationary_angular_tolerance_rad_s
            ):
                raise LocalSegmentNotStationary("zero command lacks fresh measured stationary feedback")
            return feedback
        except BaseException:
            self._safe_stop(reason="hold-zero-exception")
            raise

    def stop(self, *, reason: str = "stop") -> Any:
        return self.backend.stop(reason=reason)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.backend.close()
        finally:
            self._closed = True

    def _check_feedback(
        self,
        feedback: LocalSegmentFeedback,
        now_s: float,
        *,
        allow_same_timestamp: bool = False,
    ) -> None:
        if not isinstance(feedback, LocalSegmentFeedback):
            raise LocalSegmentFeedbackError("backend returned unknown feedback type")
        age = float(now_s) - feedback.received_at_s
        if not math.isfinite(age) or age < -self.config.feedback_timeout_s or age > self.config.feedback_timeout_s:
            raise LocalSegmentFeedbackStale(
                f"wheel feedback age {age:.3f}s exceeds {self.config.feedback_timeout_s:.3f}s"
            )
        if self._last_state_timestamp_ns is not None and (
            feedback.state_timestamp_ns < self._last_state_timestamp_ns
            or (feedback.state_timestamp_ns == self._last_state_timestamp_ns and not allow_same_timestamp)
        ):
            raise LocalSegmentFeedbackStale("wheel feedback timestamp repeated or moved backwards")
        self._last_state_timestamp_ns = feedback.state_timestamp_ns

    @property
    def last_stop_report(self) -> Any:
        """Return the concrete stop acknowledgement retained by the backend."""

        return getattr(self.backend, "last_stop_report", None)

    def _safe_stop(self, *, reason: str) -> None:
        try:
            self.backend.stop(reason=reason)
        except Exception as error:
            # Preserve the original validation/feedback/command exception.  The
            # backend retains the concrete stop report for the caller.
            _LOGGER.debug("best-effort local segment stop failed", exc_info=error)
