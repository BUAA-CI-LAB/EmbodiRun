"""Bounded LightNav-0 waypoint tracking for an XLeRobot base.

LightNav-0 returns cumulative future poses in the frame of the image that was
captured for inference.  This module keeps that frame explicit:

* model rows are ``[forward_m, lateral_m, yaw_rad]`` (left is positive),
* the rows are projected into world coordinates using the capture pose, and
* the current measured pose is transformed back into its own body frame before
  a bounded omnidirectional velocity is produced.

Only the tracker and backend protocol live here.  No real XLeRobot SDK is
imported and no socket or motor is opened by construction or import.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_LOGGER = logging.getLogger(__name__)

MAX_WAYPOINTS = 10


class LightNav0Error(RuntimeError):
    """Base error for the XLeRobot LightNav-0 control boundary."""


class LightNav0OutputError(ValueError):
    """A model output is not a valid LightNav-0 waypoint chunk."""


def wrap_angle(angle: float) -> float:
    """Return ``angle`` in ``[-pi, pi]`` without introducing a numpy dependency."""

    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LightNav0OutputError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise LightNav0OutputError(f"{name} must be finite")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LightNav0OutputError(f"{name} must be a sequence")
    return value


def _triple(value: Any, name: str, error_type: type[Exception] = LightNav0OutputError) -> tuple[float, float, float]:
    try:
        values = _sequence(value, name)
    except LightNav0OutputError as exc:
        if error_type is LightNav0OutputError:
            raise
        raise error_type(str(exc)) from exc
    if len(values) != 3:
        raise error_type(f"{name} must contain exactly three values")
    try:
        return tuple(_finite(item, f"{name}[{i}]") for i, item in enumerate(values))  # type: ignore[return-value]
    except LightNav0OutputError as exc:
        raise error_type(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Measured world pose, with yaw positive counter-clockwise."""

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("yaw", self.yaw)):
            _finite(value, f"pose.{name}")

    @classmethod
    def from_value(cls, value: Any, *, name: str = "pose") -> Pose2D:
        if isinstance(value, cls):
            return value
        return cls(*_triple(value, name, TypeError))

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw


@dataclass(frozen=True, slots=True)
class BodyVelocity:
    """XLeRobot body velocity: forward, left, and CCW yaw rate."""

    vx: float
    vy: float
    omega: float

    def __post_init__(self) -> None:
        for name, value in (("vx", self.vx), ("vy", self.vy), ("omega", self.omega)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"velocity.{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"velocity.{name} must be finite")

    @classmethod
    def zero(cls) -> BodyVelocity:
        return cls(0.0, 0.0, 0.0)

    def as_tuple(self) -> tuple[float, float, float]:
        return float(self.vx), float(self.vy), float(self.omega)


@dataclass(frozen=True, slots=True)
class LightNav0Waypoint:
    """One cumulative robot-local future pose."""

    forward_m: float
    lateral_m: float
    yaw_rad: float

    @classmethod
    def from_value(cls, value: Any, *, name: str = "waypoint") -> LightNav0Waypoint:
        return cls(*_triple(value, name, LightNav0OutputError))

    def as_tuple(self) -> tuple[float, float, float]:
        return self.forward_m, self.lateral_m, self.yaw_rad


@dataclass(frozen=True, slots=True)
class LightNav0Trajectory:
    """Validated cumulative waypoints anchored to one observation pose."""

    waypoints: tuple[LightNav0Waypoint, ...]
    stop: bool
    capture_pose: Pose2D
    captured_at_s: float

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise LightNav0OutputError("waypoints must contain at least one row")
        if len(self.waypoints) > MAX_WAYPOINTS:
            raise LightNav0OutputError(f"waypoints must contain at most {MAX_WAYPOINTS} rows")
        if not isinstance(self.stop, bool):
            raise LightNav0OutputError("stop must be boolean")
        if not isinstance(self.capture_pose, Pose2D):
            raise TypeError("capture_pose must be a Pose2D")
        _finite(self.captured_at_s, "captured_at_s")

    @classmethod
    def from_output(
        cls,
        output: Mapping[str, Any],
        *,
        capture_pose: Pose2D | Sequence[float],
        captured_at_s: float,
    ) -> LightNav0Trajectory:
        return parse_lightnav0_output(
            output,
            capture_pose=capture_pose,
            captured_at_s=captured_at_s,
        )


def parse_lightnav0_output(
    output: Mapping[str, Any],
    *,
    capture_pose: Pose2D | Sequence[float],
    captured_at_s: float,
) -> LightNav0Trajectory:
    """Validate the official ``waypoints`` + boolean ``stop`` model output.

    A chunk may contain fewer than ten rows during a transport-level truncation
    only if the caller explicitly chooses to accept it; this parser keeps the
    physical boundary simple and accepts one to ten finite ``[3]`` rows.  It
    rejects missing fields, malformed shapes, non-finite values, and an
    ambiguous non-boolean stop value.
    """

    if not isinstance(output, Mapping):
        raise LightNav0OutputError("LightNav-0 output must be an object")
    if "waypoints" not in output:
        raise LightNav0OutputError("LightNav-0 output is missing waypoints")
    if "stop" not in output:
        raise LightNav0OutputError("LightNav-0 output is missing stop")
    stop = output["stop"]
    if not isinstance(stop, bool):
        raise LightNav0OutputError("stop must be boolean")
    rows = _sequence(output["waypoints"], "waypoints")
    if not rows or len(rows) > MAX_WAYPOINTS:
        raise LightNav0OutputError(f"waypoints must contain one to {MAX_WAYPOINTS} rows")
    waypoints = tuple(LightNav0Waypoint.from_value(row, name=f"waypoints[{index}]") for index, row in enumerate(rows))
    pose = Pose2D.from_value(capture_pose, name="capture_pose")
    timestamp = _finite(captured_at_s, "captured_at_s")
    return LightNav0Trajectory(waypoints, stop, pose, timestamp)


def project_body_to_world(
    waypoints: Sequence[LightNav0Waypoint | Sequence[float]],
    capture_pose: Pose2D | Sequence[float],
) -> tuple[Pose2D, ...]:
    """Project body-frame cumulative SE(2) waypoints into world coordinates."""

    origin = Pose2D.from_value(capture_pose, name="capture_pose")
    cosine, sine = math.cos(origin.yaw), math.sin(origin.yaw)
    result: list[Pose2D] = []
    for index, raw in enumerate(waypoints):
        wp = (
            raw if isinstance(raw, LightNav0Waypoint) else LightNav0Waypoint.from_value(raw, name=f"waypoints[{index}]")
        )
        result.append(
            Pose2D(
                origin.x + cosine * wp.forward_m - sine * wp.lateral_m,
                origin.y + sine * wp.forward_m + cosine * wp.lateral_m,
                wrap_angle(origin.yaw + wp.yaw_rad),
            )
        )
    return tuple(result)


def project_world_to_body(world_pose: Pose2D | Sequence[float], current_pose: Pose2D | Sequence[float]) -> Pose2D:
    """Express one world pose as forward/left/yaw error at ``current_pose``."""

    target = Pose2D.from_value(world_pose, name="world_pose")
    origin = Pose2D.from_value(current_pose, name="current_pose")
    cosine, sine = math.cos(origin.yaw), math.sin(origin.yaw)
    dx, dy = target.x - origin.x, target.y - origin.y
    return Pose2D(
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        wrap_angle(target.yaw - origin.yaw),
    )


@dataclass(frozen=True, slots=True)
class Feedback:
    """Fresh low-level backend feedback used by the tracker/controller."""

    pose: Pose2D
    velocity: BodyVelocity = field(default_factory=BodyVelocity.zero)
    observed_at_s: float = 0.0
    collision: bool = False
    contact_events: tuple[Mapping[str, Any], ...] = ()
    final_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.observed_at_s, "observed_at_s")
        if not isinstance(self.collision, bool):
            raise TypeError("collision must be boolean")
        if not isinstance(self.contact_events, tuple):
            raise TypeError("contact_events must be a tuple")
        if not isinstance(self.final_state, Mapping):
            raise TypeError("final_state must be a mapping")


@runtime_checkable
class XLeRobotBackend(Protocol):
    """Injectable feedback/command boundary for real or simulated XLeRobot."""

    def feedback(self) -> Feedback: ...

    def set_body_velocity(self, velocity: BodyVelocity) -> None: ...

    def stop(self, *, reason: str = "stop") -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Conservative tracker settings, all expressed in SI units."""

    max_linear_velocity_m_s: float = 0.3
    max_angular_velocity_rad_s: float = 0.6
    position_gain: float = 2.0
    heading_gain: float = 2.0
    waypoint_prefix: int = 3
    waypoint_tolerance_m: float = 0.05
    yaw_tolerance_rad: float = 0.08
    trajectory_timeout_s: float = 0.5
    feedback_timeout_s: float = 0.5
    drive_mode: str = "omnidirectional"

    def __post_init__(self) -> None:
        if self.drive_mode not in {"omnidirectional", "differential"}:
            raise ValueError("drive_mode must be 'omnidirectional' or 'differential'")
        for name, value in (
            ("max_linear_velocity_m_s", self.max_linear_velocity_m_s),
            ("max_angular_velocity_rad_s", self.max_angular_velocity_rad_s),
            ("position_gain", self.position_gain),
            ("heading_gain", self.heading_gain),
            ("waypoint_tolerance_m", self.waypoint_tolerance_m),
            ("yaw_tolerance_rad", self.yaw_tolerance_rad),
            ("trajectory_timeout_s", self.trajectory_timeout_s),
            ("feedback_timeout_s", self.feedback_timeout_s),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.waypoint_prefix, bool)
            or not isinstance(self.waypoint_prefix, int)
            or self.waypoint_prefix <= 0
        ):
            raise ValueError("waypoint_prefix must be a positive integer")


@dataclass(frozen=True, slots=True)
class TrackingDecision:
    """One safe command decision and the reason that produced it."""

    command: BodyVelocity
    reason: str
    target_index: int | None = None
    trajectory_stale: bool = False


class LightNav0Tracker:
    """Track a validated trajectory using fresh measured pose feedback."""

    def __init__(self, config: TrackerConfig | None = None, *, clock: Any = time.monotonic) -> None:
        self.config = config or TrackerConfig()
        self._clock = clock
        self._trajectory: LightNav0Trajectory | None = None

    @property
    def trajectory(self) -> LightNav0Trajectory | None:
        return self._trajectory

    def set_trajectory(self, trajectory: LightNav0Trajectory) -> None:
        if not isinstance(trajectory, LightNav0Trajectory):
            raise TypeError("trajectory must be a LightNav0Trajectory")
        self._trajectory = trajectory

    def clear(self) -> None:
        self._trajectory = None

    def decide(self, feedback: Feedback, *, now_s: float | None = None) -> TrackingDecision:
        if not isinstance(feedback, Feedback):
            raise TypeError("feedback must be a Feedback")
        now = float(self._clock() if now_s is None else now_s)
        _finite(now, "now_s")
        age = now - feedback.observed_at_s
        if age < -self.config.feedback_timeout_s:
            return TrackingDecision(BodyVelocity.zero(), "feedback-from-future")
        if age > self.config.feedback_timeout_s:
            return TrackingDecision(BodyVelocity.zero(), "stale-feedback", trajectory_stale=True)
        if feedback.collision:
            return TrackingDecision(BodyVelocity.zero(), "collision")
        trajectory = self._trajectory
        if trajectory is None:
            return TrackingDecision(BodyVelocity.zero(), "no-trajectory")
        trajectory_age = now - trajectory.captured_at_s
        if trajectory_age < -self.config.trajectory_timeout_s:
            return TrackingDecision(BodyVelocity.zero(), "trajectory-from-future", trajectory_stale=True)
        if trajectory_age > self.config.trajectory_timeout_s:
            self._trajectory = None
            return TrackingDecision(BodyVelocity.zero(), "stale-trajectory", trajectory_stale=True)
        if trajectory.stop:
            self._trajectory = None
            return TrackingDecision(BodyVelocity.zero(), "explicit-stop")

        world_points = project_body_to_world(trajectory.waypoints, trajectory.capture_pose)
        prefix = world_points[: self.config.waypoint_prefix]
        target_index, target = self._select_target(prefix, feedback.pose)
        error = project_world_to_body(target, feedback.pose)
        if self.config.drive_mode == "omnidirectional":
            vx = self.config.position_gain * error.x
            vy = self.config.position_gain * error.y
            norm = math.hypot(vx, vy)
            if norm > self.config.max_linear_velocity_m_s:
                scale = self.config.max_linear_velocity_m_s / norm
                vx *= scale
                vy *= scale
            omega_error = error.yaw
        else:
            # A two-wheel base cannot command lateral velocity.  Turn toward
            # the selected waypoint and use only its forward component; once
            # the position is reached, align to the waypoint's yaw in place.
            distance = math.hypot(error.x, error.y)
            if distance > self.config.waypoint_tolerance_m:
                vx = self.config.position_gain * error.x
                omega_error = math.atan2(error.y, error.x)
            else:
                vx = 0.0
                omega_error = error.yaw
            vy = 0.0
        omega = self.config.heading_gain * omega_error
        if self.config.drive_mode == "differential":
            vx = max(
                -self.config.max_linear_velocity_m_s,
                min(self.config.max_linear_velocity_m_s, vx),
            )
        omega = max(
            -self.config.max_angular_velocity_rad_s,
            min(self.config.max_angular_velocity_rad_s, omega),
        )
        return TrackingDecision(BodyVelocity(vx, vy, omega), "tracking", target_index)

    def _select_target(self, points: Sequence[Pose2D], current: Pose2D) -> tuple[int, Pose2D]:
        for index, point in enumerate(points):
            error = project_world_to_body(point, current)
            if (
                math.hypot(error.x, error.y) > self.config.waypoint_tolerance_m
                or abs(error.yaw) > self.config.yaw_tolerance_rad
            ):
                return index, point
        return len(points) - 1, points[-1]


@dataclass(frozen=True, slots=True)
class ControllerStep:
    feedback: Feedback
    decision: TrackingDecision


class LightNav0Controller:
    """Apply tracker decisions to an injected XLeRobot backend.

    ``submit`` is called with the pose and timestamp captured with the model
    image.  ``tick`` obtains fresh feedback for each control update.  Backend
    exceptions trigger a best-effort stop before the original exception is
    re-raised; this keeps cleanup at the control boundary without hiding a
    hardware/transport failure.
    """

    def __init__(
        self,
        backend: XLeRobotBackend,
        tracker: LightNav0Tracker | None = None,
        *,
        clock: Any = None,
    ) -> None:
        if not isinstance(backend, XLeRobotBackend):
            raise TypeError("backend must implement XLeRobotBackend")
        self.backend = backend
        backend_clock = getattr(backend, "clock", None)
        if not callable(backend_clock):
            backend_clock = getattr(backend, "_clock", None)
        self._clock = clock if callable(clock) else (backend_clock or time.monotonic)
        self.tracker = tracker or LightNav0Tracker(clock=self._clock)
        self._closed = False

    def submit(
        self,
        output: Mapping[str, Any] | LightNav0Trajectory,
        *,
        capture_pose: Pose2D | Sequence[float] | None = None,
        captured_at_s: float | None = None,
    ) -> LightNav0Trajectory:
        if self._closed:
            raise RuntimeError("LightNav0Controller is closed")
        try:
            if isinstance(output, LightNav0Trajectory):
                trajectory = output
            else:
                if capture_pose is None:
                    raise ValueError("capture_pose is required for model output")
                trajectory = parse_lightnav0_output(
                    output,
                    capture_pose=capture_pose,
                    captured_at_s=self._clock() if captured_at_s is None else captured_at_s,
                )
        except BaseException:
            self._safe_stop(reason="invalid-model-output")
            raise
        self.tracker.set_trajectory(trajectory)
        return trajectory

    def tick(self, *, now_s: float | None = None) -> ControllerStep:
        if self._closed:
            raise RuntimeError("LightNav0Controller is closed")
        try:
            feedback = self.backend.feedback()
            decision = self.tracker.decide(feedback, now_s=now_s)
            if decision.reason in {
                "explicit-stop",
                "stale-trajectory",
                "stale-feedback",
                "feedback-from-future",
                "trajectory-from-future",
                "collision",
            }:
                self._safe_stop(reason=decision.reason)
            else:
                self.backend.set_body_velocity(decision.command)
            return ControllerStep(feedback, decision)
        except BaseException:
            self._safe_stop(reason="controller-exception")
            raise

    def stop(self, *, reason: str = "stop") -> Any:
        self.tracker.clear()
        return self.backend.stop(reason=reason)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop(reason="close")
        finally:
            self._closed = True
            self.backend.close()

    def _safe_stop(self, *, reason: str) -> None:
        try:
            self.backend.stop(reason=reason)
        except Exception as error:
            # Preserve the first transport/control exception.  The backend's
            # own stop report remains available to the caller when supported.
            _LOGGER.debug("best-effort LightNav-0 stop failed", exc_info=error)
