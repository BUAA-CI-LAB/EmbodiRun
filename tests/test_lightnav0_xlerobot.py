import math

import pytest

from embodirun.bindings.xlerobot.lightnav0 import (
    BodyVelocity,
    Feedback,
    LightNav0Controller,
    LightNav0OutputError,
    LightNav0Tracker,
    LightNav0Trajectory,
    Pose2D,
    TeleopLocalization,
    TrackerConfig,
    XLeRobotStopUnconfirmed,
    XLeRobotTeleopBackend,
    XLeRobotTeleopCommandError,
    XLeRobotTeleopError,
    project_body_to_world,
    project_world_to_body,
)


def _trajectory(*, stop=False, pose=(1.0, 2.0, math.pi / 2.0), timestamp=0.0, rows=None):
    return LightNav0Trajectory.from_output(
        {"waypoints": rows or [[1.0, 0.0, 0.0]], "stop": stop},
        capture_pose=pose,
        captured_at_s=timestamp,
    )


def _feedback(*, pose=(1.0, 2.0, math.pi / 2.0), timestamp=0.0, collision=False):
    return Feedback(Pose2D(*pose), BodyVelocity.zero(), timestamp, collision)


def test_capture_pose_transform_uses_forward_and_left_convention():
    world = project_body_to_world([[1.0, 0.5, math.pi / 2.0]], Pose2D(1.0, 2.0, math.pi / 2.0))[0]
    assert world.x == pytest.approx(0.5)
    assert world.y == pytest.approx(3.0)
    assert world.yaw == pytest.approx(math.pi)
    body = project_world_to_body(world, Pose2D(1.0, 2.0, math.pi / 2.0))
    assert body.as_tuple() == pytest.approx((1.0, 0.5, math.pi / 2.0))


def test_tracker_saturates_omnidirectional_linear_and_angular_velocity():
    tracker = LightNav0Tracker(
        TrackerConfig(max_linear_velocity_m_s=1.0, max_angular_velocity_rad_s=0.5),
        clock=lambda: 0.0,
    )
    tracker.set_trajectory(_trajectory(rows=[[10.0, 10.0, 10.0]]))
    decision = tracker.decide(_feedback(), now_s=0.0)
    assert math.hypot(decision.command.vx, decision.command.vy) == pytest.approx(1.0)
    assert abs(decision.command.omega) == pytest.approx(0.5)
    assert decision.reason == "tracking"


def test_differential_tracker_turns_for_lateral_target_without_vy():
    tracker = LightNav0Tracker(
        TrackerConfig(
            drive_mode="differential",
            max_linear_velocity_m_s=1.0,
            max_angular_velocity_rad_s=0.5,
        ),
        clock=lambda: 0.0,
    )
    tracker.set_trajectory(_trajectory(rows=[[0.0, 1.0, 0.0]]))
    decision = tracker.decide(_feedback(pose=(0.0, 0.0, 0.0)), now_s=0.0)
    assert decision.command.vx == pytest.approx(0.0)
    assert decision.command.vy == pytest.approx(0.0)
    assert decision.command.omega == pytest.approx(0.5)


def test_differential_tracker_uses_forward_component_and_caps_it():
    tracker = LightNav0Tracker(
        TrackerConfig(
            drive_mode="differential",
            max_linear_velocity_m_s=0.3,
            max_angular_velocity_rad_s=0.6,
        ),
        clock=lambda: 0.0,
    )
    tracker.set_trajectory(_trajectory(pose=(0.0, 0.0, 0.0), rows=[[10.0, 10.0, 0.0]]))
    decision = tracker.decide(_feedback(pose=(0.0, 0.0, 0.0)), now_s=0.0)
    assert decision.command.vy == pytest.approx(0.0)
    assert decision.command.vx == pytest.approx(0.3)


@pytest.mark.parametrize(
    "output",
    [
        {"stop": False},
        {"waypoints": [[1.0, 2.0]], "stop": False},
        {"waypoints": [[1.0, float("nan"), 0.0]], "stop": False},
        {"waypoints": [[1.0, 2.0, 3.0]], "stop": 0},
        {"waypoints": [[1.0, 2.0, 3.0]] * 11, "stop": False},
    ],
)
def test_nonfinite_and_malformed_chunks_fail_closed(output):
    with pytest.raises(LightNav0OutputError):
        LightNav0Trajectory.from_output(output, capture_pose=(0.0, 0.0, 0.0), captured_at_s=0.0)


def test_stale_trajectory_becomes_zero_and_is_cleared():
    tracker = LightNav0Tracker(TrackerConfig(trajectory_timeout_s=0.2), clock=lambda: 1.0)
    tracker.set_trajectory(_trajectory(timestamp=0.0))
    decision = tracker.decide(_feedback(timestamp=1.0), now_s=1.0)
    assert decision.command == BodyVelocity.zero()
    assert decision.reason == "stale-trajectory"
    assert tracker.trajectory is None


def test_explicit_stop_sends_zero_and_backend_stop():
    backend = _FakeBackend(_feedback())
    controller = LightNav0Controller(backend, clock=lambda: 0.0)
    controller.submit(_trajectory(stop=True))
    result = controller.tick(now_s=0.0)
    assert result.decision.command == BodyVelocity.zero()
    assert result.decision.reason == "explicit-stop"
    assert backend.stops == ["explicit-stop"]


def test_collision_stops_and_reports_reason():
    backend = _FakeBackend(_feedback(collision=True))
    controller = LightNav0Controller(backend, clock=lambda: 0.0)
    controller.submit(_trajectory())
    result = controller.tick(now_s=0.0)
    assert result.decision.command == BodyVelocity.zero()
    assert result.decision.reason == "collision"
    assert backend.stops == ["collision"]


def test_controller_stops_on_backend_exception():
    backend = _FakeBackend(_feedback(), fail_set=True)
    controller = LightNav0Controller(backend, clock=lambda: 0.0)
    controller.submit(_trajectory())
    with pytest.raises(RuntimeError, match="command failed"):
        controller.tick(now_s=0.0)
    assert backend.stops == ["controller-exception"]


class _FakeBackend:
    def __init__(self, feedback, *, fail_set=False):
        self._feedback = feedback
        self.fail_set = fail_set
        self.commands = []
        self.stops = []
        self.closed = False

    def feedback(self):
        return self._feedback

    def set_body_velocity(self, velocity):
        if self.fail_set:
            raise RuntimeError("command failed")
        self.commands.append(velocity)

    def stop(self, *, reason="stop"):
        self.stops.append(reason)

    def close(self):
        self.closed = True


class _FakeTeleopRobot:
    def __init__(self, *, state=None):
        self.state = {"x.vel": 0.12, "theta.vel": 180.0} if state is None else dict(state)
        self.commands = []
        self.stops = 0
        self.closed = False
        self.fail_command = False
        self.reject_command = False
        self.stop_confirmed = True

    def read(self):
        return {"state": dict(self.state), "images": {}}

    def command(self, action):
        if self.fail_command:
            raise RuntimeError("teleop command failed")
        self.commands.append(dict(action))
        self.state.update(action)
        if self.reject_command:
            return {
                "accepted": False,
                "command_accepted": False,
                "physical_outcome": "unknown",
                "errors": ["base is not armed"],
            }
        return {
            "accepted": True,
            "command_accepted": True,
            "physical_outcome": "unknown",
            "applied_action": dict(action),
            "errors": [],
        }

    def stop(self):
        self.stops += 1
        self.state.update({"x.vel": 0.0, "theta.vel": 0.0})
        return {
            "accepted": self.stop_confirmed,
            "command_accepted": self.stop_confirmed,
            "stationary_confirmed": self.stop_confirmed,
            "stop_confirmed": self.stop_confirmed,
            "physical_outcome": "stopped" if self.stop_confirmed else "unknown",
            "errors": [] if self.stop_confirmed else ["feedback unavailable"],
        }

    def close(self):
        self.closed = True


def _localize_fresh(_observation):
    return TeleopLocalization(Pose2D(2.0, 3.0, 0.0), observed_at_s=0.0)


def test_teleop_backend_converts_rad_per_second_to_theta_degrees():
    robot = _FakeTeleopRobot()
    backend = XLeRobotTeleopBackend(robot, _localize_fresh, clock=lambda: 0.0)
    feedback = backend.feedback()
    assert feedback.pose == Pose2D(2.0, 3.0, 0.0)
    assert feedback.velocity == BodyVelocity(0.12, 0.0, math.pi)
    backend.set_body_velocity(BodyVelocity(0.2, 0.0, 0.5))
    assert robot.commands[-1] == {
        "x.vel": pytest.approx(0.2),
        "theta.vel": pytest.approx(math.degrees(0.5)),
    }


def test_teleop_backend_rejects_lateral_velocity_and_uses_stop_route():
    robot = _FakeTeleopRobot()
    backend = XLeRobotTeleopBackend(robot, _localize_fresh, clock=lambda: 0.0)
    with pytest.raises(XLeRobotTeleopCommandError, match="lateral"):
        backend.set_body_velocity(BodyVelocity(0.0, 0.01, 0.0))
    backend.stop(reason="test")
    assert robot.stops == 1
    assert backend.last_stop_report["stop_confirmed"] is True


def test_teleop_backend_rejected_command_stops_and_raises():
    robot = _FakeTeleopRobot()
    robot.reject_command = True
    backend = XLeRobotTeleopBackend(robot, _localize_fresh, clock=lambda: 0.0)
    with pytest.raises(XLeRobotTeleopCommandError, match="not accepted"):
        backend.set_body_velocity(BodyVelocity(0.1, 0.0, 0.0))
    assert robot.stops == 1


def test_teleop_backend_does_not_claim_unknown_stop():
    robot = _FakeTeleopRobot()
    robot.stop_confirmed = False
    backend = XLeRobotTeleopBackend(robot, _localize_fresh, clock=lambda: 0.0)
    with pytest.raises(XLeRobotStopUnconfirmed, match="not confirmed"):
        backend.stop(reason="test")
    assert backend.last_stop_report["physical_outcome"] == "unknown"


def test_teleop_backend_missing_or_stale_localization_fails_and_controller_stops():
    robot = _FakeTeleopRobot()
    backend = XLeRobotTeleopBackend(robot, lambda _observation: None, clock=lambda: 0.0)
    with pytest.raises(XLeRobotTeleopError):
        backend.feedback()
    assert robot.stops == 0
    stale = XLeRobotTeleopBackend(
        robot,
        lambda _observation: TeleopLocalization(Pose2D(0.0, 0.0, 0.0), observed_at_s=-1.0),
        clock=lambda: 0.0,
    )
    controller = LightNav0Controller(stale, clock=lambda: 0.0)
    controller.submit(_trajectory())
    with pytest.raises(XLeRobotTeleopError):
        controller.tick(now_s=0.0)
    assert robot.stops == 1


def test_teleop_backend_command_exception_stops_controller_and_closes():
    robot = _FakeTeleopRobot()
    robot.fail_command = True
    backend = XLeRobotTeleopBackend(robot, _localize_fresh, clock=lambda: 0.0)
    controller = LightNav0Controller(backend, clock=lambda: 0.0)
    controller.submit(_trajectory())
    with pytest.raises(RuntimeError, match="teleop command failed"):
        controller.tick(now_s=0.0)
    assert robot.stops >= 1
    controller.close()
    assert robot.closed is True
