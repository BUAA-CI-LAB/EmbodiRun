from __future__ import annotations

import asyncio
import io
import math
from types import SimpleNamespace

import pytest

from embodirun.bindings.xlerobot.lightnav0 import (
    BodyVelocity,
    LocalSegmentConfig,
    LocalSegmentExecutor,
    LocalSegmentFeedback,
    LocalSegmentFeedbackStale,
    XLeRobotStopUnconfirmed,
    XLeRobotTeleopLocalSegmentBackend,
    select_unicycle_waypoint,
    waypoint_to_body_velocity,
)
from embodirun.bindings.xlerobot.lightnav0 import cli as teleop_app
from embodirun.bindings.xlerobot.lightnav0.cli import (
    CameraFreshnessUnavailable,
    TeleopRunConfig,
    _prepare_output_path,
    _run_from_args,
    read_fresh_camera,
    run_local_segment_loop,
)
from embodirun.bindings.xlerobot.lightnav0.tracker import LightNav0Waypoint


def _output(rows, stop=False):
    return {"waypoints": rows, "stop": stop}


def test_first_unicycle_waypoint_skips_lateral_only_rows():
    rows = (
        LightNav0Waypoint(0.0, 0.2, 0.0),
        LightNav0Waypoint(0.1, 0.0, 0.0),
    )
    index, waypoint = select_unicycle_waypoint(rows)
    assert index == 1
    assert waypoint == rows[1]


def test_local_waypoint_velocity_preserves_mixed_curvature_when_both_exceed_caps():
    command = waypoint_to_body_velocity(
        LightNav0Waypoint(1.0, 0.4, -1.0),
        duration_s=0.25,
        max_linear_velocity_m_s=0.3,
        max_angular_velocity_rad_s=0.6,
    )
    assert command == BodyVelocity(0.3, 0.0, -0.3)
    assert command.omega / command.vx == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("forward_m", "yaw_rad", "expected_vx", "expected_omega"),
    [
        (1.0, 0.1, 0.3, 0.03),
        (-1.0, -0.1, -0.3, -0.03),
    ],
)
def test_local_waypoint_velocity_single_linear_limit_keeps_angular_ratio(
    forward_m, yaw_rad, expected_vx, expected_omega
):
    command = waypoint_to_body_velocity(
        LightNav0Waypoint(forward_m, 0.0, yaw_rad),
        duration_s=0.25,
        max_linear_velocity_m_s=0.3,
        max_angular_velocity_rad_s=0.6,
    )
    assert command == BodyVelocity(expected_vx, 0.0, expected_omega)


@pytest.mark.parametrize(
    ("forward_m", "yaw_rad", "expected_vx", "expected_omega"),
    [
        (0.05, 1.0, 0.03, 0.6),
        (-0.05, -1.0, -0.03, -0.6),
    ],
)
def test_local_waypoint_velocity_single_angular_limit_keeps_linear_ratio(
    forward_m, yaw_rad, expected_vx, expected_omega
):
    command = waypoint_to_body_velocity(
        LightNav0Waypoint(forward_m, 0.0, yaw_rad),
        duration_s=0.25,
        max_linear_velocity_m_s=0.3,
        max_angular_velocity_rad_s=0.6,
    )
    assert command == BodyVelocity(expected_vx, 0.0, expected_omega)


@pytest.mark.parametrize("yaw_rad", [1.0, -1.0])
def test_local_waypoint_velocity_pure_rotation_has_no_forward_component(yaw_rad):
    command = waypoint_to_body_velocity(
        LightNav0Waypoint(0.0, 0.0, yaw_rad),
        duration_s=0.25,
        max_linear_velocity_m_s=0.3,
        max_angular_velocity_rad_s=0.6,
    )
    assert command == BodyVelocity(0.0, 0.0, 0.6 if yaw_rad > 0 else -0.6)


def test_local_waypoint_velocity_avoids_raw_division_overflow():
    command = waypoint_to_body_velocity(
        LightNav0Waypoint(1e100, 0.0, 1e100),
        duration_s=math.nextafter(0.0, 1.0),
        max_linear_velocity_m_s=0.3,
        max_angular_velocity_rad_s=0.6,
    )
    assert command == BodyVelocity(0.3, 0.0, 0.3)


class _FakeSegmentBackend:
    def __init__(self, feedback: LocalSegmentFeedback):
        self.feedback_value = feedback
        self.commands = []
        self.stops = []
        self.closed = False

    def feedback(self):
        return self.feedback_value

    def set_body_velocity(self, velocity):
        self.commands.append(velocity)

    def stop(self, *, reason="stop"):
        self.stops.append(reason)
        return {
            "stop_confirmed": True,
            "stationary_confirmed": True,
            "physical_outcome": "stopped",
        }

    def close(self):
        self.closed = True


def test_local_waypoint_velocity_overflow_fails_closed_and_executor_stops():
    backend = _FakeSegmentBackend(LocalSegmentFeedback(BodyVelocity.zero(), received_at_s=0.0, state_timestamp_ns=1))
    executor = LocalSegmentExecutor(backend, LocalSegmentConfig(duration_s=math.nextafter(0.0, 1.0)))
    with pytest.raises(ValueError, match="effective duration"):
        executor.execute(_output([[1e308, 0.0, 0.0]]), now_s=0.0)
    assert backend.stops == ["local-segment-exception"]


def test_local_segment_executor_sends_forward_yaw_without_pose():
    backend = _FakeSegmentBackend(LocalSegmentFeedback(BodyVelocity.zero(), received_at_s=0.0, state_timestamp_ns=1))
    executor = LocalSegmentExecutor(backend, LocalSegmentConfig(duration_s=0.25), clock=lambda: 0.0)
    step = executor.execute(_output([[0.0, 0.2, 0.0], [0.1, 0.0, 0.2]]), now_s=0.0)
    assert step.waypoint_index == 1
    assert step.command == BodyVelocity(0.3, 0.0, 0.6)
    assert backend.commands == [BodyVelocity(0.3, 0.0, 0.6)]


def test_local_segment_executor_stale_feedback_stops_fail_closed():
    backend = _FakeSegmentBackend(LocalSegmentFeedback(BodyVelocity.zero(), received_at_s=-1.0, state_timestamp_ns=1))
    executor = LocalSegmentExecutor(backend, clock=lambda: 0.0)
    with pytest.raises(LocalSegmentFeedbackStale):
        executor.execute(_output([[0.1, 0.0, 0.0]]), now_s=0.0)
    assert backend.stops == ["local-segment-exception"]


def test_local_segment_executor_rejects_backwards_feedback_timestamp():
    class BackwardsBackend(_FakeSegmentBackend):
        def __init__(self):
            super().__init__(LocalSegmentFeedback(BodyVelocity.zero(), 0.0, 2))
            self.samples = [
                LocalSegmentFeedback(BodyVelocity.zero(), 0.0, 2),
                LocalSegmentFeedback(BodyVelocity.zero(), 0.0, 1),
            ]

        def feedback(self):
            return self.samples.pop(0)

    backend = BackwardsBackend()
    executor = LocalSegmentExecutor(backend, clock=lambda: 0.0)
    executor.execute(_output([[0.1, 0.0, 0.0]]), now_s=0.0)
    with pytest.raises(LocalSegmentFeedbackStale, match="backwards"):
        executor.execute(_output([[0.1, 0.0, 0.0]]), now_s=0.0)
    assert backend.stops[-1] == "local-segment-exception"


class _FakeRobot:
    armed = True

    def __init__(self, *, state_cached=False, camera_status=True):
        self.state_cached = state_cached
        self.camera_status = camera_status
        self.state_timestamp_ns = 100
        self.commands = []
        self.stops = []
        self.closed = False
        self.state = {"x.vel": 0.0, "theta.vel": 0.0}

    def read(self):
        self.state_timestamp_ns += 1
        status = {
            "head": {
                "fresh": self.camera_status,
                "timestamp_ns": self.state_timestamp_ns,
            }
        }
        return (
            {
                "state": dict(self.state),
                "state_cached": self.state_cached,
                "state_timestamp_ns": self.state_timestamp_ns,
                "errors": [],
                "armed": self.armed,
                "control_owned": self.armed,
                "camera_status": status,
            },
            {"head": _jpeg()},
        )

    def command(self, action):
        self.commands.append(dict(action))
        self.state.update(action)
        return {
            "accepted": True,
            "command_accepted": True,
            "errors": [],
            "physical_outcome": "commanded",
        }

    def stop(self):
        self.stops.append(True)
        self.state.update({"x.vel": 0.0, "theta.vel": 0.0})
        return {
            "stop_confirmed": True,
            "stationary_confirmed": True,
            "physical_outcome": "stopped",
            "errors": [],
        }

    def close(self):
        self.closed = True


def _jpeg():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (4, 4), "red")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_local_segment_backend_rejects_cached_feedback():
    robot = _FakeRobot(state_cached=True)
    backend = XLeRobotTeleopLocalSegmentBackend(robot, clock=lambda: 0.0)
    with pytest.raises(RuntimeError, match="cached"):
        backend.feedback()


def test_local_segment_backend_reuses_command_and_stop_safety():
    robot = _FakeRobot()
    backend = XLeRobotTeleopLocalSegmentBackend(robot, clock=lambda: 0.0)
    feedback = backend.feedback()
    assert feedback.velocity == BodyVelocity.zero()
    assert feedback.state_timestamp_ns > 0
    backend.set_body_velocity(BodyVelocity(0.2, 0.0, 0.4))
    assert robot.commands[-1]["x.vel"] == pytest.approx(0.2)
    assert robot.commands[-1]["theta.vel"] == pytest.approx(0.4 * 180.0 / 3.141592653589793)
    backend.stop(reason="test")
    assert backend.last_stop_report["stationary_confirmed"] is True


def test_camera_freshness_is_required_and_separate_from_wheel_feedback():
    robot = _FakeRobot(camera_status=False)
    with pytest.raises(CameraFreshnessUnavailable, match="stale"):
        read_fresh_camera(robot, "head")


class _FakeProvider:
    def __init__(self):
        self.calls = 0

    async def reset_session(self, _session_id):
        pass

    async def predict(self, rgb, **kwargs):
        self.calls += 1
        stop = self.calls >= 2
        return SimpleNamespace(
            output=SimpleNamespace(
                actions=((0.1, 0.0, 0.0),) * 10,
                metadata={"stop": stop, "raw_text": ""},
            )
        )


def test_rgb_local_segment_loop_repredicts_without_pose():
    robot = _FakeRobot()

    async def no_sleep(_duration):
        return None

    result = asyncio.run(
        run_local_segment_loop(
            _FakeProvider(),
            robot,
            TeleopRunConfig(
                instruction="go forward",
                camera="head",
                max_steps=3,
                segment=LocalSegmentConfig(duration_s=0.01),
            ),
            clock=lambda: 0.0,
            sleep=no_sleep,
        ),
    )
    assert result["mode"] == "xlerobot-local-segment"
    assert result["metric_pose_available"] is False
    assert result["decisions"] == 2
    assert result["steps"][-1]["reason"] == "explicit-stop"
    assert robot.closed is True


def test_rgb_loop_serializes_numpy_actions_and_reserves_output(tmp_path):
    np = pytest.importorskip("numpy")
    output = tmp_path / "run.json"
    reserved = _prepare_output_path(str(output))
    assert reserved == output.resolve()
    assert output.exists()
    with pytest.raises(FileExistsError):
        _prepare_output_path(str(output))

    class NumpyProvider(_FakeProvider):
        async def predict(self, rgb, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                output=SimpleNamespace(
                    actions=np.asarray([[0.0, 0.0, 0.0]] * 10, dtype=np.float32),
                    metadata={"stop": True},
                )
            )

    robot = _FakeRobot()
    result = asyncio.run(
        run_local_segment_loop(
            NumpyProvider(),
            robot,
            TeleopRunConfig(instruction="stop", camera="head", max_steps=1),
            clock=lambda: 0.0,
        )
    )
    assert isinstance(result["steps"][0]["waypoints"], list)
    import json

    output.write_text(json.dumps(result, allow_nan=False))


def test_rgb_loop_retries_prediction_after_local_age_budget():
    class Clock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return 0.6 if self.calls == 6 else 0.0

    robot = _FakeRobot()
    result = asyncio.run(
        run_local_segment_loop(
            _FakeProvider(),
            robot,
            TeleopRunConfig(
                instruction="stop",
                camera="head",
                max_steps=1,
                prediction_timeout_s=0.5,
                max_prediction_retries=1,
            ),
            clock=Clock(),
        )
    )
    assert result["decisions"] == 2
    assert result["steps"][0]["reason"] == "prediction-stale"
    assert result["steps"][1]["reason"] == "explicit-stop"


def test_rgb_loop_rejects_repeated_camera_source_timestamp():
    class FixedCameraRobot(_FakeRobot):
        def read(self):
            observation, images = super().read()
            observation["camera_status"]["head"]["timestamp_ns"] = 123
            return observation, images

    robot = FixedCameraRobot()
    with pytest.raises(CameraFreshnessUnavailable, match="timestamp"):
        asyncio.run(
            run_local_segment_loop(
                _FakeProvider(),
                robot,
                TeleopRunConfig(instruction="go", camera="head", max_steps=2),
                clock=lambda: 0.0,
            )
        )


def test_final_close_stop_failure_attaches_concrete_report():
    class FinalStopUnknownRobot(_FakeRobot):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return super().stop()
            self.stops.append(True)
            return {
                "stop_confirmed": False,
                "stationary_confirmed": False,
                "physical_outcome": "unknown",
                "errors": ["stationary feedback unavailable"],
            }

    class StopProvider:
        async def reset_session(self, _session_id):
            return None

        async def predict(self, rgb, **kwargs):
            return SimpleNamespace(
                output=SimpleNamespace(
                    actions=((0.0, 0.0, 0.0),) * 10,
                    metadata={"stop": True},
                )
            )

    robot = FinalStopUnknownRobot()
    with pytest.raises(XLeRobotStopUnconfirmed) as raised:
        asyncio.run(
            run_local_segment_loop(
                StopProvider(),
                robot,
                TeleopRunConfig(instruction="stop", camera="head", max_steps=1),
                clock=lambda: 0.0,
            )
        )
    report = raised.value.lightnav_stop_report
    assert report["stop_confirmed"] is False
    assert report["physical_outcome"] == "unknown"


def test_cli_error_is_written_after_output_reservation(tmp_path):
    output = tmp_path / "error.json"
    args = SimpleNamespace(
        robot_factory="module_that_does_not_exist:build_robot",
        checkpoint="unused",
        instruction="go",
        camera="head",
        output=str(output),
        session_id="test",
        max_steps=1,
        duration_s=0.01,
        max_linear_velocity=0.3,
        max_angular_velocity=0.6,
        prediction_timeout_s=0.5,
        max_prediction_retries=1,
        device="cpu",
        dtype="bfloat16",
        backend="hf",
        attention="sdpa",
        cache_vision=True,
        use_cuda_graph=False,
    )
    with pytest.raises(ModuleNotFoundError):
        asyncio.run(_run_from_args(args))
    import json

    saved = json.loads(output.read_text())
    assert saved["status"] == "error"
    assert saved["error"]["type"] == "ModuleNotFoundError"


def test_cli_loads_provider_before_invoking_robot_factory(monkeypatch):
    events = []

    class FakeProvider:
        async def start(self):
            events.append("provider-start")

        async def aclose(self):
            events.append("provider-close")

    robot = _FakeRobot()

    def factory():
        events.append("factory")
        return robot

    async def fake_loop(_provider, _robot, _config):
        events.append("loop")
        return {"mode": "test"}

    monkeypatch.setattr(teleop_app, "load_robot_factory", lambda _spec: factory)
    monkeypatch.setattr(teleop_app, "create_client", lambda _args: FakeProvider())
    monkeypatch.setattr(teleop_app, "run_local_segment_loop", fake_loop)
    args = SimpleNamespace(
        robot_factory="ignored:factory",
        checkpoint="unused",
        instruction="go",
        camera="head",
        output=None,
        session_id="test",
        max_steps=1,
        duration_s=0.01,
        max_linear_velocity=0.3,
        max_angular_velocity=0.6,
        prediction_timeout_s=0.5,
        max_prediction_retries=1,
        device="cpu",
        dtype="bfloat16",
        backend="hf",
        attention="sdpa",
        cache_vision=True,
        use_cuda_graph=False,
    )
    result = asyncio.run(teleop_app._run_from_args(args))
    assert result == {"mode": "test"}
    assert events == ["provider-start", "factory", "loop", "provider-close"]
