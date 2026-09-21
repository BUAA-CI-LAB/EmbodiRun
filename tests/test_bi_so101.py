from dataclasses import replace

import pytest

from embodirun.application.model_loop import ControlRuntime
from embodirun.bindings.lerobot.bi_so101.pi05 import Pi05BiSO101Mapper
from embodirun.bindings.lerobot.so101.pi05 import Pi05SO101MapperError
from embodirun.model_services import PolicyAction, PolicyResult, Session
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.lerobot.bi_so101 import (
    BI_SO101_ACTION_SPACE,
    BI_SO101_POSITION_FEATURES,
    BiSO101Adapter,
    BiSO101Config,
)
from embodirun.robots.lerobot.so101 import (
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
)
from embodirun.robots.sensors.cameras import CameraFrame


class Bus:
    def __init__(self, positions):
        self.positions = dict(zip(SO101_POSITION_FEATURES, positions))
        self.is_connected = False
        self.is_calibrated = True
        self.actions = []
        self.disconnects = 0
        self.fail_connect = False
        self.fail_send = False
        self.fail_disconnect = False

    def connect(self, *, calibrate, prepare=True):
        assert calibrate is False
        if not prepare:
            return
        self.is_connected = True
        if self.fail_connect:
            raise OSError("connect failure")

    def get_observation(self):
        return dict(self.positions)

    def send_action(self, action):
        self.actions.append(dict(action))
        if self.fail_send:
            self.fail_send = False
            raise OSError("send failure")
        self.positions.update(action)

    def disconnect(self):
        self.disconnects += 1
        if self.fail_disconnect:
            raise OSError("disconnect failure")
        self.is_connected = False


@pytest.fixture
def robot():
    config = BiSO101Config.from_mapping(
        "pair",
        {
            "left_port": "/dev/left",
            "right_port": "/dev/right",
            "max_joint_step_deg": 5.0,
            "max_gripper_step": 10.0,
        },
    )
    left = Bus([0, 1, 2, 3, 4, 50])
    right = Bus([5, 6, 7, 8, 9, 60])
    adapter = BiSO101Adapter(
        config,
        left=SO101Adapter(config.left, controller=left),
        right=SO101Adapter(config.right, controller=right),
    )
    return adapter, left, right


def target():
    return RobotAction(
        1.0,
        {
            "type": "joint_position",
            "left": {"joint_positions_deg": [1, 2, 3, 4, 5], "gripper_position": 55},
            "right": {"joint_positions_deg": [6, 7, 8, 9, 10], "gripper_position": 65},
        },
        {"action_space": BI_SO101_ACTION_SPACE},
    )


def result(rows, names=BI_SO101_POSITION_FEATURES):
    return PolicyResult(
        request_id="request",
        session_id="session",
        step_id=0,
        session_revision=1,
        action_space="pi05.action_chunk.v1",
        actions=(PolicyAction("action_chunk", {"data": rows, "feature_names": names}),),
    )


def test_observe_and_execute_keep_left_right_and_grippers_separate(robot):
    adapter, left, right = robot
    adapter.connect()
    observation = adapter.observe()
    assert list(observation.values) == list(BI_SO101_POSITION_FEATURES)
    assert list(observation.values.values()) == [0, 1, 2, 3, 4, 50, 5, 6, 7, 8, 9, 60]
    adapter.execute(target())
    assert list(left.actions[0].values()) == [1, 2, 3, 4, 5, 55]
    assert list(right.actions[0].values()) == [6, 7, 8, 9, 10, 65]


def test_observe_reports_per_arm_timestamps_and_sample_skew(robot, monkeypatch):
    adapter, _, _ = robot
    row = {
        "joint_positions_deg": [0, 1, 2, 3, 4],
        "gripper_position": 50,
    }
    monkeypatch.setattr(
        adapter.left,
        "observe",
        lambda: RobotObservation(10.0, row),
    )
    monkeypatch.setattr(
        adapter.right,
        "observe",
        lambda: RobotObservation(10.125, row),
    )

    observation = adapter.observe()

    assert observation.timestamp_s == 10.0
    assert observation.metadata["arm_timestamps_s"] == {
        "left": 10.0,
        "right": 10.125,
    }
    assert observation.metadata["arm_sample_skew_s"] == pytest.approx(0.125)


@pytest.mark.parametrize(
    "invalid",
    [
        {"joint_positions_deg": [100] * 5, "gripper_position": 60},
        {"joint_positions_deg": [5, 6, 7, 8, 9], "gripper_position": 100},
        {"joint_positions_deg": [5, 6, 7, 8, 9], "gripper_position": 101},
        {"joint_positions_deg": [float("nan")] * 5, "gripper_position": 60},
        {"joint_positions_deg": [5] * 4, "gripper_position": 60},
        {"joint_positions_deg": [5] * 5},
        {"joint_positions_deg": [5] * 5, "gripper_position": True},
        {"type": "stop"},
    ],
)
def test_invalid_right_target_never_writes_left(robot, invalid):
    adapter, left, right = robot
    adapter.connect()
    action = target()
    action.values["right"] = invalid
    with pytest.raises(SO101AdapterError):
        adapter.execute(action)
    assert left.actions == right.actions == []


def test_partial_connect_failure_disconnects_both(robot):
    adapter, left, right = robot
    right.fail_connect = True
    with pytest.raises(OSError, match="connect failure"):
        adapter.connect()
    assert not left.is_connected and not right.is_connected
    assert left.disconnects == right.disconnects == 1


def test_partial_send_failure_attempts_measured_holds_on_both(robot):
    adapter, left, right = robot
    adapter.connect()
    right.fail_send = True
    with pytest.raises(OSError, match="send failure"):
        adapter.execute(target())
    assert len(left.actions) == len(right.actions) == 2
    assert list(right.actions[-1].values()) == [5, 6, 7, 8, 9, 60]
    assert left.actions[-1] == left.positions


def test_failed_hold_still_attempts_other_arm(robot):
    adapter, left, right = robot
    adapter.connect()
    left.fail_send = True
    with pytest.raises(SO101AdapterError, match="left: send failure"):
        adapter.stop()
    assert len(left.actions) == len(right.actions) == 1


def test_execute_reports_hold_failure(robot, monkeypatch):
    adapter, left, right = robot
    adapter.connect()
    right.fail_send = True

    def failed_hold():
        raise OSError("hold error")

    monkeypatch.setattr(adapter.left, "stop", failed_hold)
    with pytest.raises(SO101AdapterError, match="send failure.*hold failed.*hold error"):
        adapter.execute(target())
    assert len(right.actions) == 2


def test_close_attempts_both_on_failure(robot):
    adapter, left, right = robot
    adapter.connect()
    left.fail_disconnect = True
    with pytest.raises(SO101AdapterError, match="left: disconnect failure"):
        adapter.close()
    assert left.disconnects == right.disconnects == 1
    assert not right.is_connected


def test_explicit_clip_limits_both_arms(robot):
    adapter, left, right = robot
    for arm in (adapter.left, adapter.right):
        arm.config = replace(arm.config, step_limit_mode="clip")
    adapter.connect()
    action = target()
    for side in ("left", "right"):
        action.values[side] = {
            "joint_positions_deg": [100] * 5,
            "gripper_position": 100,
        }
    adapter.execute(action)
    assert list(left.actions[0].values()) == [5, 6, 7, 8, 9, 60]
    assert list(right.actions[0].values()) == [10, 11, 12, 13, 14, 70]


@pytest.mark.parametrize(
    "options",
    [
        {"right_port": "/dev/left"},
        {"right_port": ""},
        {"left_calibration_id": "same", "right_calibration_id": "same"},
        {"port": "/dev/other"},
        {"calibration_id": "same"},
        {"max_joint_step_deg": -1},
        {"unknown": True},
    ],
)
def test_invalid_dual_arm_config(options):
    with pytest.raises(ValueError):
        BiSO101Config.from_mapping("pair", {"left_port": "/dev/left", "right_port": "/dev/right", **options})


def test_mapper_uses_names_and_preserves_entire_chunk():
    row = [1, 2, 3, 4, 5, 55, 6, 7, 8, 9, 10, 65]
    actions = Pi05BiSO101Mapper().map_result(result([row[::-1]] * 50, BI_SO101_POSITION_FEATURES[::-1]))
    assert len(actions) == 50
    assert actions[0].values == target().values
    assert actions[-1].metadata["chunk_index"] == 49
    assert actions[-1].metadata["action_space"] == BI_SO101_ACTION_SPACE


@pytest.mark.parametrize(
    "names,row",
    [
        (SO101_POSITION_FEATURES, [0] * 6),
        (BI_SO101_POSITION_FEATURES[:-1], [0] * 11),
        (BI_SO101_POSITION_FEATURES[:-1] + (BI_SO101_POSITION_FEATURES[0],), [0] * 12),
        (BI_SO101_POSITION_FEATURES, [0] * 11),
        (BI_SO101_POSITION_FEATURES, [float("inf")] * 12),
    ],
)
def test_mapper_rejects_incompatible_features_or_values(names, row):
    with pytest.raises(Pi05SO101MapperError):
        Pi05BiSO101Mapper().map_result(result([row], names))


def test_control_runtime_sends_12_state_values_and_three_images(robot):
    adapter, left, right = robot
    adapter.connect()

    class Client:
        def open_session(self, **kwargs):
            assert kwargs == {
                "robot_id": "pair",
                "action_space": "pi05.action_chunk.v1",
            }
            return Session("session", 0)

        def step(self, request):
            assert list(request.state.values()) == [
                0,
                1,
                2,
                3,
                4,
                50,
                5,
                6,
                7,
                8,
                9,
                60,
            ]
            assert [image.name for image in request.images] == [
                "front",
                "left_wrist",
                "right_wrist",
            ]
            return result([[1, 2, 3, 4, 5, 55, 6, 7, 8, 9, 10, 65]] * 2)

    runtime = ControlRuntime(
        adapter,
        Client(),
        instruction="pick chips",
        mapper=Pi05BiSO101Mapper(),
        chunk_steps=2,
        control_hz=15,
        sleep=lambda _: None,
    )
    runtime.step(tuple(CameraFrame(name, "image/jpeg", b"jpeg") for name in ("front", "left_wrist", "right_wrist")))
    assert len(left.actions) == len(right.actions) == 2
