from types import SimpleNamespace

import pytest

from embodirun.bindings.lerobot.so101.pi05 import Pi05SO101Mapper
from embodirun.robots import RobotAction
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.runtime import ControlRuntime
from embodirun.services.inference import (
    ImagePayload,
    PolicyAction,
    PolicyResult,
    Session,
)


class FakeRobot:
    robot_id = "so101-1"

    def __init__(self) -> None:
        self.actions = []

    def observe(self):
        return SimpleNamespace(timestamp_s=1.5, values={"shoulder_pan.pos": 0.0})

    def execute(self, action):
        self.actions.append(action)

    def stop(self):
        return None


class FakeClient:
    def __init__(self) -> None:
        self.observation = None

    def open_session(self, *, robot_id, action_space):
        assert robot_id == "so101-1"
        assert action_space == "pi05.action_chunk.v1"
        return Session("session-1", 0)

    def step(self, observation):
        self.observation = observation
        return SimpleNamespace(session_revision=1)


class FakeMapper(Pi05SO101Mapper):
    def map_result(self, _result):
        return (RobotAction(2.0, {"type": "joint_position"}),)


class ThreeActionMapper(FakeMapper):
    def map_result(self, _result):
        return tuple(RobotAction(float(index), {"type": "joint_position", "index": index}) for index in range(3))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_so101_binding_converts_camera_frames_to_inference_images() -> None:
    robot = FakeRobot()
    client = FakeClient()
    runtime = ControlRuntime(
        robot,
        client,
        instruction="pick up the block",
        mapper=FakeMapper(),
        chunk_steps=1,
    )

    runtime.step((CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),))

    assert client.observation is not None
    assert client.observation.images == (ImagePayload("observation.images.front", "image/jpeg", b"jpeg"),)
    assert client.observation.metadata == {"robot_timestamp_s": 1.5}
    assert len(robot.actions) == 1


def test_so101_mapper_preserves_all_action_chunk_rows() -> None:
    actions = Pi05SO101Mapper().map_result(
        PolicyResult(
            request_id="request-1",
            session_id="session-1",
            step_id=0,
            session_revision=1,
            action_space="pi05.action_chunk.v1",
            actions=(
                PolicyAction(
                    "action_chunk",
                    {
                        "data": [
                            [1.0, 2.0, 3.0, 4.0, 5.0, 60.0],
                            [6.0, 7.0, 8.0, 9.0, 10.0, 70.0],
                        ],
                        "feature_names": [
                            "shoulder_pan.pos",
                            "shoulder_lift.pos",
                            "elbow_flex.pos",
                            "wrist_flex.pos",
                            "wrist_roll.pos",
                            "gripper.pos",
                        ],
                    },
                ),
            ),
        )
    )

    assert len(actions) == 2
    assert actions[0].values["joint_positions_deg"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert actions[1].values["joint_positions_deg"] == [6.0, 7.0, 8.0, 9.0, 10.0]
    assert actions[1].values["gripper_position"] == 70.0
    assert actions[0].metadata["chunk_index"] == 0
    assert actions[1].metadata["chunk_index"] == 1
    assert {action.metadata["chunk_size"] for action in actions} == {2}


def test_control_runtime_plays_action_chunk_at_control_rate() -> None:
    robot = FakeRobot()
    client = FakeClient()
    clock = FakeClock()
    runtime = ControlRuntime(
        robot,
        client,
        instruction="pick up the block",
        mapper=ThreeActionMapper(),
        chunk_steps=3,
        control_hz=20.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    runtime.step((CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),))

    assert [action.values["index"] for action in robot.actions] == [0, 1, 2]
    assert clock.sleeps == [0.05, 0.05]


def test_control_runtime_executes_only_requested_chunk_steps() -> None:
    robot = FakeRobot()
    client = FakeClient()
    runtime = ControlRuntime(
        robot,
        client,
        instruction="pick up the block",
        mapper=ThreeActionMapper(),
        chunk_steps=2,
    )

    runtime.step((CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),))

    assert [action.values["index"] for action in robot.actions] == [0, 1]


def test_control_runtime_rejects_short_chunk_before_execution() -> None:
    robot = FakeRobot()
    client = FakeClient()
    runtime = ControlRuntime(
        robot,
        client,
        instruction="pick up the block",
        mapper=FakeMapper(),
        chunk_steps=2,
    )

    with pytest.raises(RuntimeError, match="fewer than requested"):
        runtime.step((CameraFrame("observation.images.front", "image/jpeg", b"jpeg"),))

    assert robot.actions == []
