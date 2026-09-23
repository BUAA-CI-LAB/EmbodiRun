from types import SimpleNamespace

import pytest

from embodirun.bindings.franka.fr3.pi05 import (
    Pi05FR3Mapper,
    Pi05FR3MapperConfig,
    Pi05FR3MapperError,
)
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.runtime import ControlRuntime
from embodirun.services.inference import (
    PolicyAction,
    PolicyObservation,
    PolicyResult,
    Session,
)


class FakeRobot:
    def __init__(self) -> None:
        self.robot_id = "fr3"
        self.executed = []
        self.stopped = False

    def observe(self):
        return SimpleNamespace(
            timestamp_s=1.0,
            values={"joint_positions_rad": [0.0] * 7},
        )

    def execute(self, action) -> None:
        self.executed.append(action)

    def stop(self) -> None:
        self.stopped = True


class FakeEmbodiInferClient:
    def __init__(self) -> None:
        self.open_session_called = 0
        self.step_called = 0
        self.reset_called = False
        self.closed = False

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata=None,
    ) -> Session:
        self.open_session_called += 1
        assert robot_id == "fr3"
        assert action_space == "pi05.action_chunk.v1"
        assert metadata is None
        return Session("session-1", 0)

    def step(self, observation: PolicyObservation) -> PolicyResult:
        self.step_called += 1
        return PolicyResult(
            request_id=observation.request_id,
            session_id=observation.session_id,
            step_id=observation.step_id,
            session_revision=1,
            action_space="pi05.action_chunk.v1",
            actions=(
                PolicyAction(
                    "action_chunk",
                    {
                        "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                        "feature_names": [
                            "joint_1",
                            "joint_2",
                            "joint_3",
                            "joint_4",
                            "joint_5",
                            "joint_6",
                            "joint_7",
                            "gripper_width_m",
                        ],
                    },
                ),
            ),
        )

    def reset(self, session_id: str, request_id: str) -> Session:
        self.reset_called = True
        assert session_id == "session-1"
        assert request_id.startswith("reset-")
        return Session(session_id, 1)

    def close(self, session_id: str) -> None:
        self.closed = True
        assert session_id == "session-1"


def test_pi05_mapper_extracts_joint_position_chunk() -> None:
    mapper = Pi05FR3Mapper(config=Pi05FR3MapperConfig(joint_indices=(0, 1, 2, 3, 4, 5, 6)))
    actions = mapper.map_result(
        PolicyResult(
            request_id="r1",
            session_id="s1",
            step_id=0,
            session_revision=1,
            action_space="pi05.action_chunk.v1",
            actions=(
                PolicyAction(
                    "action_chunk",
                    {
                        "data": [
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        ],
                        "feature_names": [],
                    },
                ),
            ),
        )
    )
    assert len(actions) == 2
    assert actions[0].values["type"] == "joint_position"
    assert actions[0].values["joint_positions_rad"] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    assert actions[1].values["joint_positions_rad"] == [
        8.0,
        7.0,
        6.0,
        5.0,
        4.0,
        3.0,
        2.0,
    ]


def test_pi05_runtime_steps_and_reset_flow() -> None:
    robot = FakeRobot()
    client = FakeEmbodiInferClient()
    runtime = ControlRuntime(
        robot,
        client,
        instruction="pick",
        mapper=Pi05FR3Mapper(),
        chunk_steps=1,
    )
    frame = CameraFrame("camera-0", "image/jpeg", b"123")
    result = runtime.step((frame,))
    assert result.step_id == 0
    assert len(robot.executed) == 1
    assert client.step_called == 1
    assert robot.executed[0].values["joint_positions_rad"][0] == 0.1
    assert client.open_session_called == 1
    runtime.step((frame,), reset=True)
    assert client.reset_called
    assert len(robot.executed) == 2
    assert client.step_called == 2
    runtime.close()
    assert client.closed


def test_pi05_mapper_guard_on_wrong_kind() -> None:
    mapper = Pi05FR3Mapper()
    with pytest.raises(Pi05FR3MapperError):
        mapper.map_result(
            PolicyResult(
                request_id="r1",
                session_id="s1",
                step_id=0,
                session_revision=1,
                action_space="pi05.action_chunk.v1",
                actions=(
                    PolicyAction(
                        "joint_position",
                        {"joint_positions_rad": [0.0] * 7},
                    ),
                ),
            )
        )
