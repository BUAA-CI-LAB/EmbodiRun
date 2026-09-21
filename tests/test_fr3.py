import pytest

from embodirun.robots import RobotAction
from embodirun.robots.franka.fr3 import (
    FR3_ACTION_SPACE,
    FR3Adapter,
    FR3AdapterError,
    FR3Config,
)


class FakeState:
    q = [0.0] * 7
    dq = [0.0] * 7
    O_T_EE = [0.0] * 16


class FakeRobot:
    def __init__(self, host):
        self.host = host
        self.state = FakeState()
        self.moves = []
        self.recovered = False
        self.stopped = False
        self.relative_dynamics_factor = None

    def recover_from_errors(self):
        self.recovered = True

    def move(self, motion):
        self.moves.append(motion)

    def stop(self):
        self.stopped = True


class FakeGripper:
    def __init__(self, host):
        self.host = host
        self.width = 0.08
        self.moves = []
        self.stopped = False

    def move(self, width, speed):
        self.moves.append((width, speed))
        return True

    def stop(self):
        self.stopped = True


class FakeMotion:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeFranky:
    Robot = FakeRobot
    Gripper = FakeGripper
    JointMotion = FakeMotion
    CartesianMotion = FakeMotion
    JointStopMotion = FakeMotion
    Affine = FakeMotion

    class ReferenceType:
        Relative = "relative"


def action(values):
    return RobotAction(
        timestamp_s=1.0,
        values=values,
        metadata={"action_space": FR3_ACTION_SPACE},
    )


def test_joint_motion_and_observation_follow_franky_contract() -> None:
    adapter = FR3Adapter(FR3Config(host="172.16.0.2"), franky_module=FakeFranky)
    adapter.connect()
    assert adapter.robot.recovered
    assert adapter.robot.relative_dynamics_factor == pytest.approx(0.05)
    observation = adapter.observe()
    assert observation.values["joint_positions_rad"] == [0.0] * 7
    adapter.execute(action({"type": "joint_position", "joint_positions_rad": [0.1] * 7}))
    assert len(adapter.robot.moves) == 1


def test_joint_step_limit_is_fail_closed() -> None:
    adapter = FR3Adapter(FR3Config(host="172.16.0.2"), franky_module=FakeFranky)
    adapter.connect()
    with pytest.raises(FR3AdapterError, match="exceeds"):
        adapter.execute(action({"type": "joint_position", "joint_positions_rad": [0.5] * 7}))
    assert adapter.robot.moves == []


def test_relative_cartesian_and_gripper_commands() -> None:
    adapter = FR3Adapter(FR3Config(host="172.16.0.2"), franky_module=FakeFranky)
    adapter.connect()
    adapter.execute(action({"type": "cartesian_delta", "translation_m": [0.01, 0.0, 0.0]}))
    adapter.execute(action({"type": "gripper", "gripper_width_m": 0.04}))
    assert len(adapter.robot.moves) == 1
    assert adapter.gripper.moves == [(0.04, 0.02)]
