import json
import sys
import time
from types import SimpleNamespace

import pytest

from embodirun.robots import RobotAction
from embodirun.robots.lerobot.so101 import (
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
    SO101Config,
)


class FakeSO101:
    def __init__(self, positions):
        self.positions = dict(zip(SO101_POSITION_FEATURES, positions))
        self.is_connected = False
        self.is_calibrated = True
        self.actions = []

    def connect(self, *, calibrate):
        assert calibrate is False
        self.is_connected = True

    def get_observation(self):
        return dict(self.positions)

    def send_action(self, action):
        self.actions.append(dict(action))
        self.positions.update(action)

    def disconnect(self):
        self.is_connected = False


def action(joints, gripper):
    return RobotAction(
        timestamp_s=time.time(),
        values={
            "type": "joint_position",
            "joint_positions_deg": joints,
            "gripper_position": gripper,
        },
    )


def test_so101_reject_mode_remains_fail_closed() -> None:
    hardware = FakeSO101([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    adapter = SO101Adapter(
        SO101Config(port="/dev/fake", max_joint_step_deg=5.0),
        controller=hardware,
    )
    adapter.connect()

    with pytest.raises(SO101AdapterError, match="exceeds 5.000000 degrees"):
        adapter.execute(action([20.0, 0.0, 0.0, 0.0, 0.0], 50.0))

    assert hardware.actions == []


def test_so101_clip_mode_bounds_every_joint_and_gripper() -> None:
    hardware = FakeSO101([0.0, 10.0, -10.0, 20.0, -20.0, 50.0])
    adapter = SO101Adapter(
        SO101Config(
            port="/dev/fake",
            max_joint_step_deg=5.0,
            max_gripper_step=10.0,
            step_limit_mode="clip",
        ),
        controller=hardware,
    )
    adapter.connect()

    adapter.execute(action([20.0, -20.0, -7.0, 16.0, -25.0], 80.0))

    assert hardware.actions == [
        {
            "shoulder_pan.pos": 5.0,
            "shoulder_lift.pos": 5.0,
            "elbow_flex.pos": -7.0,
            "wrist_flex.pos": 16.0,
            "wrist_roll.pos": -25.0,
            "gripper.pos": 60.0,
        }
    ]


def test_so101_clip_mode_bounds_each_command_in_a_sequence() -> None:
    hardware = FakeSO101([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    adapter = SO101Adapter(
        SO101Config(
            port="/dev/fake",
            max_joint_step_deg=5.0,
            max_gripper_step=10.0,
            step_limit_mode="clip",
        ),
        controller=hardware,
    )
    adapter.connect()

    target = action([20.0, 0.0, 0.0, 0.0, 0.0], 50.0)
    adapter.execute(target)
    adapter.execute(target)

    assert [item["shoulder_pan.pos"] for item in hardware.actions] == [5.0, 10.0]


def test_so101_config_rejects_unknown_step_limit_mode() -> None:
    with pytest.raises(ValueError, match="step_limit_mode"):
        SO101Config(port="/dev/fake", step_limit_mode="unsafe")


@pytest.mark.parametrize("read_only", [False, True])
def test_feetech_read_only_capture_never_writes_motor_registers(tmp_path, monkeypatch, read_only):
    writes = []

    class Port:
        is_open = False

        def openPort(self):
            self.is_open = True
            return True

        def closePort(self):
            self.is_open = False

        def setPacketTimeoutMillis(self, value):
            pass

    class Packet:
        def ping(self, port, motor):
            return 777, 0, 0

        def read1ByteTxRx(self, port, motor, address):
            return (1 if address in (0, 1) else 0), 0, 0

        def read2ByteTxRx(self, port, motor, address):
            return {9: 1000, 11: 3000, 31: 0, 56: 2000}[address], 0, 0

        def write1ByteTxRx(self, port, motor, address, value):
            writes.append((motor, address, value))
            return 0, 0

        write2ByteTxRx = write1ByteTxRx

    class Reader:
        def __init__(self, *args):
            pass

        def addParam(self, motor):
            return True

        def txRxPacket(self):
            return 0

        def isAvailable(self, *args):
            return True

        def getData(self, *args):
            return 2000

    port = Port()
    monkeypatch.setitem(
        sys.modules,
        "scservo_sdk",
        SimpleNamespace(
            PortHandler=lambda _: port, PacketHandler=lambda _: Packet(), GroupSyncRead=Reader, COMM_SUCCESS=0
        ),
    )
    calibration = {
        name.removesuffix(".pos"): {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 1000,
            "range_max": 3000,
        }
        for index, name in enumerate(SO101_POSITION_FEATURES, 1)
    }
    (tmp_path / "arm.json").write_text(json.dumps(calibration))
    robot = SO101Adapter(SO101Config(port="/dev/fake", robot_id="arm", calibration_dir=tmp_path), read_only=read_only)
    robot.connect()
    observation = robot.observe()
    assert observation.values == {"joint_positions_deg": [0.0] * 5, "gripper_position": 50.0}
    assert isinstance(observation.metadata["captured_timestamp_ns"], int)
    assert observation.metadata["clock_domain"] == "host_monotonic_ns"
    if read_only:
        with pytest.raises(SO101AdapterError, match="read-only"):
            robot.execute(action([0.0] * 5, 50.0))
        with pytest.raises(SO101AdapterError, match="read-only"):
            robot.stop()
    robot.close()
    assert not port.is_open
    assert bool(writes) is not read_only
