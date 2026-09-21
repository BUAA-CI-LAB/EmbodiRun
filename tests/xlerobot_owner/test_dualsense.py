import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun_xlerobot_owner import dualsense_drive
from embodirun_xlerobot_owner.dualsense_drive import (
    TETHERED_MAX_HOLD_S,
    TETHERED_MAX_LINEAR_M_S,
    ZERO,
    BaseSession,
    DriveMapping,
    HeadTiltMapping,
    check_feedback,
    resolve_motion_profile,
)
from embodirun_xlerobot_owner.dualsense_input import (
    DualSenseDevice,
    axis,
    decode_report,
    linux_devices,
)


def simple_report(sequence=1):
    return bytes([1, 128, 0, 255, 127, 8, 2, sequence << 2, 0, 255])


def full_report(bluetooth=True):
    report = bytearray(78 if bluetooth else 64)
    report[0] = 0x31 if bluetooth else 1
    offset = 2 if bluetooth else 1
    report[offset : offset + 10] = bytes([128, 0, 255, 127, 0, 255, 42, 8, 2, 0])
    report[offset + 27 : offset + 31] = (123456).to_bytes(4, "little")
    if bluetooth:
        report[-4:] = zlib.crc32(b"\xa1" + report[:-4]).to_bytes(4, "little")
    return bytes(report)


@pytest.mark.parametrize("report", [simple_report(), full_report(), full_report(False)])
def test_sticks_triggers_and_r1_have_same_mapping_in_all_transports(report):
    sample = decode_report(report, 10)
    assert sample.raw_axes == (128, 0, 255, 127)
    assert sample.sticks == (0, -1, 1, 0)
    assert sample.triggers == (0, 1)
    assert sample.buttons == {"r1"}
    assert sample.received_at == 10


def test_78_byte_simple_report_is_not_misread_as_enhanced_report():
    assert decode_report(simple_report() + bytes(68), 10) == decode_report(simple_report(), 10)


@pytest.mark.parametrize("report", [b"", b"\x31" * 77, b"\x01" * 20, b"\x32" * 78])
def test_invalid_reports_rejected(report):
    with pytest.raises(ValueError):
        decode_report(report, 10)


def test_corrupt_bluetooth_crc_rejected():
    report = bytearray(full_report())
    report[3] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode_report(report, 10)


def test_dpad_and_face_buttons_and_sequence_are_independent():
    report = bytearray(simple_report(31))
    report[5] = 0x91
    sample = decode_report(report, 10)
    assert sample.buttons == {"square", "triangle", "up", "right", "r1"}
    assert sample.sequence == 31
    assert "mute" not in sample.buttons


@pytest.mark.parametrize("raw", range(114, 142))
def test_centre_drift_stays_zero(raw):
    assert axis(raw) == 0


class FakeHid:
    def __init__(self, reports):
        self.reports = iter(reports)

    def read(self, size):
        return next(self.reports, [])


def reader(reports):
    device = object.__new__(DualSenseDevice)
    device.device, device.last, device.reports = FakeHid(reports), None, 0
    return device


def test_queued_reports_drained_to_newest(monkeypatch):
    monkeypatch.setattr("embodirun_xlerobot_owner.dualsense_input.time.monotonic", lambda: 10)
    device = reader([simple_report(1), simple_report(2)])
    assert device.poll().sequence == 2
    assert device.reports == 2


def test_disconnect_and_duplicate_reports_cannot_extend_motion(monkeypatch):
    device = reader([simple_report(1)])
    monkeypatch.setattr("embodirun_xlerobot_owner.dualsense_input.time.monotonic", lambda: 10)
    device.poll()
    device.device = FakeHid([simple_report(1)])
    monkeypatch.setattr("embodirun_xlerobot_owner.dualsense_input.time.monotonic", lambda: 10.3)
    with pytest.raises(RuntimeError, match="过期"):
        device.poll()


def test_input_flood_rejected(monkeypatch):
    monkeypatch.setattr("embodirun_xlerobot_owner.dualsense_input.time.monotonic", lambda: 10)
    with pytest.raises(RuntimeError, match="积压"):
        reader([simple_report(i % 64) for i in range(256)]).poll()


def test_linux_enumeration_only_selects_exact_dualsense_vid_pid(tmp_path):
    for i, hid_id in enumerate(
        ("0005:0000054C:00000CE6", "0003:0000054C:00000CE6", "0005:0000054C:00000DF2", "invalid")
    ):
        node = tmp_path / f"hidraw{i}" / "device"
        node.mkdir(parents=True)
        (node / "uevent").write_text(f"HID_ID={hid_id}\nHID_NAME=DualSense\n")
    found = linux_devices(tmp_path)
    assert [d["transport"] for d in found] == ["bluetooth", "usb"]
    assert [d["path"] for d in found] == ["/dev/hidraw0", "/dev/hidraw1"]


def test_real_bluetooth_sample_from_agx():
    sample = decode_report(bytes.fromhex("017a7d877908001c0000"), 10)
    assert sample.raw_axes == (122, 125, 135, 121)
    assert sample.neutral
    assert sample.sequence == 7


def sample(buttons=(), sticks=(0, 0, 0, 0)):
    neutral = decode_report(bytes.fromhex("017a7d877908001c0000"), 10)
    return replace(neutral, buttons=frozenset(buttons), sticks=sticks)


def metadata(**changes):
    return {
        "allow_motion": True,
        "enable_base": True,
        "control_scopes": ["arms", "base"],
        "base_velocity_limits": {"linear_m_s": 0.15, "angular_deg_s": 30},
        **changes,
    }


def test_startup_requires_release_then_neutral_r1_edge():
    mapper = DriveMapping(metadata())
    assert not mapper.update(sample(["r1"]))[0]
    assert not mapper.update(sample())[0]
    assert mapper.update(sample(["r1"])) == (True, ZERO)
    assert not mapper.update(sample(["r1"]))[0]


def test_takeover_defaults_to_tethered_without_changing_ordinary_manual_mode():
    assert resolve_motion_profile(takeover=True, profile=None) == "tethered"
    assert resolve_motion_profile(takeover=False, profile=None) == "free"
    mapper = DriveMapping(metadata(), motion_profile="tethered")
    assert mapper.speed == (TETHERED_MAX_LINEAR_M_S, 0.0)
    assert TETHERED_MAX_HOLD_S == 2.0
    mapper.update(sample())
    _, action = mapper.update(sample(["r1"], (0, -1, -1, -1)))
    assert action == {"x.vel": TETHERED_MAX_LINEAR_M_S, "theta.vel": 0.0}
    mapper.require_new_enable_edge()
    assert not mapper.update(sample(["r1"]))[0]
    mapper.update(sample())
    assert mapper.update(sample(["r1"]))[0]


def test_moving_stick_cannot_enable():
    mapper = DriveMapping(metadata())
    mapper.update(sample())
    assert not mapper.update(sample(["r1"], (0, -1, 0, 0)))[0]
    assert not mapper.update(sample(["r1"]))[0]
    mapper.update(sample())
    assert mapper.update(sample(["r1"]))[0]


@pytest.mark.parametrize(
    "sticks,expected",
    [
        ((0, -1, 0, 0), {"x.vel": 0.05, "theta.vel": 0}),
        ((0, 1, 0, 0), {"x.vel": -0.05, "theta.vel": 0}),
        ((0, 0, -1, 0), {"x.vel": 0, "theta.vel": 10}),
        ((0, 0, 1, 0), {"x.vel": 0, "theta.vel": -10}),
        ((0, -0.5, 0.25, 0), {"x.vel": 0.025, "theta.vel": -2.5}),
    ],
)
def test_axes_signs_analog_speed_and_release(sticks, expected):
    mapper = DriveMapping(metadata())
    assert mapper.update(sample(["r1"], sticks))[1] == expected
    assert mapper.update(sample([], sticks))[1] == ZERO
    assert mapper.update(sample(["r1", "cross"], sticks))[1] == ZERO
    assert mapper.update(sample(["r1", "circle"], sticks))[1] == ZERO


def test_speed_changes_only_on_stopped_dpad_edges_and_respects_hardware_limits():
    mapper = DriveMapping(metadata(base_velocity_limits={"linear_m_s": 0.12, "angular_deg_s": 25}))
    mapper.update(sample(["up"]))
    assert mapper.speed == (0.1, 20)
    mapper.update(sample(["up"]))
    assert mapper.level == 2
    mapper.update(sample())
    mapper.update(sample(["up"]))
    assert mapper.speed == (0.12, 25)
    mapper.update(sample())
    mapper.update(sample(["r1", "down"]))
    assert mapper.level == 3
    mapper.update(sample())
    mapper.update(sample(["down"], (0, -1, 0, 0)))
    assert mapper.level == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"allow_motion": False},
        {"enable_base": False},
        {"control_scopes": ["arms"]},
        {"base_velocity_limits": {}},
        {"base_velocity_limits": {"linear_m_s": float("nan"), "angular_deg_s": 30}},
        {"base_velocity_limits": {"linear_m_s": True, "angular_deg_s": 30}},
    ],
)
def test_invalid_or_disabled_service_cannot_control(changes):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        DriveMapping(metadata(**changes))


class FakeRobot:
    def __init__(self, arm_error=None, stop_confirmed=True):
        self.arm_error, self.stop_confirmed = arm_error, stop_confirmed
        self.stops = 0
        self.takeovers = 0
        self.stop_errors = []
        self.actions = []

    def arm(self):
        if self.arm_error:
            raise self.arm_error
        return {"armed": True, "held_positions": {}}

    def command(self, action):
        self.actions.append(action)
        return {"accepted": True, "applied_action": action}

    def release(self):
        self.stops += 1
        return {"stop_confirmed": self.stop_confirmed}

    def stop(self):
        self.takeovers += 1
        return {"scope": "base", "stop_confirmed": self.stop_confirmed, "errors": list(self.stop_errors)}


def test_explicit_takeover_preempts_base_only_and_requires_confirmation():
    robot = FakeRobot()
    session = BaseSession(robot)
    result = session.takeover()
    assert result["scope"] == "base"
    assert robot.takeovers == 1 and robot.stops == 0
    assert not session.owns

    robot.stop_confirmed = False
    with pytest.raises(RuntimeError, match="停止未确认"):
        session.takeover()
    assert robot.takeovers == 2 and robot.stops == 0

    robot.stop_confirmed = True
    robot.stop_errors = ["feedback stale"]
    with pytest.raises(RuntimeError, match="停止未确认"):
        session.takeover()


class RunRobot(FakeRobot):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.metadata = metadata()
        self.arm_calls = 0

    def connect(self):
        return None

    def arm(self):
        self.arm_calls += 1
        return super().arm()


class RunDevice:
    def __init__(self, reports):
        self.reports = iter(reports)

    def poll(self):
        return next(self.reports)

    def close(self):
        return None


def run_args(token_file: Path, *, takeover=True, motion_profile=None):
    return SimpleNamespace(
        robot_url="http://127.0.0.1:8766",
        token_file=token_file,
        output=None,
        task="test",
        max_seconds=10,
        takeover=takeover,
        motion_profile=motion_profile,
    )


def test_run_takeover_waits_for_explicit_r1_before_preempt(monkeypatch, tmp_path):
    robot = RunRobot()
    device = RunDevice([sample(["cross"])])
    fresh = iter([sample()])
    monkeypatch.setattr(dualsense_drive, "RemoteRobot", lambda *args, **kwargs: robot)
    monkeypatch.setattr(dualsense_drive, "DualSenseDevice", lambda: device)
    monkeypatch.setattr(dualsense_drive, "fresh_input", lambda _: next(fresh))
    token = tmp_path / "robot-token"
    token.write_text("test-token")

    dualsense_drive.run(run_args(token))

    assert robot.takeovers == 0
    assert robot.arm_calls == 0
    assert robot.stops == 0


def test_run_takeover_preempts_once_and_tethered_hold_requires_repress(monkeypatch, tmp_path):
    robot = RunRobot()
    polls = (
        [sample(), sample(["r1"])]
        + [sample(["r1"], (0, -1, 0, 0)) for _ in range(60)]
        + [sample(), sample(["r1"]), sample(["cross"])]
    )
    device = RunDevice(polls)
    fresh = iter([sample(), sample(["r1"]), sample(["r1"]), sample(), sample(["r1"])])
    clock = iter(i * 0.05 for i in range(1000))
    monkeypatch.setattr(dualsense_drive, "RemoteRobot", lambda *args, **kwargs: robot)
    monkeypatch.setattr(dualsense_drive, "DualSenseDevice", lambda: device)
    monkeypatch.setattr(dualsense_drive, "fresh_input", lambda _: next(fresh))
    monkeypatch.setattr(dualsense_drive.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(dualsense_drive.time, "sleep", lambda _: None)
    token = tmp_path / "robot-token"
    token.write_text("test-token")

    dualsense_drive.run(run_args(token))

    assert robot.takeovers == 1
    assert robot.arm_calls == 2
    assert robot.stops == 2
    assert robot.actions
    assert all(
        abs(action["x.vel"]) <= TETHERED_MAX_LINEAR_M_S and action["theta.vel"] == 0.0 for action in robot.actions
    )


def test_session_stop_without_ownership_does_not_stop_keyboard_or_arm():
    robot = FakeRobot(arm_error=RuntimeError("robot already controlled"))
    session = BaseSession(robot)
    with pytest.raises(RuntimeError):
        session.enable()
    session.stop()
    assert robot.stops == 0


def test_uncertain_enable_timeout_requests_stop():
    robot = FakeRobot(arm_error=TimeoutError("response lost"))
    session = BaseSession(robot)
    with pytest.raises(TimeoutError):
        session.enable()
    session.stop()
    assert robot.stops == 1


def test_session_commands_require_enable_and_confirmed_stop():
    robot = FakeRobot()
    session = BaseSession(robot)
    with pytest.raises(RuntimeError):
        session.send(ZERO)
    session.enable()
    session.send(ZERO)
    session.stop()
    assert robot.actions == [ZERO] and robot.stops == 1
    assert not session.owns
    session.enable()
    robot.stop_confirmed = False
    with pytest.raises(RuntimeError, match="停止未确认"):
        session.stop()


@pytest.mark.parametrize(
    "feedback",
    [
        {},
        {"accepted": False},
        {"accepted": True, "applied_action": {"x.vel": 0}},
        {"accepted": True, "applied_action": {"x.vel": float("nan"), "theta.vel": 0}},
        {"accepted": True, "applied_action": {**ZERO, "left_arm_shoulder_pan.pos": 0}},
    ],
)
def test_command_rejection_missing_ack_and_arm_targets_rejected(feedback):
    with pytest.raises(RuntimeError):
        check_feedback(feedback)


def test_camera_tilt_is_right_y_and_does_not_change_turn_axis():
    head = HeadTiltMapping(
        metadata(
            head_tilt={"joint": "head_motor_2.pos", "scope": "base", "max_speed_deg_s": 15, "up_sign": 1},
            joint_limits={"head_motor_2.pos": [-40, 40]},
        )
    )
    head.reset({"held_positions": {"head_motor_2.pos": 20}})
    moving = sample(["r1"], (0, 0, 0.5, -1))
    assert head.action(moving, 0.05) == {"head_motor_2.pos": 20.75}
    assert DriveMapping(metadata()).update(moving)[1]["theta.vel"] == -5
    head.accept({"head_motor_2.pos": 20.5})
    assert head.action(sample([], (0, 0, 0, -1)), 0.05) == {"head_motor_2.pos": 20.5}
    head.target = 39.9
    assert head.action(moving, 0.05) == {"head_motor_2.pos": 40}
    assert HeadTiltMapping(metadata()).action(moving, 0.05) == {}


def test_release_cannot_stop_replacement_owner():
    robot = FakeRobot()
    session = BaseSession(robot)
    session.enable()
    robot.release = lambda: {"released": False, "control_owned": False}
    session.stop()
    assert robot.stops == 0
    assert not session.owns
