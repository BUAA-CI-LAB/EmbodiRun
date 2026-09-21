from dataclasses import replace

import pytest

from embodirun_xlerobot_owner.control import CAMERAS, JOINT_NAMES, MappingConfig
from embodirun_xlerobot_owner.joycon_input import (
    JoyconDevice,
    JoyconSample,
    decode_report,
    normalize_axis,
    resting_centre,
)
from embodirun_xlerobot_owner.joycon_teleop import EnableGesture, JoyconMapper, validate_observation


def sample(side, buttons=(), stick=(0.0, 0.0), raw=(2048, 2048), at=10):
    return JoyconSample(side, raw, frozenset(buttons), 4, at, 0, stick)


def pair(**changes):
    return {side: sample(side, **changes.get(side, {})) for side in ("left", "right")}


def state_and_limits():
    state = {name: 50 if name.endswith("gripper.pos") else 0 for name in JOINT_NAMES}
    limits = {name: [0, 100] if name.endswith("gripper.pos") else [-120, 120] for name in JOINT_NAMES}
    return state, limits


@pytest.mark.parametrize(
    "side,offset,button_byte,buttons",
    [
        ("left", 6, 5, {"l", "zl"}),
        ("right", 9, 3, {"r", "zr"}),
    ],
)
def test_hid_report_decodes_distinct_left_right_fields(side, offset, button_byte, buttons):
    report = bytearray(49)
    report[0:3] = bytes([0x30, 67, 0x80])
    # Non-centred values catch swapped nibbles and left/right offsets.
    x, y = 1734, 2611
    report[offset : offset + 3] = bytes([x & 255, (x >> 8) | ((y & 15) << 4), y >> 4])
    report[button_byte] = 0xC0
    result = decode_report(report, side, 12.5)
    assert result.raw_stick == (x, y)
    assert result.buttons == buttons
    assert result.battery_level == 4
    assert result.received_at == 12.5


@pytest.mark.parametrize("report", [b"", b"\x30" * 12, b"\x3f" * 49])
def test_short_or_wrong_report_cannot_be_control_input(report):
    with pytest.raises(ValueError):
        decode_report(report, "left", 0)


def test_actual_resting_centres_and_deadzone_prevent_author_controller_drift():
    samples = [sample("left", raw=(1900 + i % 3, 2300)) for i in range(30)]
    centre = resting_centre(samples)
    assert centre == (1901, 2300)
    assert normalize_axis(centre[0] + 30, centre[0]) == 0
    assert normalize_axis(centre[0] - 1000, centre[0]) == -1
    assert normalize_axis(centre[0] + 2000, centre[0]) == 1
    samples[-1] = replace(samples[-1], raw_stick=(2700, 2300))
    with pytest.raises(RuntimeError, match="移动"):
        resting_centre(samples)


def test_disconnection_does_not_keep_last_stick_command_alive():
    reader = object.__new__(JoyconDevice)
    reader.side, reader.centre = "left", (2048, 2048)
    reader.last = sample("left", stick=(1, 0), at=10)

    class EmptyDevice:
        def read(self, size, timeout):
            return []

    reader.device = EmptyDevice()
    with pytest.raises(RuntimeError, match="过期"):
        reader.poll(now=10.3)


def test_duplicate_device_timer_does_not_refresh_stale_input(monkeypatch):
    reader = object.__new__(JoyconDevice)
    reader.side, reader.centre = "left", None
    report = bytearray(49)
    report[0] = 0x30
    reader.last = decode_report(report, "left", 10)

    class RepeatedDevice:
        def __init__(self):
            self.reports = [report, []]

        def read(self, size, timeout):
            return self.reports.pop(0)

    reader.device = RepeatedDevice()
    monkeypatch.setattr("embodirun_xlerobot_owner.joycon_input.time.monotonic", lambda: 10.4)
    with pytest.raises(RuntimeError, match="过期"):
        reader.poll()


def test_enable_requires_released_then_held_neutral_chord():
    gesture = EnableGesture()
    chord = pair(left={"buttons": ["minus"]}, right={"buttons": ["plus"]})
    assert not gesture.update(chord, 0)
    assert not gesture.update(chord, 2)
    assert not gesture.update(pair(), 3)
    assert not gesture.update(chord, 4)
    assert gesture.update(chord, 5.1)
    assert not gesture.update(chord, 7)
    gesture.update(pair(), 8)
    moving = dict(chord, left=replace(chord["left"], stick=(0.5, 0)))
    assert not gesture.update(moving, 9)
    assert not gesture.update(moving, 11)


def test_startup_and_neutral_hold_do_not_zero_or_follow_sag():
    state, limits = state_and_limits()
    state["left_arm_shoulder_pan.pos"] = 27
    mapper = JoyconMapper()
    assert mapper.map(pair(), state, 0.05, limits) == state
    sagged = dict(state, **{"left_arm_shoulder_pan.pos": 25})
    assert mapper.map(pair(), sagged, 0.05, limits) == state
    assert not ({"x.vel", "theta.vel"} & mapper.targets.keys())


def test_pan_limit_does_not_block_other_arm_and_release_does_not_queue_motion():
    state, limits = state_and_limits()
    state["left_arm_shoulder_pan.pos"] = 120
    mapper = JoyconMapper(MappingConfig(max_joint_speed_deg_s=10))
    action = mapper.map(pair(left={"stick": (1, 0)}, right={"stick": (1, 0)}), state, 0.05, limits)
    assert action["left_arm_shoulder_pan.pos"] == 120
    assert 0 < action["right_arm_shoulder_pan.pos"] <= 0.5
    assert mapper.map(pair(), state, 0.05, limits) == action


def test_reach_uses_existing_ik_and_all_joints_remain_speed_limited():
    state, limits = state_and_limits()
    action = JoyconMapper().map(pair(left={"stick": (0, 1)}), state, 0.05, limits)
    assert action["left_arm_shoulder_lift.pos"] != state["left_arm_shoulder_lift.pos"]
    assert max(abs(action[name] - state[name]) for name in JOINT_NAMES) <= 0.5 + 1e-9


def test_trigger_hold_moves_gripper_release_holds_next_press_reverses():
    state, limits = state_and_limits()
    mapper = JoyconMapper()
    closing = mapper.map(pair(left={"buttons": ["zl"]}), state, 0.05, limits)
    assert closing["left_arm_gripper.pos"] == 48.75
    assert mapper.map(pair(), state, 0.05, limits) == closing
    opening = mapper.map(pair(left={"buttons": ["zl"]}), state, 0.05, limits)
    assert opening["left_arm_gripper.pos"] == 50


def test_mapper_follows_applied_quantized_target_and_rejects_missing_ack():
    state, limits = state_and_limits()
    mapper = JoyconMapper()
    mapper.map(pair(right={"stick": (1, 0)}), state, 0.05, limits)
    mapper.accept({"applied_action": state})
    assert mapper.map(pair(), state, 0.05, limits) == state
    with pytest.raises(RuntimeError, match="回执"):
        mapper.accept({"applied_action": {}})
    with pytest.raises(RuntimeError, match="拒绝"):
        mapper.accept({"accepted": False})
    with pytest.raises(RuntimeError, match="delayed"):
        mapper.map(pair(), state, 1, limits)


def test_stale_camera_state_or_units_refuse_enable_and_control():
    state, limits = state_and_limits()
    metadata = {
        "joint_unit": "degrees",
        "gripper_unit": "range_0_100",
        "joint_limits": limits,
        "camera_roles_confirmed": True,
    }
    observation = {
        "state": state,
        "source_timestamp_ns": 1_000_000_000,
        "state_timestamp_ns": 1_000_000_000,
        "camera_timestamps_ns": dict.fromkeys(CAMERAS, 1_000_000_000),
    }
    images = dict.fromkeys(CAMERAS, b"jpeg")
    assert validate_observation(observation, images, metadata) == 1_000_000_000
    with pytest.raises(RuntimeError, match="更新"):
        validate_observation(observation, images, metadata, 1_000_000_000)
    with pytest.raises(RuntimeError, match="反馈"):
        validate_observation(dict(observation, state_timestamp_ns=1), images, metadata)
    with pytest.raises(RuntimeError, match="相机"):
        validate_observation(observation, {}, metadata)
    with pytest.raises(RuntimeError, match="degrees"):
        validate_observation(observation, images, dict(metadata, joint_unit="range_m100_100"))
