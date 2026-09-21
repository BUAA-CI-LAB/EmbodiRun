from dataclasses import replace

import pytest

from embodirun_xlerobot_owner.control import MappingConfig, QuestMapper

from .test_mapping_regression import _frame, _state


def packet(seq, side, stick, *, grip=True, position=(0, 1, -0.5)):
    frame = _frame(seq)
    return replace(
        frame,
        controllers={
            **frame.controllers,
            side: replace(frame.controllers[side], grip=grip, thumbstick=stick, position=position),
        },
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_stick_turns_only_own_bottom_joint_and_center_holds(side, direction):
    state = _state()
    mapper = QuestMapper()
    mapper.map(packet(0, side, (0, 0)), state, 0.05)
    name = f"{side}_arm_shoulder_pan.pos"
    for seq in range(1, 5):
        action = mapper.map(packet(seq, side, (direction, 0)), state, 0.05)
        assert action[name] == pytest.approx(seq * direction * 15 * 0.05)
        assert all(action[k] == v for k, v in state.items() if k != name)
        assert action["x.vel"] == action["theta.vel"] == 0
        assert mapper.diagnostics[side]["pan_source"] == "thumbstick"
    centered = mapper.map(packet(5, side, (0, 0)), state, 0.05)
    assert centered == action
    released = mapper.map(packet(6, side, (-direction, 0), grip=False), state, 0.05)
    assert released == action


@pytest.mark.parametrize("stick", [(0, 1), (0, -1), (0.15, 0), (-0.15, 0)])
def test_vertical_stick_and_deadband_cannot_move_arms(stick):
    state = _state()
    mapper = QuestMapper()
    mapper.map(packet(0, "left", (0, 0)), state, 0.05)
    result = mapper.map(packet(1, "left", stick), state, 0.05)
    assert all(result[k] == v for k, v in state.items())


def test_stick_priority_consumes_lateral_hand_motion_without_return_jump():
    state = _state()
    mapper = QuestMapper()
    mapper.map(packet(0, "left", (0, 0)), state, 0.05)
    position = (-0.02, 1, -0.5)
    moved = mapper.map(packet(1, "left", (1, 0), position=position), state, 0.05)
    assert moved["left_arm_shoulder_pan.pos"] == pytest.approx(0.75)
    held = mapper.map(packet(2, "left", (0, 0), position=position), state, 0.05)
    assert held == moved
    follow = mapper.map(packet(3, "left", (0, 0), position=(-0.018, 1, -0.5)), state, 0.05)
    assert follow["left_arm_shoulder_pan.pos"] == pytest.approx(0.99)


def test_stick_does_not_override_height_or_pitch():
    state = _state()
    with_stick, without_stick = QuestMapper(), QuestMapper()
    for mapper in (with_stick, without_stick):
        mapper.map(packet(0, "left", (0, 0)), state, 0.05)
    location = (0, 1.001, -0.5)
    a = with_stick.map(packet(1, "left", (1, 0), position=location), state, 0.05)
    b = without_stick.map(packet(1, "left", (0, 0), position=location), state, 0.05)
    for name in state:
        if name != "left_arm_shoulder_pan.pos":
            assert a[name] == b[name]


def test_stick_obeys_per_joint_limits_speed_and_dt_cap():
    state = _state()
    mapper = QuestMapper(MappingConfig(thumbstick_pan_deg_s=120, max_joint_speed_deg_s=15))
    mapper.map(packet(0, "left", (0, 0)), state, 0.05)
    name = "left_arm_shoulder_pan.pos"
    moved = mapper.map(packet(1, "left", (1, 0)), state, 1)
    assert moved[name] == pytest.approx(1.5)
    bounded = mapper.map(packet(2, "left", (1, 0)), state, 0.05, {name: [-2, 1.6]})
    assert bounded[name] == pytest.approx(1.6)
    still = mapper.map(packet(3, "left", (1, 0)), state, 0.05, {name: [-2, 1.6]})
    assert still[name] == pytest.approx(1.6)


def test_stick_half_deflection_and_mirrored_pan_sign():
    state = _state()
    mapper = QuestMapper(MappingConfig(pan_signs={"left": -1, "right": 1}))
    mapper.map(packet(0, "left", (0, 0)), state, 0.05)
    moved = mapper.map(packet(1, "left", (0.575, 0)), state, 0.05)
    assert moved["left_arm_shoulder_pan.pos"] == pytest.approx(-0.375)
