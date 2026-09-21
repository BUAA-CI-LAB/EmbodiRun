import copy
import math

import pytest

from embodirun_xlerobot_owner.control import (
    InputClock,
    InputFrame,
    MappingConfig,
    QuestMapper,
    SO101Kinematics,
    relative_wrist_angles,
)
from embodirun_xlerobot_owner.robot import DemoRobot


def input_data(seq=0, ms=0, grip=False):
    c = {
        "position": [0, 1, -0.5],
        "orientation": [0, 0, 0, 1],
        "tracked": True,
        "grip": grip,
        "trigger": 0.0,
        "thumbstick": [0, 0],
    }
    return {
        "type": "input",
        "seq": seq,
        "timestamp_ms": ms,
        "controllers": {"left": copy.deepcopy(c), "right": copy.deepcopy(c)},
    }


@pytest.mark.parametrize("shoulder,elbow", [(0, 0), (-30, 20), (25, 40), (60, 60)])
def test_kinematic_roundtrip(shoulder, elbow):
    kin = SO101Kinematics()
    assert kin.inverse(*kin.forward(shoulder, elbow)) == pytest.approx((shoulder, elbow))


def test_workspace_unreachable_is_rejected():
    with pytest.raises(ValueError, match="workspace"):
        SO101Kinematics().inverse(1, 0)


def test_relative_wrist_rotation_does_not_depend_on_initial_world_yaw():
    pitch = math.radians(20) / 2
    yaw = math.radians(90) / 2
    # current = initial yaw * local pitch. World Euler subtraction would
    # mix axes; the gripped controller frame must still report pitch=20.
    reference = (0, math.sin(yaw), 0, math.cos(yaw))
    current = (
        math.cos(yaw) * math.sin(pitch),
        math.sin(yaw) * math.cos(pitch),
        -math.sin(yaw) * math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )
    assert relative_wrist_angles(current, reference) == pytest.approx((20, 0))
    assert relative_wrist_angles(tuple(-v for v in current), reference) == pytest.approx((20, 0))


def test_clutch_preserves_initial_pose_and_trigger():
    state = DemoRobot().state
    mapper = QuestMapper()
    data = input_data(grip=True)
    data["controllers"]["left"]["trigger"] = 1
    result = mapper.map(InputFrame.parse(data), state, 0.05)
    assert result == pytest.approx(state)
    data["controllers"]["left"]["position"][1] += 0.01
    result = mapper.map(InputFrame.parse(data), state, 0.05)
    assert result["left_arm_shoulder_lift.pos"] != 0
    assert result["right_arm_shoulder_lift.pos"] == pytest.approx(0)
    assert result["left_arm_gripper.pos"] == 50
    assert max(abs(result[k] - state[k]) for k in state) <= 0.75


def test_regrip_anchors_to_measured_position_without_jump():
    state = DemoRobot().state
    mapper = QuestMapper()
    data = input_data(grip=True)
    mapper.map(InputFrame.parse(data), state, 0.05)
    data["controllers"]["left"]["grip"] = False
    data["controllers"]["left"]["position"][0] = 1.5
    mapper.map(InputFrame.parse(data), state, 0.05)
    data["controllers"]["left"]["grip"] = True
    result = mapper.map(InputFrame.parse(data), state, 0.05)
    assert result == pytest.approx(state)


def test_controller_relocalization_stops_mapping():
    mapper = QuestMapper()
    data = input_data(grip=True)
    mapper.map(InputFrame.parse(data), DemoRobot().state, 0.05)
    data["controllers"]["left"]["position"][0] = 2
    with pytest.raises(ValueError, match="jumped"):
        mapper.map(InputFrame.parse(data), DemoRobot().state, 0.05)


def test_base_is_two_wheel_explicit_enable_and_deadman():
    data = input_data()
    data["controllers"]["right"]["thumbstick"] = [1, -1]
    frame, state = InputFrame.parse(data), DemoRobot().state
    assert QuestMapper().map(frame, state, 0.05)["x.vel"] == 0
    cfg = MappingConfig(enable_base=True)
    mapper = QuestMapper(cfg)
    assert mapper.map(frame, state, 0.05)["x.vel"] == 0
    mapper.set_control_mode("drive")
    assert mapper.map(frame, state, 0.05)["x.vel"] == 0
    data["controllers"]["right"]["grip"] = True
    action = mapper.map(InputFrame.parse(data), state, 0.05)
    assert action["x.vel"] == cfg.max_linear_m_s
    assert action["theta.vel"] == -cfg.max_angular_deg_s
    assert "y.vel" not in action


def test_drive_holds_arm_pose_and_grippers_while_sticks_drive():
    mapper = QuestMapper(MappingConfig(enable_base=True))
    mapper.set_control_mode("drive")
    data, state = input_data(grip=True), DemoRobot().state
    mapper.map(InputFrame.parse(data), state, 0.05)
    for side in ("left", "right"):
        data["controllers"][side]["position"] = [3, 2, 1]
        data["controllers"][side]["trigger"] = 1
        data["controllers"][side]["orientation"] = [0, 0, 1, 0]
    data["controllers"]["right"]["thumbstick"] = [-1, 1]
    sagged = {name: value + 0.1 for name, value in state.items()}
    result = mapper.map(InputFrame.parse(data), sagged, 0.05)
    assert all(result[name] == state[name] for name in state if name.endswith(".pos"))
    assert result["x.vel"] == -0.05
    assert result["theta.vel"] == 10
    data["controllers"]["right"]["grip"] = False
    stopped = mapper.map(InputFrame.parse(data), state, 0.05)
    assert stopped["x.vel"] == stopped["theta.vel"] == 0
    data["controllers"]["right"]["grip"] = True
    data["controllers"]["right"]["thumbstick"] = [0.1, -0.1]
    centered = mapper.map(InputFrame.parse(data), state, 0.05)
    assert centered["x.vel"] == centered["theta.vel"] == 0


def test_arm_mode_never_drives_and_drive_switch_clears_pose_anchors():
    mapper = QuestMapper(MappingConfig(enable_base=True))
    data, state = input_data(grip=True), DemoRobot().state
    data["controllers"]["right"]["thumbstick"] = [1, -1]
    result = mapper.map(InputFrame.parse(data), state, 0.05)
    assert mapper.anchors
    assert result["x.vel"] == result["theta.vel"] == 0
    mapper.set_control_mode("drive")
    assert not mapper.anchors
    mapper.map(InputFrame.parse(data), state, 0.05)
    mapper.set_control_mode("arms")
    assert not mapper.drive_hold
    data["controllers"]["right"]["position"][0] = 0.5
    result = mapper.map(InputFrame.parse(data), state, 0.05)
    assert result == pytest.approx(state)
    with pytest.raises(ValueError, match="disabled"):
        QuestMapper().set_control_mode("drive")
    with pytest.raises(ValueError, match="must be"):
        mapper.set_control_mode("both")


def test_input_clock_rejects_backlog_reorder_and_reset():
    clock = InputClock()
    clock.accept(InputFrame.parse(input_data(1, 100)), 10)
    with pytest.raises(ValueError, match="reordered"):
        clock.accept(InputFrame.parse(input_data(1, 110)), 10.01)
    with pytest.raises(ValueError, match="delayed"):
        clock.accept(InputFrame.parse(input_data(2, 120)), 10.9)


def test_input_clock_drop_does_not_rebase_old_packets_as_fresh():
    clock = InputClock()
    assert clock.accept(InputFrame.parse(input_data(1, 100)), 10) == 10
    for seq, stamp in ((2, 120), (3, 130)):
        with pytest.raises(ValueError, match="delayed"):
            clock.accept(InputFrame.parse(input_data(seq, stamp)), 10.9)
    with pytest.raises(ValueError, match="reordered"):
        clock.accept(InputFrame.parse(input_data(3, 990)), 10.91)
    assert clock.accept(InputFrame.parse(input_data(4, 1000)), 10.91) == pytest.approx(10.9)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_invalid_input_never_reaches_mapping(value):
    data = input_data()
    data["controllers"]["right"]["position"][1] = value
    with pytest.raises((ValueError, TypeError)):
        InputFrame.parse(data)


def test_lost_tracking_stops_and_clears_anchor():
    mapper = QuestMapper()
    data = input_data(grip=True)
    mapper.map(InputFrame.parse(data), DemoRobot().state, 0.05)
    data["controllers"]["left"]["tracked"] = False
    with pytest.raises(ValueError, match="tracking"):
        mapper.map(InputFrame.parse(data), DemoRobot().state, 0.05)
    assert mapper.anchors == {}
