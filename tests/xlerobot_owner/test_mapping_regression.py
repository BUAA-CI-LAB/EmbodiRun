from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest

from embodirun_xlerobot_owner.control import InputFrame, MappingConfig, QuestMapper, SO101Kinematics

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_JOINTS = ARM_JOINTS + ("gripper",)


def _state(**overrides: float) -> dict[str, float]:
    state = {
        f"{side}_arm_{joint}.pos": (50.0 if joint == "gripper" else 0.0)
        for side in ("left", "right")
        for joint in ALL_JOINTS
    }
    state.update(overrides)
    return state


def _frame(
    seq: int,
    *,
    left_grip: bool = False,
    left_trigger: float = 0.0,
    left_direction: str = "close",
    left_position: tuple[float, float, float] = (0.0, 1.0, -0.5),
    left_tracked: bool = True,
) -> InputFrame:
    controller = {
        "position": list(left_position),
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "tracked": left_tracked,
        "grip": left_grip,
        "trigger": left_trigger,
        "gripper_direction": left_direction,
        "thumbstick": [0.0, 0.0],
    }
    return InputFrame.parse(
        {
            "type": "input",
            "seq": seq,
            "timestamp_ms": seq * 50,
            "controllers": {"left": copy.deepcopy(controller), "right": copy.deepcopy(controller)},
        }
    )


def _left(name: str) -> str:
    return f"left_arm_{name}.pos"


def test_inverse_reference_roundtrips_both_elbow_branches() -> None:
    kin = SO101Kinematics()
    for shoulder, elbow in ((30.0, 60.0), (-105.0, -97.0), (-87.120879, 97.054945)):
        x, y = kin.forward(shoulder, elbow)
        assert kin.inverse(x, y, reference=(shoulder, elbow)) == pytest.approx((shoulder, elbow))


def test_fixed_arm_targets_ignore_measured_sag() -> None:
    for gripped in (False, True):
        state = _state(
            **{
                _left("shoulder_pan"): 12.0,
                _left("shoulder_lift"): -35.0,
                _left("elbow_flex"): 48.0,
                _left("wrist_flex"): 22.0,
                _left("wrist_roll"): -17.0,
            }
        )
        mapper = QuestMapper()
        mapper.map(_frame(0, left_grip=gripped), state, 0.05)
        sagged = dict(state)
        for joint in ARM_JOINTS:
            sagged[_left(joint)] += 0.2

        result = mapper.map(_frame(1, left_grip=gripped), sagged, 0.05)
        for joint in ARM_JOINTS:
            assert result[_left(joint)] == pytest.approx(state[_left(joint)])


def test_stationary_negative_branch_grip_has_no_drift_for_100_frames() -> None:
    state = _state(
        **{
            _left("shoulder_pan"): 8.0,
            _left("shoulder_lift"): 30.0,
            _left("elbow_flex"): -97.0,
            _left("wrist_flex"): 15.0,
            _left("wrist_roll"): -11.0,
            _left("gripper"): 42.0,
        }
    )
    mapper = QuestMapper()
    position = (0.2, 0.8, -0.3)
    for seq in range(101):
        result = mapper.map(
            _frame(seq, left_grip=True, left_trigger=0.42, left_position=position),
            state,
            0.05,
        )
        for joint in ALL_JOINTS:
            assert result[_left(joint)] == pytest.approx(state[_left(joint)])


def test_limited_backward_translation_uses_shared_cartesian_fraction() -> None:
    shoulder, elbow, wrist = -87.120879, 97.054945, 62.285714
    state = _state(
        **{
            _left("shoulder_lift"): shoulder,
            _left("elbow_flex"): elbow,
            _left("wrist_flex"): wrist,
        }
    )
    config = MappingConfig(max_joint_speed_deg_s=2000.0)
    mapper = QuestMapper(config)
    anchor = (0.0, 1.0, -0.5)
    mapper.map(_frame(0, left_grip=True, left_position=anchor), state, 0.05)
    limits = {_left(joint): [-180.0, 180.0] for joint in ALL_JOINTS}
    limits[_left("shoulder_lift")] = [-90.0, 180.0]
    result = mapper.map(
        _frame(1, left_grip=True, left_position=(anchor[0], anchor[1], anchor[2] + 0.02)),
        state,
        0.05,
        limits=limits,
    )

    kin = SO101Kinematics()
    before_x, before_y = kin.forward(shoulder, elbow)
    after_x, after_y = kin.forward(result[_left("shoulder_lift")], result[_left("elbow_flex")])
    assert result[_left("shoulder_lift")] >= -90.0
    assert after_x <= before_x + 1e-7
    assert after_y == pytest.approx(before_y, abs=1e-6)
    assert (
        result[_left("shoulder_lift")] + result[_left("elbow_flex")] + result[_left("wrist_flex")]
    ) == pytest.approx(shoulder + elbow + wrist, abs=1e-6)


def test_trigger_closes_release_holds_and_selected_direction_opens_at_rate() -> None:
    gripper = _left("gripper")
    state = _state(**{gripper: 1.16})
    mapper = QuestMapper(MappingConfig(max_gripper_speed_pct_s=25.0))

    held = mapper.map(_frame(0, left_grip=True, left_trigger=0.0), state, 0.05)
    assert held[gripper] == pytest.approx(1.16)

    closed = mapper.map(_frame(1, left_grip=True, left_trigger=1.0), state, 0.05)
    assert closed[gripper] == pytest.approx(0.0)

    for seq in range(2, 6):
        held = mapper.map(_frame(seq, left_grip=True, left_trigger=0), state, 0.05)
        assert held[gripper] == 0  # measured sag must not change the hold target
    switched = mapper.map(_frame(6, left_grip=True, left_direction="open"), state, 0.05)
    assert switched[gripper] == 0
    expected = 0.0
    for seq in range(7, 11):
        opened = mapper.map(_frame(seq, left_grip=True, left_trigger=1, left_direction="open"), state, 0.05)
        expected += 25.0 * 0.05
        assert opened[gripper] == pytest.approx(expected)
    released = mapper.map(_frame(11, left_trigger=1, left_direction="open"), state, 0.05)
    assert released[gripper] == expected  # trigger without Grip cannot move


def test_direction_toggle_with_held_trigger_needs_release_and_resqueeze():
    mapper = QuestMapper(MappingConfig(max_gripper_speed_pct_s=25))
    state = _state()
    grip = _left("gripper")
    mapper.map(_frame(0, left_grip=True), state, 0.05)
    closed = mapper.map(_frame(1, left_grip=True, left_trigger=1), state, 0.05)[grip]
    for seq in (2, 3):
        switched = mapper.map(_frame(seq, left_grip=True, left_trigger=1, left_direction="open"), state, 0.05)
        assert switched[grip] == closed
        assert not mapper.diagnostics["left"]["trigger_ready"]
    mapper.map(_frame(4, left_grip=True, left_direction="open"), state, 0.05)
    opened = mapper.map(_frame(5, left_grip=True, left_trigger=0.5, left_direction="open"), state, 0.05)
    assert opened[grip] == pytest.approx(closed + 25 * 0.05 * 0.5)


def test_initial_held_trigger_needs_release_before_claw_motion():
    mapper = QuestMapper()
    state = _state()
    for seq in (0, 1):
        held = mapper.map(_frame(seq, left_grip=True, left_trigger=1), state, 0.05)
        assert held[_left("gripper")] == 50
        assert not mapper.diagnostics["left"]["trigger_ready"]


def test_left_and_right_claw_direction_are_independent():
    mapper = QuestMapper()
    state = _state()
    initial = _frame(0, left_grip=True)
    initial = replace(
        initial,
        controllers={
            **initial.controllers,
            "right": replace(initial.controllers["right"], gripper_direction="open"),
        },
    )
    mapper.map(initial, state, 0.05)
    squeezed = replace(
        initial,
        seq=1,
        timestamp_ms=50,
        controllers={side: replace(c, trigger=1) for side, c in initial.controllers.items()},
    )
    result = mapper.map(squeezed, state, 0.05)
    assert result["left_arm_gripper.pos"] < 50 < result["right_arm_gripper.pos"]


def test_unknown_gripper_direction_is_rejected():
    with pytest.raises(ValueError, match="gripper_direction"):
        _frame(0, left_direction="toggle-all")


def test_hold_uses_quantized_applied_receipt_without_chasing_unsent_targets():
    mapper = QuestMapper()
    state = _state()
    pan = _left("shoulder_pan")
    mapper.map(_frame(0, left_grip=True), state, 0.05)
    moved_frame = _frame(1, left_grip=True, left_position=(0.02, 1, -0.5))
    proposal = mapper.map(moved_frame, state, 0.05)
    assert proposal[pan] > 0.25
    applied = {**proposal, pan: 0.25}
    mapper.accept_applied_action(applied)
    held = mapper.map(moved_frame, state, 0.05)
    assert held[pan] == 0.25  # no motion left in the software queue
    released = mapper.map(_frame(2), state, 0.05)
    assert released[pan] == 0.25  # not stale measurement (zero) or proposal


@pytest.mark.parametrize("direction", [(0, 0, -0.02), (0, 0, 0.02), (0, 0.02, 0), (0, -0.02, 0)])
def test_real_pose_cartesian_step_keeps_direction_limits_and_has_no_queued_motion(direction):
    state = _state(
        **{
            _left("shoulder_lift"): -87.12087912087912,
            _left("elbow_flex"): 97.05494505494505,
            _left("wrist_flex"): 62.285714285714285,
        }
    )
    limits = {
        _left("shoulder_lift"): [-105.58241758241758, 105.58241758241758],
        _left("elbow_flex"): [-97.23076923076923, 97.23076923076923],
        _left("wrist_flex"): [-103.07692307692308, 103.07692307692308],
    }
    mapper = QuestMapper()
    mapper.map(_frame(0, left_grip=True), state, 0.05, limits)
    position = tuple(a + b for a, b in zip((0, 1, -0.5), direction))
    action = mapper.map(_frame(1, left_grip=True, left_position=position), state, 0.05, limits)
    for name in (_left(j) for j in ARM_JOINTS):
        assert abs(action[name] - state[name]) <= 0.750001
        if name in limits:
            assert limits[name][0] <= action[name] <= limits[name][1]
    kin = SO101Kinematics()
    old = kin.forward(state[_left("shoulder_lift")], state[_left("elbow_flex")])
    new = kin.forward(action[_left("shoulder_lift")], action[_left("elbow_flex")])
    request = (-direction[2], direction[1])
    displacement = (new[0] - old[0], new[1] - old[1])
    assert displacement[0] * request[1] - displacement[1] * request[0] == pytest.approx(0, abs=1e-8)
    assert sum(a * b for a, b in zip(request, displacement)) >= -1e-10
    for seq in range(2, 22):
        held = mapper.map(_frame(seq, left_grip=True, left_position=position), state, 0.05, limits)
        assert held == pytest.approx(action)


def _folded_pose():
    limits = {
        _left("shoulder_lift"): [-105.58241758241758, 105.58241758241758],
        _left("elbow_flex"): [-97.23076923076923, 97.23076923076923],
        _left("wrist_flex"): [-103.07692307692308, 103.07692307692308],
    }
    state = _state(
        **{
            _left("shoulder_lift"): limits[_left("shoulder_lift")][0],
            _left("elbow_flex"): 95.3,
            _left("wrist_flex"): 77.3,
        }
    )
    return state, limits


@pytest.mark.parametrize("depth_noise", [0.0002, 0.002])
def test_shoulder_limit_cannot_freeze_independent_pan(depth_noise):
    state, limits = _folded_pose()
    mapper = QuestMapper()
    mapper.map(_frame(0, left_grip=True), state, 0.05, limits)
    frame = _frame(1, left_grip=True, left_position=(0.005, 1, -0.5 + depth_noise))
    result = mapper.map(frame, state, 0.05, limits)
    assert result[_left("shoulder_pan")] == pytest.approx(0.6)
    assert result[_left("shoulder_lift")] == pytest.approx(state[_left("shoulder_lift")])
    diagnostic = mapper.diagnostics["left"]
    if depth_noise < 0.001:
        assert diagnostic["limited"] is False
        assert diagnostic["joint_limit_joints"] == []
    else:
        assert _left("shoulder_lift") in diagnostic["joint_limit_joints"]
    for seq in range(2, 22):
        held = mapper.map(replace(frame, seq=seq), state, 0.05, limits)
        assert held == pytest.approx(result), "blocked motion must not resume on its own"


def test_wrist_limit_does_not_freeze_pan_or_planar_translation():
    state = _state(**{_left("wrist_flex"): 103.0})
    limits = {_left("wrist_flex"): [-103.0, 103.0]}
    mapper = QuestMapper()
    mapper.map(_frame(0, left_grip=True), state, 0.05, limits)
    frame = _frame(1, left_grip=True, left_position=(0.005, 1.002, -0.5))
    result = mapper.map(frame, state, 0.05, limits)
    assert result[_left("shoulder_pan")] == pytest.approx(0.6)
    assert result[_left("wrist_flex")] == 103.0
    kin = SO101Kinematics()
    old = kin.forward(0, 0)
    new = kin.forward(result[_left("shoulder_lift")], result[_left("elbow_flex")])
    assert new[0] == pytest.approx(old[0], abs=1e-8)
    assert new[1] > old[1] + 0.0005
    assert _left("wrist_flex") in mapper.diagnostics["left"]["joint_limit_joints"]
    for joint in ARM_JOINTS:
        assert abs(result[_left(joint)] - state[_left(joint)]) <= 0.750001


def test_wrist_rotation_at_limit_does_not_freeze_other_axes():
    state, limits = _folded_pose()
    state[_left("wrist_flex")] = limits[_left("wrist_flex")][1]
    mapper = QuestMapper()
    mapper.map(_frame(0, left_grip=True), state, 0.05, limits)
    frame = _frame(1, left_grip=True, left_position=(0.005, 1, -0.5))
    angle = math.radians(2) / 2
    left = replace(frame.controllers["left"], orientation=(math.sin(angle), 0, 0, math.cos(angle)))
    frame = replace(frame, controllers={**frame.controllers, "left": left})
    result = mapper.map(frame, state, 0.05, limits)
    assert result[_left("shoulder_pan")] == pytest.approx(0.6)
    assert result[_left("wrist_flex")] == state[_left("wrist_flex")]


def test_deadband_accumulates_slow_height_and_wrist_while_pan_is_moving():
    state = _state()
    mapper = QuestMapper()
    mapper.map(_frame(0, left_grip=True), state, 0.05)
    for seq in range(1, 7):
        frame = _frame(seq, left_grip=True, left_position=(seq * 0.005, 1 + seq * 0.0004, -0.5))
        angle = math.radians(seq * 0.2) / 2
        left = replace(frame.controllers["left"], orientation=(math.sin(angle), 0, 0, math.cos(angle)))
        frame = replace(frame, controllers={**frame.controllers, "left": left})
        result = mapper.map(frame, state, 0.05)
    reference = mapper.motion_reference["left"]
    assert reference.position[1] == pytest.approx(1.0024)
    assert reference.orientation == pytest.approx(frame.controllers["left"].orientation)
    assert result[_left("shoulder_pan")] == pytest.approx(3.6)
    old = mapper.kinematics.forward(0, 0)
    new = mapper.kinematics.forward(result[_left("shoulder_lift")], result[_left("elbow_flex")])
    assert new[1] > old[1] + 0.001
