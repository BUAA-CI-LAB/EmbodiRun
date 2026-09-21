import time

import pytest

from embodirun_xlerobot_owner.hardware import JOINT_NAMES, WATCHDOG_TIMEOUT_S, HardwareRobot

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class FakeBus:
    default_baudrate = 1_000_000

    def __init__(self, side, port, motors, calibration, *, fail_field=None, fail_motor=None):
        self.side = side
        self.port = port
        self.motors = motors
        self.calibration = calibration
        self.is_connected = False
        self.connect_calls = []
        self.disconnect_calls = []
        self.baudrates = []
        self.writes = []
        self.fail_field = fail_field
        self.fail_motor = fail_motor
        self.values = {}
        for name in motors:
            calibration_entry = calibration.get(name, {})
            minimum = calibration_entry.get("range_min", 1000)
            maximum = calibration_entry.get("range_max", 3000)
            self.values[name] = {
                "Present_Position": 2000,
                "Present_Velocity": 0,
                "Moving": 0,
                "Torque_Enable": 0,
                "Operating_Mode": 1 if name.startswith("base_") else 0,
                "Homing_Offset": calibration_entry.get("homing_offset", 0),
                "Min_Position_Limit": minimum,
                "Max_Position_Limit": maximum,
                "Goal_Position": 2000,
                "Goal_Velocity": 0,
                "Acceleration": 0,
            }

    def connect(self, handshake=True):
        assert handshake is False
        self.connect_calls.append(handshake)
        self.is_connected = True

    def set_baudrate(self, value):
        self.baudrates.append(value)

    def ping(self, motor_id, num_retry=0, raise_on_error=False):
        assert num_retry == 0
        assert raise_on_error is False
        return 777

    def read(self, field, motor, normalize=False):
        assert normalize is False
        return self.values[motor][field]

    def write(self, field, motor, value, normalize=False):
        assert normalize is False
        self.writes.append((field, motor, value))
        if field == self.fail_field and motor == self.fail_motor:
            if field == "Torque_Enable":
                # Simulate an ambiguous write: the servo accepted it but the
                # transport raised before returning to the caller.
                self.values[motor][field] = value
            raise OSError("simulated transport failure")
        self.values[motor][field] = value

    def disconnect(self, disable_torque=True):
        self.disconnect_calls.append(disable_torque)
        self.is_connected = False


def _calibration():
    result = {}
    for side in ("left", "right"):
        for motor_id, suffix in enumerate(ARM_SUFFIXES, start=1):
            result[f"{side}_arm_{suffix}"] = {
                "id": motor_id,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 1000,
                "range_max": 3000,
            }
    result["base_left_wheel"] = {
        "id": 9,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 1000,
        "range_max": 3000,
    }
    result["base_right_wheel"] = {
        "id": 10,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 1000,
        "range_max": 3000,
    }
    return result


def _config(tmp_path, **overrides):
    buses = {}
    fail_field = overrides.pop("_fail_field", None)
    fail_motor = overrides.pop("_fail_motor", None)

    def factory(side, port, motors, calibration):
        bus = FakeBus(
            side,
            port,
            motors,
            calibration,
            fail_field=fail_field,
            fail_motor=fail_motor,
        )
        buses[side] = bus
        return bus

    config = {
        "sdk_src": "/does/not/load-with-a-fake-bus",
        "calibration": _calibration(),
        "ports": {
            "left": str(tmp_path / "left.tty"),
            "right": str(tmp_path / "right.tty"),
        },
        "cameras": {},
        "allow_motion": True,
        "enabled_arms": ["left", "right"],
        "enable_base": False,
        "arm_velocity_raw": 50,
        "arm_acceleration_raw": 254,
        "read_interval_s": 0,
        "stop_poll_interval_s": 0.001,
        "stop_timeout_s": 0.05,
        "cleanup_timeout_s": 0.1,
        "_bus_factory": factory,
    }
    config.update(overrides)
    config["_buses"] = buses
    return config


def _robot(tmp_path, **overrides):
    config = _config(tmp_path, **overrides)
    robot = HardwareRobot(config)
    robot.connect()
    return robot, config["_buses"]


def _left_action(**changes):
    action = {name: 0.0 for name in JOINT_NAMES if name.startswith("left_")}
    action["left_arm_gripper.pos"] = 50.0
    action.update(changes)
    return action


def _arm(robot):
    result = robot.arm()
    assert result["status"] == "armed", result
    assert robot.armed is True
    return result


@pytest.mark.parametrize("first,second", [("arms", "base"), ("base", "arms")])
def test_independent_arm_stop_and_rearm_do_not_write_other_scope(tmp_path, first, second):
    robot, buses = _robot(tmp_path, enable_base=True, wheel_directions={"base_left_wheel": 1, "base_right_wheel": -1})
    try:
        assert robot.arm(first)["armed"]
        # The already active group may be moving while the other is enabled.
        first_motor = "left_arm_shoulder_pan" if first == "arms" else "base_left_wheel"
        bus = buses["left"] if first == "arms" else buses["right"]
        bus.values[first_motor]["Moving"] = 1
        result = robot.arm(second)
        assert result["armed"], result
        assert all((w["motor"].startswith("base_")) == (second == "base") for w in result["writes"])
        assert robot.control_state() == {"arms": True, "base": True}
        result = robot.stop(second)
        assert result["stop_confirmed"], result
        assert all((w["motor"].startswith("base_")) == (second == "base") for w in result["writes"])
        assert robot.control_state()[first] and not robot.control_state()[second]
        assert robot.arm(second)["armed"]
        bus.values[first_motor]["Moving"] = 0
        assert robot.stop()["stop_confirmed"]
        assert not any(robot.control_state().values())
    finally:
        robot.close()


@pytest.mark.parametrize("live,expired", [("arms", "base"), ("base", "arms")])
def test_one_scope_heartbeat_cannot_keep_other_scope_alive(tmp_path, live, expired):
    robot, _ = _robot(tmp_path, enable_base=True, wheel_directions={"base_left_wheel": 1, "base_right_wheel": -1})
    try:
        assert robot.arm()["armed"]
        with robot._io_lock:
            robot._scope_command_mono[expired] = time.monotonic() - WATCHDOG_TIMEOUT_S - 1
            action = _left_action() if live == "arms" else {"x.vel": 0.0, "theta.vel": 0.0}
            assert robot.command(action)["accepted"]
        deadline = time.monotonic() + 0.2
        while robot.control_state()[expired] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert robot.control_state() == {live: True, expired: False}
        rejected = _left_action() if expired == "arms" else {"x.vel": 0, "theta.vel": 0}
        assert robot.command(rejected)["accepted"] is False
        assert robot.command(action)["accepted"]
    finally:
        robot.close()


def _held_snapshot(robot, buses):
    goals = {}
    for name in robot._motor_names_for_control():
        bus = buses["left" if name.startswith("left_") else "right"]
        bus.values[name]["Torque_Enable"] = 1
        goals[name] = bus.values[name]["Goal_Position"]
    return {
        "source": "physical",
        "stop_confirmed": True,
        "created_monotonic_s": time.monotonic(),
        "ports": dict(robot._ports),
        "enable_base": False,
        "goals": goals,
    }


def test_explicit_held_restore_is_read_only_and_does_not_arm(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        snapshot = _held_snapshot(robot, buses)
        # Default startup must still reject powered servos without a handoff.
        assert robot.arm()["status"] == "refused"
        result = robot.restore_held_state(snapshot)
        assert result["restored"] is True, result
        assert result["register_writes"] == 0
        assert not robot.armed
        assert robot._held_raw == snapshot["goals"]
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()
    assert all(not bus.writes for bus in buses.values())
    assert all(bus.disconnect_calls == [False] for bus in buses.values())


def test_user_can_arm_after_verified_held_restore(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        result = robot.restore_held_state(_held_snapshot(robot, buses))
        assert result["restored"] is True
        _arm(robot)
    finally:
        robot.close()


@pytest.mark.parametrize("bad_wheel", [None, "Torque_Enable", "Goal_Velocity", "Moving"])
def test_held_arm_handoff_can_add_only_cold_stationary_wheels(tmp_path, bad_wheel):
    robot, buses = _robot(tmp_path, enable_base=True, wheel_directions={"base_left_wheel": 1, "base_right_wheel": 1})
    try:
        names = robot._motor_names_for_control("arms")
        goals = {}
        for name in names:
            bus = buses["left" if name.startswith("left_") else "right"]
            bus.values[name]["Torque_Enable"] = 1
            goals[name] = bus.values[name]["Goal_Position"]
        if bad_wheel:
            buses["right"].values["base_left_wheel"][bad_wheel] = 1
        snapshot = {
            "source": "physical",
            "stop_confirmed": True,
            "created_monotonic_s": time.monotonic(),
            "ports": robot._ports,
            "enable_base": False,
            "goals": goals,
        }
        result = robot.restore_held_state(snapshot)
        assert result["restored"] is (bad_wheel is None), result
        assert not robot.armed and all(not b.writes for b in buses.values())
        if bad_wheel is None:
            assert robot.arm("base")["armed"]
            assert robot.arm("arms")["armed"]
    finally:
        robot.close()


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "ports",
        "missing_joint",
        "bool_goal",
        "goal_changed",
        "torque_off",
        "moving",
        "velocity",
        "eeprom",
        "unconfirmed",
        "base",
    ],
)
def test_held_restore_rejects_invalid_handoff_without_writes(tmp_path, fault):
    robot, buses = _robot(tmp_path)
    try:
        snapshot = _held_snapshot(robot, buses)
        name = "left_arm_shoulder_lift"
        fields = buses["left"].values[name]
        if fault == "stale":
            snapshot["created_monotonic_s"] -= 121
        elif fault == "ports":
            snapshot["ports"]["left"] += ".other"
        elif fault == "missing_joint":
            snapshot["goals"].pop(name)
        elif fault == "bool_goal":
            snapshot["goals"][name] = True
        elif fault == "goal_changed":
            fields["Goal_Position"] += 1
        elif fault == "torque_off":
            fields["Torque_Enable"] = 0
        elif fault == "moving":
            fields["Moving"] = 1
        elif fault == "velocity":
            fields["Present_Velocity"] = 1
        elif fault == "eeprom":
            fields["Min_Position_Limit"] += 1
        elif fault == "unconfirmed":
            snapshot["stop_confirmed"] = False
        else:
            snapshot["enable_base"] = True
        result = robot.restore_held_state(snapshot)
        assert not result["restored"], result
        assert result["errors"]
        assert not robot.armed
        assert not robot._owned_torque_names
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()


@pytest.mark.parametrize("fault", [None, "cold_powered", "cold_moving", "overlap", "missing", "duplicate"])
def test_fresh_handoff_retains_powered_arm_and_independently_checks_cold_arm(tmp_path, fault):
    robot, buses = _robot(tmp_path)
    try:
        snapshot = _held_snapshot(robot, buses)
        cold = [n for n in snapshot["goals"] if n.startswith("right_")]
        snapshot["cold_motors"] = list(cold)
        for name in cold:
            snapshot["goals"].pop(name)
            buses["right"].values[name]["Torque_Enable"] = 0
        if fault == "cold_powered":
            buses["right"].values[cold[0]]["Torque_Enable"] = 1
        elif fault == "cold_moving":
            buses["right"].values[cold[0]]["Moving"] = 1
        elif fault == "overlap":
            snapshot["goals"][cold[0]] = 2000
        elif fault == "missing":
            snapshot["cold_motors"].pop()
        elif fault == "duplicate":
            snapshot["cold_motors"].append(cold[0])
        result = robot.restore_held_state(snapshot)
        assert result["restored"] is (fault is None), result
        assert not robot.armed and all(not b.writes for b in buses.values())
        if fault is None:
            assert robot._owned_torque_names == set(snapshot["goals"])
            assert robot.arm("arms")["armed"]
    finally:
        robot.close()


def test_camera_only_connect_is_dependency_lazy_and_read_only():
    robot = HardwareRobot(
        {
            "camera_roles_confirmed": False,
            "allow_motion": False,
        }
    )
    try:
        result = robot.connect()
        observation, cameras = robot.read()
        assert result["read_only_startup"] is True
        assert observation["state"] == {}
        assert cameras == {}
        assert observation["metadata"]["camera_roles_confirmed"] is False
        assert observation["metadata"]["camera_names"] == ["front", "left_wrist", "right_wrist"]
    finally:
        robot.close()


def test_connect_and_read_do_not_write_registers(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        observation, cameras = robot.read()
        assert cameras == {}
        assert set(observation["state"]) == set(JOINT_NAMES)
        assert all(not bus.writes for bus in buses.values())
        assert all(bus.connect_calls == [False] for bus in buses.values())
        assert observation["metadata"]["enabled_arms"] == ["left", "right"]
        assert observation["metadata"]["enable_base"] is False
        assert observation["metadata"]["allow_motion"] is True
        assert observation["metadata"]["arm_velocity_raw_configured"] is True
        assert observation["metadata"]["joint_limits"]["left_arm_gripper.pos"] == [0, 100]
    finally:
        robot.close()


def test_out_of_range_preflight_refuses_before_any_write(tmp_path):
    robot, buses = _robot(tmp_path)
    buses["left"].values["left_arm_shoulder_lift"]["Present_Position"] = 3001
    try:
        result = robot.arm()
        assert result["status"] == "refused"
        assert result["armed"] is False
        assert any("outside saved range" in error for error in result["errors"])
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()


def test_small_static_feedback_overshoot_is_tolerated_but_hold_target_is_clamped(tmp_path):
    robot, buses = _robot(tmp_path, position_feedback_tolerance_raw=16)
    name = "left_arm_shoulder_lift"
    buses["left"].values[name]["Present_Position"] = 3009
    try:
        observation, _ = robot.read()
        assert not any(name in error for error in observation["errors"])
        result = robot.arm()
        assert result["status"] == "armed", result
        assert result["held_positions_raw"][name] == 3000
        assert ("Goal_Position", name, 3000) in buses["left"].writes
        assert robot.metadata["position_feedback_tolerance_raw"] == 16
    finally:
        robot.close()


def test_arm_holds_actual_start_before_torque_enable(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        result = _arm(robot)
        for side in ("left", "right"):
            bus = buses[side]
            holds = {motor: value for field, motor, value in bus.writes if field == "Goal_Position"}
            assert all(holds[f"{side}_arm_{suffix}"] == 2000 for suffix in ARM_SUFFIXES)
            assert all(field != "Operating_Mode" for field, _, _ in bus.writes)
            assert all(
                field not in {"Min_Position_Limit", "Max_Position_Limit", "Homing_Offset"} for field, _, _ in bus.writes
            )
            assert all(("Acceleration", f"{side}_arm_{suffix}", 254) in bus.writes for suffix in ARM_SUFFIXES)
            assert all(("Goal_Velocity", f"{side}_arm_{suffix}", 50) in bus.writes for suffix in ARM_SUFFIXES)
            torque_writes = [entry for entry in bus.writes if entry[0] == "Torque_Enable"]
            assert {entry[1] for entry in torque_writes} == {f"{side}_arm_{suffix}" for suffix in ARM_SUFFIXES}
        assert result["held_positions_raw"]["left_arm_shoulder_pan"] == 2000
    finally:
        robot.close()


def test_arm_retries_a_motion_profile_write_that_did_not_take_effect(tmp_path):
    robot, buses = _robot(tmp_path)
    bus = buses["left"]
    original_write = bus.write
    dropped = False

    def drop_first_acceleration(field, motor, value, normalize=False):
        nonlocal dropped
        if field == "Acceleration" and motor == "left_arm_shoulder_pan" and not dropped:
            assert normalize is False
            bus.writes.append((field, motor, value))
            dropped = True
            return
        original_write(field, motor, value, normalize=normalize)

    bus.write = drop_first_acceleration
    try:
        result = _arm(robot)
        attempts = [write for write in bus.writes if write[:2] == ("Acceleration", "left_arm_shoulder_pan")]
        assert len(attempts) == 2
        assert result["goal_acceleration_raw"] == 254
    finally:
        robot.close()


def test_arm_refuses_before_torque_if_motion_profile_never_reads_back(tmp_path):
    robot, buses = _robot(tmp_path)
    bus = buses["left"]
    original_write = bus.write

    def ignore_acceleration(field, motor, value, normalize=False):
        if field == "Acceleration" and motor == "left_arm_shoulder_pan":
            assert normalize is False
            bus.writes.append((field, motor, value))
            return
        original_write(field, motor, value, normalize=normalize)

    bus.write = ignore_acceleration
    try:
        result = robot.arm()
        assert result["status"] == "error"
        assert result["armed"] is False
        assert any("after 3 attempts (actual=0)" in error for error in result["errors"])
        assert all(field != "Torque_Enable" for candidate in buses.values() for field, _, _ in candidate.writes)
    finally:
        robot.close()


def test_missing_arm_velocity_raw_refuses_motion_without_writes(tmp_path):
    robot, buses = _robot(tmp_path, arm_velocity_raw=None)
    try:
        result = robot.arm()
        assert result["status"] == "refused"
        assert any("arm_velocity_raw" in error for error in result["errors"])
        assert robot.metadata["arm_velocity_raw_configured"] is False
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()


@pytest.mark.parametrize("bad_value", [-1, 3401, 1.5, float("nan")])
def test_invalid_arm_motion_profile_refuses_without_writes(tmp_path, bad_value):
    robot, buses = _robot(tmp_path, arm_velocity_raw=bad_value)
    try:
        result = robot.arm()
        assert result["status"] == "refused"
        assert result["errors"]
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()


def test_stop_clamps_feedback_overshoot_to_saved_target_range(tmp_path):
    robot, buses = _robot(tmp_path, position_feedback_tolerance_raw=16)
    name = "left_arm_shoulder_lift"
    try:
        _arm(robot)
        buses["left"].values[name]["Present_Position"] = 3009
        result = robot.stop()
        assert result["stop_confirmed"] is True, result
        assert ("Goal_Position", name, 3000) in buses["left"].writes
    finally:
        robot.close()


def test_command_writes_nonzero_goal_and_rejects_invalid_or_fast_targets(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        _arm(robot)
        time.sleep(0.08)
        result = robot.command(_left_action(**{"left_arm_shoulder_pan.pos": 0.5}))
        assert result["accepted"] is True, result
        left_goal_writes = [entry for entry in buses["left"].writes if entry[0] == "Goal_Position"]
        assert any(motor == "left_arm_shoulder_pan" and value != 2000 for _, motor, value in left_goal_writes)

        writes_before = sum(len(bus.writes) for bus in buses.values())
        invalid = robot.command(_left_action(**{"left_arm_shoulder_pan.pos": float("nan")}))
        assert invalid["accepted"] is False
        assert sum(len(bus.writes) for bus in buses.values()) == writes_before

        missing = _left_action()
        missing.pop("left_arm_wrist_roll.pos")
        missing_result = robot.command(missing)
        assert missing_result["accepted"] is False
        assert sum(len(bus.writes) for bus in buses.values()) == writes_before

        fast = robot.command(_left_action(**{"left_arm_shoulder_pan.pos": 20.0}))
        assert fast["accepted"] is True
        assert abs(fast["applied_action"]["left_arm_shoulder_pan.pos"]) < 20.0
        assert sum(len(bus.writes) for bus in buses.values()) > writes_before
    finally:
        robot.close()


def test_base_velocity_is_clamped_and_only_wheels_are_written(tmp_path):
    robot, buses = _robot(
        tmp_path,
        enabled_arms=[],
        enable_base=True,
        wheel_directions={"left": 1, "right": -1},
    )
    try:
        _arm(robot)
        result = robot.command({"x.vel": 10.0, "theta.vel": -50.0})
        assert result["accepted"] is True, result
        assert result["applied_action"] == {"x.vel": 0.05, "theta.vel": -10.0}
        assert all(
            entry["field"] == "Goal_Velocity" and entry["motor"].startswith("base_") for entry in result["writes"]
        )
        assert not buses["left"].writes
    finally:
        robot.close()


def test_watchdog_stops_without_replaying_stale_command(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        _arm(robot)
        deadline = time.monotonic() + WATCHDOG_TIMEOUT_S + 0.4
        while (robot.armed or robot._last_stop is None) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert robot.armed is False
        assert robot._last_stop["reason"] == "watchdog"
        assert robot._last_stop["stop_confirmed"] is True
        retry = robot.stop()
        assert retry["status"] == "confirmed"
        assert retry["reason"] == "operator_retry"
        assert retry["stop_confirmed"] is True
        assert any(entry[0] == "Goal_Position" for entry in buses["left"].writes)
    finally:
        robot.close()


def test_stop_holds_last_gripper_goal_without_disabling_torque(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        _arm(robot)
        time.sleep(0.08)
        command = robot.command(_left_action(**{"left_arm_gripper.pos": 52.0}))
        assert command["accepted"] is True, command
        expected_raw = command["raw_action"]["left_arm_gripper"]
        result = robot.stop()
        assert result["stop_confirmed"] is True, result
        gripper_holds = [
            value
            for field, motor, value in buses["left"].writes
            if field == "Goal_Position" and motor == "left_arm_gripper"
        ]
        assert gripper_holds[-1] == expected_raw
        assert all(not (field == "Torque_Enable" and value == 0) for field, _, value in buses["left"].writes)
    finally:
        robot.close()


def test_enabled_arm_isolation_does_not_touch_other_arm(tmp_path):
    robot, buses = _robot(tmp_path, enabled_arms=["left"])
    try:
        _arm(robot)
        time.sleep(0.08)
        result = robot.command(_left_action(**{"left_arm_shoulder_pan.pos": 0.5}))
        assert result["accepted"] is True, result
        assert not buses["right"].writes
        assert all(not motor.startswith("right_") for _, motor, _ in buses["left"].writes)
        assert robot.metadata["enabled_arms"] == ["left"]
    finally:
        robot.close()


def test_torque_write_failure_latches_uncertainty_and_reports_partial_writes(tmp_path):
    robot, buses = _robot(
        tmp_path,
        _fail_field="Torque_Enable",
        _fail_motor="left_arm_elbow_flex",
    )
    try:
        result = robot.arm()
        assert result["status"] == "error"
        assert result["armed"] is False
        assert robot._stop_uncertain is True
        assert any(entry["status"] == "attempted" for entry in result["writes"])
        assert all(not (field == "Torque_Enable" and value == 0) for field, _, value in buses["left"].writes)
    finally:
        robot.close()


def test_close_disconnects_even_if_stop_raises(tmp_path):
    robot, buses = _robot(tmp_path)
    _arm(robot)
    robot._stop_locked = lambda reason: (_ for _ in ()).throw(RuntimeError("stop boom"))
    result = robot.close()
    assert result["closed"] is True
    assert any("close stop failed" in error for error in result["errors"])
    assert all(bus.disconnect_calls == [False] for bus in buses.values())
    assert all(
        not (field == "Torque_Enable" and value == 0) for bus in buses.values() for field, _, value in bus.writes
    )


def test_unowned_stop_is_fresh_read_only_stationary_poll(tmp_path):
    robot, buses = _robot(tmp_path)
    try:
        result = robot.stop()
        assert result["reason"] == "no_control"
        assert result["stop_confirmed"] is True
        assert result["command_accepted"] is False
        assert result["writes"] == []
        assert all(not bus.writes for bus in buses.values())
    finally:
        robot.close()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_base_velocity_is_rejected_before_any_write(tmp_path, bad_value):
    robot, buses = _robot(
        tmp_path,
        enabled_arms=[],
        enable_base=True,
        wheel_directions={"left": 1, "right": 1},
    )
    try:
        _arm(robot)
        writes_before = sum(len(bus.writes) for bus in buses.values())
        result = robot.command({"x.vel": bad_value, "theta.vel": 0.0})
        assert result["accepted"] is False
        assert sum(len(bus.writes) for bus in buses.values()) == writes_before
    finally:
        robot.close()
