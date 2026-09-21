import time

from tests.xlerobot_owner.test_hardware import _config

from embodirun_xlerobot_owner.hardware import HEAD_TILT_NAME, HardwareRobot


def station(tmp_path):
    config = _config(
        tmp_path,
        enable_base=True,
        enable_head_tilt=True,
        wheel_directions={"base_left_wheel": -1, "base_right_wheel": 1},
    )
    config["calibration"][HEAD_TILT_NAME] = {
        "id": 8,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 1000,
        "range_max": 3000,
    }
    robot = HardwareRobot(config)
    robot.connect()
    return robot, config["_buses"]


def test_head_read_only_connection_and_no_pan_motor(tmp_path):
    robot, buses = station(tmp_path)
    try:
        assert HEAD_TILT_NAME in buses["left"].motors
        assert "head_motor_1" not in buses["left"].motors
        assert not buses["left"].writes and not buses["right"].writes
        observation, _ = robot.read()
        assert observation["state"][HEAD_TILT_NAME + ".pos"] == 0
        assert robot.metadata["head_tilt"]["scope"] == "base"
    finally:
        robot.close()


def test_enable_base_holds_current_camera_before_torque_without_arm_writes(tmp_path):
    robot, buses = station(tmp_path)
    try:
        result = robot.arm("base")
        assert result["armed"], result
        writes = buses["left"].writes
        assert all(motor == HEAD_TILT_NAME for _, motor, _ in writes)
        assert writes.index(("Goal_Position", HEAD_TILT_NAME, 2000)) < writes.index(
            ("Torque_Enable", HEAD_TILT_NAME, 1)
        )
        assert result["held_positions"] == {HEAD_TILT_NAME + ".pos": 0}
        assert robot.control_state() == {"arms": False, "base": True}
    finally:
        robot.close()


def test_camera_rate_limit_independent_of_arm_commands_and_stop_holds(tmp_path):
    robot, buses = station(tmp_path)
    try:
        assert robot.arm("base")["armed"]
        robot._last_head_target_mono = time.monotonic() - 0.05
        robot._last_target_mono = time.monotonic() - 100
        result = robot.command({"x.vel": 0, "theta.vel": 0, HEAD_TILT_NAME + ".pos": 50})
        assert result["accepted"], result
        assert 0 < result["applied_action"][HEAD_TILT_NAME + ".pos"] < 1.1
        stop = robot.stop("base")
        assert stop["stop_confirmed"]
        assert all(not w["motor"].startswith(("left_arm", "right_arm")) for w in stop["writes"])
        assert buses["left"].values[HEAD_TILT_NAME]["Torque_Enable"] == 1
    finally:
        robot.close()


def test_missing_camera_calibration_and_moving_camera_refuse_enable_without_writes(tmp_path):
    robot, buses = station(tmp_path)
    try:
        buses["left"].values[HEAD_TILT_NAME]["Moving"] = 1
        assert not robot.arm("base")["armed"]
        assert not buses["left"].writes and not buses["right"].writes
        buses["left"].values[HEAD_TILT_NAME]["Moving"] = 0
        del robot._calibration[HEAD_TILT_NAME]
        assert not robot.arm("base")["armed"]
        assert not buses["left"].writes and not buses["right"].writes
    finally:
        robot.close()


def test_arms_scope_does_not_touch_camera(tmp_path):
    robot, buses = station(tmp_path)
    try:
        assert robot.arm("arms")["armed"]
        assert all(motor != HEAD_TILT_NAME for _, motor, _ in buses["left"].writes)
        result = robot.command({HEAD_TILT_NAME + ".pos": 0})
        assert not result["accepted"]
        stop = robot.stop("arms")
        assert stop["stop_confirmed"]
        assert all(w["motor"] != HEAD_TILT_NAME for w in stop["writes"])
    finally:
        robot.close()


def test_adopt_stationary_head_goal_without_writes_or_zeroing(tmp_path):
    robot, buses = station(tmp_path)
    try:
        head = buses["left"].values[HEAD_TILT_NAME]
        head.update(Torque_Enable=1, Goal_Position=2090, Present_Position=2091, Goal_Velocity=120)
        positions = [n for n in robot._motor_names_for_control() if not n.startswith("base_")]
        snapshot = {
            "source": "physical",
            "stop_confirmed": True,
            "created_monotonic_s": time.monotonic(),
            "ports": robot._ports,
            "enable_base": False,
            "goals": {HEAD_TILT_NAME: 2090},
            "cold_motors": [n for n in positions if n != HEAD_TILT_NAME],
        }
        result = robot.restore_held_state(snapshot)
        assert result["restored"], result
        assert not buses["left"].writes and not buses["right"].writes
        assert not robot.armed
        assert head["Goal_Position"] == 2090 and head["Torque_Enable"] == 1
        result = robot.arm("base")
        assert result["armed"], result
        assert head["Goal_Position"] == 2090
    finally:
        robot.close()
