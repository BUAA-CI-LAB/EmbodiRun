import time

import pytest
from tests.xlerobot_owner.test_hardware import _held_snapshot, _robot

from embodirun_xlerobot_owner.hardware import WHEEL_NAMES


def station(tmp_path, monkeypatch):
    robot, buses = _robot(tmp_path, enable_base=True, wheel_directions={"left": -1, "right": 1})
    monkeypatch.setattr(robot, "_boot_id", lambda: "test-boot")
    snapshot = _held_snapshot(robot, buses)
    for name in WHEEL_NAMES:
        snapshot["goals"].pop(name)
    status = {
        "connected": True,
        "armed": False,
        "control_owner": None,
        "control_state": {"arms": False, "base": False},
        "stop_unconfirmed": False,
        "recording": False,
        "error": None,
        "observation_errors": [],
        "feedback": {"stop_confirmed": True, "errors": []},
    }
    raw = {
        n: {"Present_Position": 2000, "Present_Velocity": 0, "Moving": 0} for n in robot._motor_names_for_control("all")
    }
    evidence = {
        "source": "physical",
        "boot_id": "test-boot",
        "created_monotonic_s": time.monotonic(),
        "status_before": dict(status),
        "status_after": dict(status),
        "samples": [{"state_timestamp_ns": i + 1, "state_cached": False, "errors": [], "raw": raw} for i in range(3)],
    }
    snapshot.update(wheel_torque=dict.fromkeys(WHEEL_NAMES, 1), prior_idle=evidence)
    for name in WHEEL_NAMES:
        buses["right"].values[name].update(Torque_Enable=1, Goal_Velocity=0)
    return robot, buses, snapshot


def test_powered_zero_wheels_restore_without_writes_or_arming(tmp_path, monkeypatch):
    robot, buses, snapshot = station(tmp_path, monkeypatch)
    try:
        result = robot.restore_held_state(snapshot)
        assert result["restored"] and result["register_writes"] == 0, result
        assert not robot.armed and robot.control_state() == {"arms": False, "base": False}
        assert robot._owned_torque_names == set(snapshot["goals"]) | set(WHEEL_NAMES)
        assert all(not b.writes for b in buses.values())
        assert all(buses["right"].values[n]["Torque_Enable"] == 1 for n in WHEEL_NAMES)
        assert robot.arm("base")["armed"]  # offline fake motors only; no auto-arm on restore
    finally:
        robot.close()


@pytest.mark.parametrize("bad_feedback", [False, True])
def test_prior_readonly_restore_counts_only_without_later_feedback(tmp_path, monkeypatch, bad_feedback):
    robot, _, snapshot = station(tmp_path, monkeypatch)
    for key in ("status_before", "status_after"):
        status = snapshot["prior_idle"][key]
        status["metadata"] = {
            "holding_resume": {"restored": True, "register_writes": 0, "stop_confirmed": True, "errors": []}
        }
        status["feedback"] = {"stop_confirmed": False} if bad_feedback else None
    try:
        assert robot.restore_held_state(snapshot)["restored"] is not bad_feedback
    finally:
        robot.close()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_prior",
        "wrong_boot",
        "stale",
        "owner",
        "active",
        "uncertain_stop",
        "missing_stop",
        "recording",
        "incomplete_samples",
        "cached",
        "duplicate_time",
        "prior_moving",
        "prior_missing_motor",
        "goal_nonzero",
        "torque_changed",
        "live_moving",
        "live_velocity",
        "mode_changed",
        "bad_calibration",
        "missing_wheel",
        "bool_torque",
        "existing_stop_latch",
        "existing_torque_latch",
        "missing_goal_read",
    ],
)
def test_powered_handoff_refuses_invalid_evidence_without_writes(tmp_path, monkeypatch, fault):
    robot, buses, snapshot = station(tmp_path, monkeypatch)
    prior = snapshot["prior_idle"]
    status = prior["status_after"]
    wheel = buses["right"].values["base_left_wheel"]
    if fault == "missing_prior":
        snapshot.pop("prior_idle")
    elif fault == "wrong_boot":
        prior["boot_id"] = "other"
    elif fault == "stale":
        prior["created_monotonic_s"] -= 121
    elif fault == "owner":
        status["control_owner"] = "other"
    elif fault == "active":
        status["control_state"] = {"arms": False, "base": True}
    elif fault == "uncertain_stop":
        status["stop_unconfirmed"] = True
    elif fault == "missing_stop":
        status["feedback"] = {}
    elif fault == "recording":
        status["recording"] = True
    elif fault == "incomplete_samples":
        prior["samples"].pop()
    elif fault == "cached":
        prior["samples"][0]["state_cached"] = True
    elif fault == "duplicate_time":
        prior["samples"][1]["state_timestamp_ns"] = 1
    elif fault == "prior_moving":
        prior["samples"][0]["raw"]["base_left_wheel"]["Moving"] = 1
    elif fault == "prior_missing_motor":
        prior["samples"][0]["raw"].pop("base_left_wheel")
    elif fault == "goal_nonzero":
        wheel["Goal_Velocity"] = 1
    elif fault == "torque_changed":
        wheel["Torque_Enable"] = 0
    elif fault == "live_moving":
        wheel["Moving"] = 1
    elif fault == "live_velocity":
        wheel["Present_Velocity"] = -50
    elif fault == "mode_changed":
        wheel["Operating_Mode"] = 0
    elif fault == "bad_calibration":
        wheel["Homing_Offset"] = 1
    elif fault == "missing_wheel":
        snapshot["wheel_torque"].pop("base_left_wheel")
    elif fault == "bool_torque":
        snapshot["wheel_torque"]["base_left_wheel"] = True
    elif fault == "existing_stop_latch":
        robot._stop_uncertain = True
    elif fault == "existing_torque_latch":
        robot._torque_ownership_uncertain = True
    elif fault == "missing_goal_read":
        wheel.pop("Goal_Velocity")
    try:
        result = robot.restore_held_state(snapshot)
        assert not result["restored"], result
        assert not robot.armed and not robot._owned_torque_names
        assert all(not b.writes for b in buses.values())
    finally:
        robot.close()
