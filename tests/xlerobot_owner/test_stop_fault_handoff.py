import copy
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from tests.xlerobot_owner.test_hardware import FakeBus, _config
from tests.xlerobot_owner.test_motor_diagnostics import Packet
from tests.xlerobot_owner.test_server import HEADERS, TOKEN, station

from embodirun_xlerobot_owner.hardware import HEAD_TILT_NAME, WHEEL_NAMES, HardwareRobot
from embodirun_xlerobot_owner.server import Platform
from embodirun_xlerobot_owner.stop_fault_handoff import validate_snapshot


def prepared(tmp_path, monkeypatch, *, legacy=False, torque_fault=False):
    monkeypatch.setattr(HardwareRobot, "_boot_id", staticmethod(lambda: "test-boot"))
    config = _config(
        tmp_path,
        enable_base=True,
        enable_head_tilt=True,
        wheel_directions={"left": -1, "right": 1},
        stop_timeout_s=0.006,
    )
    config["calibration"][HEAD_TILT_NAME] = {
        "id": 8,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 1000,
        "range_max": 3000,
    }
    old = HardwareRobot(config)
    old.connect()
    started = time.time_ns()
    arm = old.arm()
    assert arm["armed"], arm
    # The gripper is loaded: its goal must survive, not be replaced by its position.
    for name in ("left_arm_gripper", "right_arm_gripper"):
        old._bus_for_name(name).values[name]["Present_Position"] = 2017
    old._buses["right"].values[WHEEL_NAMES[0]].update(Present_Velocity=50, Moving=1)
    old._torque_ownership_uncertain = torque_fault
    run = {"started_at_ns": started, "completed_at_ns": time.time_ns(), "arm_feedback": arm}
    stop = old._stop_locked("test-only fault seed")
    assert not stop["stop_confirmed"] and len(stop["writes"]) == 15
    status = {
        "connected": True,
        "armed": False,
        "control_owner": None,
        "control_state": {"arms": False, "base": False},
        "stop_unconfirmed": True,
        "recording": False,
        "error": None,
        "observation_errors": [],
        "feedback": stop,
        "hardware_safety": old.safety_state(),
    }
    samples = []
    for _ in range(3):
        observation, _ = old.read()
        samples.append({k: observation[k] for k in ("state_timestamp_ns", "state_cached", "errors", "raw")})
    snapshot = {
        "source": "physical",
        "ports": config["ports"],
        "boot_id": "test-boot",
        "created_monotonic_s": time.monotonic(),
        "status_before": copy.deepcopy(status),
        "status_after": copy.deepcopy(status),
        "samples": samples,
        "failed_stop": stop,
    }
    values = {side: copy.deepcopy(bus.values) for side, bus in old._buses.items()}
    old.close()
    if legacy:
        for key in ("status_before", "status_after"):
            snapshot[key].pop("hardware_safety")
            snapshot[key]["metadata"] = {
                "holding_resume": {
                    "restored": True,
                    "register_writes": 0,
                    "stop_confirmed": True,
                    "errors": [],
                    "motors": [n for n in old._motor_names_for_control() if n not in WHEEL_NAMES],
                    "cold_motors": [],
                    "wheel_torque": dict.fromkeys(WHEEL_NAMES, 1),
                }
            }
        snapshot["legacy_stop_only_review"] = {"last_arm_run": run, "no_enable_attempts_since": True}

    buses = {}

    def factory(side, port, motors, calibration):
        bus = FakeBus(side, port, motors, calibration)
        bus.values = copy.deepcopy(values[side])
        if side == "right":
            bus.port_handler, bus.packet_handler = object(), Packet(bus)
        buses[side] = bus
        return bus

    config.update(_bus_factory=factory, _resume_stop_fault=snapshot)
    return HardwareRobot(config), buses, snapshot


@pytest.mark.parametrize("legacy", [False, True])
def test_fault_visible_before_startup_goals_preserved_and_zero_observations_do_not_clear(tmp_path, monkeypatch, legacy):
    robot, buses, snapshot = prepared(tmp_path, monkeypatch, legacy=legacy)
    platform = Platform(robot, TOKEN, output=tmp_path)
    assert platform.stop_unconfirmed and robot.stop_unconfirmed
    try:
        robot.connect()
        assert not robot.arm("base")["armed"] and not robot.arm("arms")["armed"]
        assert robot.inherited_stop_feedback == snapshot["failed_stop"]
        assert len(robot._owned_torque_names) == 15 and len(robot._held_raw) == 13
        assert robot._last_gripper_goal_raw == {"left_arm_gripper": 2000, "right_arm_gripper": 2000}
        assert robot.metadata["stop_fault_resume"]["initial_readback"]["left_arm_gripper"]["Present_Position"] == 2017
        for bus in buses.values():
            for fields in bus.values.values():
                fields.update(Present_Velocity=0, Moving=0)
        for _ in range(3):
            robot.read()
        assert robot.stop_unconfirmed and all(not b.writes for b in buses.values())
        assert robot.control_state() == {"arms": False, "base": False}
        # Even strict readonly zeros and automatic all-stop do not wash the fault.
        assert robot._stop_locked("read-only", write_commands=False)["stop_confirmed"]
        assert robot.stop_unconfirmed
        assert robot.stop()["stop_confirmed"] and robot.stop_unconfirmed
        assert robot.stop("base")["stop_confirmed"] and robot.stop_unconfirmed
        assert robot.stop_all()["stop_confirmed"] and not robot.stop_unconfirmed
        assert robot.arm("base")["armed"]  # fake devices only, after explicit successful all-stop
    finally:
        robot.close()


@pytest.mark.parametrize("failure", ["nonzero", "write", "bad_packet"])
def test_failed_explicit_all_stop_keeps_both_faults(tmp_path, monkeypatch, failure):
    robot, buses, _ = prepared(tmp_path, monkeypatch)
    try:
        robot.connect()
        if failure == "write":
            buses["right"].values[WHEEL_NAMES[0]].update(Present_Velocity=0, Moving=0)
            buses["right"].fail_field, buses["right"].fail_motor = "Goal_Velocity", WHEEL_NAMES[0]
        if failure == "bad_packet":
            buses["right"].packet_handler.overrides[56] = ([], -3002, 0)
        result = robot.stop_all()
        assert not result["stop_confirmed"] and robot.stop_unconfirmed and robot._inherited_stop_fault
        assert not robot.arm("base")["armed"]
    finally:
        robot.close()


def test_a_separate_torque_fault_is_preserved_even_after_successful_all_stop(tmp_path, monkeypatch):
    robot, buses, _ = prepared(tmp_path, monkeypatch, torque_fault=True)
    try:
        robot.connect()
        buses["right"].values[WHEEL_NAMES[0]].update(Present_Velocity=0, Moving=0)
        assert robot.stop_all()["stop_confirmed"] and not robot.stop_unconfirmed
        assert robot._torque_ownership_uncertain and not robot.arm("base")["armed"]
    finally:
        robot.close()


@pytest.mark.parametrize(
    "bad",
    [
        "stale",
        "boot",
        "ports",
        "owner",
        "active",
        "missing_sample",
        "changed_stop",
        "missing_write",
        "torque_unknown",
        "ownership",
        "gripper",
        "cached",
        "hardware_goal",
        "hardware_torque",
        "hardware_mode",
        "hardware_calibration",
        "legacy_no_assertion",
        "legacy_no_arm",
    ],
)
def test_bad_fault_handoff_fails_startup_without_writes(tmp_path, monkeypatch, bad):
    robot, buses, snapshot = prepared(tmp_path, monkeypatch, legacy=bad.startswith("legacy"))
    if bad == "stale":
        snapshot["created_monotonic_s"] -= 121
    elif bad in {"boot", "ports"}:
        snapshot["boot_id" if bad == "boot" else "ports"] = "wrong"
    elif bad == "owner":
        snapshot["status_after"]["control_owner"] = "other"
    elif bad == "active":
        snapshot["status_after"]["control_state"]["arms"] = True
    elif bad == "missing_sample":
        snapshot["samples"].pop()
    elif bad == "cached":
        snapshot["samples"][0]["state_cached"] = True
    elif bad == "changed_stop":
        snapshot["status_before"]["feedback"]["reason"] = "different"
    elif bad == "missing_write":
        snapshot["failed_stop"]["writes"].pop()
    elif bad in {"torque_unknown", "ownership", "gripper"}:
        for key in ("status_before", "status_after"):
            safety = snapshot[key]["hardware_safety"]
            if bad == "torque_unknown":
                safety.pop("torque_ownership_uncertain")
            elif bad == "ownership":
                safety["owned_motors"].pop()
            else:
                safety["gripper_goals"]["left_arm_gripper"] = 2017
    elif bad == "legacy_no_assertion":
        snapshot["legacy_stop_only_review"].pop("no_enable_attempts_since")
    elif bad == "legacy_no_arm":
        snapshot["legacy_stop_only_review"]["last_arm_run"]["arm_feedback"]["armed"] = False
    if bad.startswith("hardware_"):
        factory = robot._bus_factory

        def changed(side, port, motors, calibration):
            bus = factory(side, port, motors, calibration)
            field = {
                "hardware_goal": "Goal_Position",
                "hardware_torque": "Torque_Enable",
                "hardware_mode": "Operating_Mode",
                "hardware_calibration": "Homing_Offset",
            }[bad]
            next(iter(bus.values.values()))[field] += 1
            return bus

        robot._bus_factory = changed
    try:
        with pytest.raises((ValueError, RuntimeError)):
            robot.connect()
        assert robot.stop_unconfirmed and not robot.connected and not robot.armed
        assert all(not b.writes and b.disconnect_calls == [False] for b in buses.values())
    finally:
        robot.close()


@pytest.mark.asyncio
async def test_api_restarts_blocked_and_only_explicit_successful_all_stop_clears_both(tmp_path, monkeypatch):
    robot, buses, snapshot = prepared(tmp_path, monkeypatch)
    headers = {**HEADERS, "X-Teleop-Owner": "maintenance", "X-Teleop-Scope": "base"}
    async with station(tmp_path, robot=robot, robot_api=True) as (platform, client):
        assert platform.stop_unconfirmed and robot.stop_unconfirmed
        assert platform.last_feedback == snapshot["failed_stop"]
        assert not platform.robot_owners
        response = await client.post("/robot/arm", headers=headers)
        assert response.status != 200 and all(not b.writes for b in buses.values())
        assert not (await (await client.post("/robot/stop_all", headers=headers)).json())["stop_confirmed"]
        assert platform.stop_unconfirmed and robot.stop_unconfirmed
        buses["right"].values[WHEEL_NAMES[0]].update(Present_Velocity=0, Moving=0)
        for _ in range(3):
            await client.get("/robot/observe", headers=headers)
        status = await (await client.get("/robot/status", headers=headers)).json()
        assert status["stop_unconfirmed"] and status["hardware_safety"]["stop_unconfirmed"]
        assert (await (await client.post("/robot/stop_all", headers=headers)).json())["stop_confirmed"]
        assert not platform.stop_unconfirmed and not robot.stop_unconfirmed
        assert not robot.armed and not platform.robot_owners


def test_json_round_trip_preserves_raw_failed_feedback(tmp_path, monkeypatch):
    robot, _, snapshot = prepared(tmp_path, monkeypatch)
    assert validate_snapshot(robot, json.loads(json.dumps(snapshot))) == snapshot["status_before"]["hardware_safety"]


def test_capture_only_reads_existing_api_no_device_access(tmp_path, monkeypatch):
    from integrations.xlerobot_owner.tools.archive import start_orin_shared_teleop as launcher

    robot, buses, snapshot = prepared(tmp_path, monkeypatch)
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.index = 0

        def _request(self, endpoint):
            assert endpoint == "status"
            calls.append(endpoint)
            return copy.deepcopy(snapshot["status_before"])

        def read(self):
            calls.append("observe")
            item = snapshot["samples"][self.index]
            self.index += 1
            return item, {}

    monkeypatch.setattr(launcher, "RemoteRobot", Client)
    monkeypatch.setattr(launcher, "HardwareRobot", lambda config: robot)
    token = tmp_path / "token"
    token.write_text(TOKEN)
    name = launcher.capture_stop_fault(robot.config, token, tmp_path)
    saved = json.loads(Path(name).read_text())
    assert saved["failed_stop"] == snapshot["failed_stop"]
    assert calls == ["status", "observe", "observe", "observe", "status"]
    assert not buses and robot.stop_unconfirmed and not robot.connected


def test_cli_consumes_fault_file_once_without_fallback(tmp_path, monkeypatch):
    from aiohttp import web

    from embodirun_xlerobot_owner import __main__ as cli
    from embodirun_xlerobot_owner import hardware
    from embodirun_xlerobot_owner.server import PLATFORM

    robot, _, snapshot = prepared(tmp_path, monkeypatch)
    source = tmp_path / "fault.json"
    source.write_text(json.dumps(snapshot))
    config = tmp_path / "config.json"
    config.write_text("{}")
    token = tmp_path / "token"
    token.write_text(TOKEN)
    called = []

    def constructor(config):
        assert config["_resume_stop_fault"] == snapshot and "_resume_held_state" not in config
        return robot

    def run_app(app, **kwargs):
        called.append(app)
        assert app[PLATFORM].stop_unconfirmed and app[PLATFORM].robot.stop_unconfirmed

    monkeypatch.setattr(hardware, "HardwareRobot", constructor)
    monkeypatch.setattr(web, "run_app", run_app)
    args = [
        "serve",
        "--mode",
        "hardware",
        "--token-file",
        str(token),
        "--hardware-config",
        str(config),
        "--resume-stop-fault",
        str(source),
        "--output",
        str(tmp_path),
    ]
    assert cli.main(args) == 0
    assert not source.exists() and source.with_suffix(".json.used").is_file()
    with pytest.raises(SystemExit):
        cli.main(args)
    assert len(called) == 1
    with pytest.raises(SystemExit):
        cli.main(args + ["--resume-held-state", str(source)])
    assert len(called) == 1


def test_fault_launcher_bypasses_old_cli_and_refuses_reuse(tmp_path, monkeypatch):
    from aiohttp import web
    from integrations.xlerobot_owner.tools.archive import start_orin_shared_teleop as launcher

    from embodirun_xlerobot_owner.server import PLATFORM

    robot, buses, snapshot = prepared(tmp_path, monkeypatch)
    source, config, token = (tmp_path / n for n in ("fault.json", "config.json", "token"))
    source.write_text(json.dumps(snapshot))
    config.write_text("{}")
    token.write_text(TOKEN)
    calls = []

    def constructor(config):
        assert config["_resume_stop_fault"] == snapshot
        return robot

    def run_app(app, **kwargs):
        assert app[PLATFORM].stop_unconfirmed and robot.stop_unconfirmed
        assert not robot.connected and not buses
        calls.append("serve-blocked")

    monkeypatch.setattr(launcher, "require_idle_devices", lambda _: calls.append("check-devices"))
    monkeypatch.setattr(launcher, "HardwareRobot", constructor)
    monkeypatch.setattr(web, "run_app", run_app)
    monkeypatch.setattr(launcher.os, "execv", lambda *a: pytest.fail("must not call old serve CLI"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "start",
            "--config",
            str(config),
            "--token-file",
            str(token),
            "--output",
            str(tmp_path),
            "--resume-stop-fault",
            str(source),
        ],
    )
    launcher.main()
    assert calls == ["check-devices", "serve-blocked"]
    with pytest.raises(RuntimeError, match="already claimed"):
        launcher.main()
    assert source.with_suffix(".json.used").is_file()
