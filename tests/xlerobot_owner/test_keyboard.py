import asyncio
import subprocess
import time
from pathlib import Path

import pytest
from tests.xlerobot_owner.test_hardware import _config
from tests.xlerobot_owner.test_server import HEADERS, station, wait_for

from embodirun_xlerobot_owner.control import InputClock, MappingConfig
from embodirun_xlerobot_owner.keyboard import KeyboardFrame
from embodirun_xlerobot_owner.robot import DemoRobot


def packet(seq=0, keys=(), **extra):
    return {
        "type": "keyboard_input",
        "seq": seq,
        "timestamp_ms": time.monotonic() * 1000,
        "keys": list(keys),
        "focused": True,
        "video_ready": True,
        **extra,
    }


@pytest.mark.parametrize(
    "keys,expected",
    [
        ((), (0, 0)),
        (("KeyW",), (0.05, 0)),
        (("KeyS",), (-0.05, 0)),
        (("KeyA",), (0, 10)),
        (("KeyD",), (0, -10)),
        (("KeyW", "KeyA"), (0.05, 10)),
        (("KeyW", "KeyS", "KeyA", "KeyD"), (0, 0)),
    ],
)
def test_keyboard_maps_only_two_base_axes(keys, expected):
    result = KeyboardFrame.parse(packet(keys=keys)).action(MappingConfig(enable_base=True))
    assert result == dict(zip(("x.vel", "theta.vel"), expected))


@pytest.mark.parametrize(
    "extra",
    [
        {"keys": ["KeyX"]},
        {"keys": ["KeyW", "KeyW"]},
        {"keys": "W"},
        {"focused": 1},
        {"video_ready": None},
        {"seq": True},
        {"seq": -1},
        {"timestamp_ms": float("nan")},
        {"timestamp_ms": -1},
    ],
)
def test_invalid_keyboard_packets(extra):
    with pytest.raises((ValueError, TypeError)):
        KeyboardFrame.parse(packet(**extra))


def test_keyboard_freshness_and_availability():
    clock = InputClock()
    clock.accept(KeyboardFrame.parse(packet(timestamp_ms=0)), 100)
    with pytest.raises(ValueError):
        clock.accept(KeyboardFrame.parse(packet(timestamp_ms=0)), 100.1)
    with pytest.raises(ValueError):
        clock.accept(KeyboardFrame.parse(packet(1, timestamp_ms=50)), 101)
    for change in ({"focused": False}, {"video_ready": False}):
        value = KeyboardFrame.parse(packet(**change))
        assert not value.neutral
        with pytest.raises(ValueError):
            value.action(MappingConfig(enable_base=True))
    with pytest.raises(ValueError):
        KeyboardFrame.parse(packet()).action(MappingConfig())


async def enable(p, ws):
    await ws.send_json({"type": "set_control_mode", "mode": "drive"})
    await wait_for(lambda: p.mapper.control_mode == "drive")
    await ws.send_json(packet())
    await wait_for(lambda: bool(p.frames))
    await ws.send_json({"type": "arm", "activation": "keyboard"})
    await wait_for(lambda: p.armed)


@pytest.mark.asyncio
async def test_keyboard_drive_release_and_no_arm_commands(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        before = dict(p.robot.state)
        await enable(p, ws)
        await ws.send_json(packet(1, ("KeyW", "KeyD")))
        await wait_for(lambda: p.last_action == {"x.vel": 0.05, "theta.vel": -10})
        assert all(p.robot.state[k] == v for k, v in before.items() if k.endswith(".pos"))
        await ws.send_json(packet(2))
        await wait_for(lambda: p.last_action == {"x.vel": 0, "theta.vel": 0})
        assert p.armed  # release stops travel without a new enable per keypress
        status = p.status(ws)
        assert status["input_status"]["kind"] == "keyboard_input"
        assert status["input_status"]["tracked"] == {}
        await ws.close()
        await wait_for(lambda: not p.armed and not p.robot.armed, timeout=3)
        assert p.robot.state["x.vel"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["focus", "video", "timeout", "reorder", "switch_input"])
async def test_keyboard_faults_stop_and_never_auto_rearm(tmp_path, fault):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await enable(p, ws)
        await ws.send_json(packet(1, ("KeyW",)))
        await wait_for(lambda: p.robot.state["x.vel"] == 0.05)
        if fault == "switch_input":
            from tests.xlerobot_owner.test_server import frame

            await ws.send_json(frame(2))
        elif fault == "focus":
            await ws.send_json(packet(2, focused=False))
        elif fault == "video":
            await ws.send_json(packet(2, video_ready=False))
        elif fault == "reorder":
            await ws.send_json(packet(1))
        await wait_for(lambda: not p.armed and not p.robot.armed, timeout=3)
        assert p.robot.state["x.vel"] == 0
        await ws.send_json(packet(3))
        await asyncio.sleep(0.1)
        assert not p.armed
        await ws.close()


@pytest.mark.asyncio
async def test_keyboard_refuses_non_neutral_wrong_mode_and_competing_owner(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(packet())
        await ws.send_json({"type": "arm", "activation": "keyboard"})
        await wait_for(lambda: p.error is not None)
        assert not p.armed
        await ws.send_json({"type": "set_control_mode", "mode": "drive"})
        await wait_for(lambda: p.mapper.control_mode == "drive")
        await ws.send_json(packet(1, ("KeyW",)))
        await ws.send_json({"type": "arm", "activation": "keyboard"})
        await asyncio.sleep(0.1)
        assert not p.armed
        await ws.send_json(packet(2))
        await ws.send_json({"type": "arm", "activation": "keyboard"})
        await wait_for(lambda: p.armed)
        observer = await c.ws_connect("/ws", headers=HEADERS)
        await observer.send_json(packet())
        await observer.send_json({"type": "arm", "activation": "keyboard"})
        await asyncio.sleep(0.05)
        assert p.status(observer)["control_owner"] == "other"
        await observer.close()
        await ws.close()


@pytest.mark.asyncio
async def test_keyboard_does_not_arm_a_physical_arm_configuration(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        p.robot.metadata.update(source="physical", enable_base=True, allow_motion=True, enabled_arms=["left", "right"])
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json({"type": "set_control_mode", "mode": "drive"})
        await wait_for(lambda: p.mapper.control_mode == "drive")
        await ws.send_json(packet())
        await ws.send_json({"type": "arm", "activation": "keyboard"})
        await wait_for(lambda: p.error is not None)
        assert not p.robot.armed
        assert "独立底盘通道" in p.error
        await ws.close()


def test_base_only_hardware_arm_drive_stop_do_not_write_arm_registers(tmp_path):
    from embodirun_xlerobot_owner.hardware import HardwareRobot

    config = _config(tmp_path, enabled_arms=[], enable_base=True, wheel_directions={"left": 1, "right": -1})
    buses = config["_buses"]
    robot = HardwareRobot(config)
    try:
        robot.connect()
        for bus in buses.values():
            for name in bus.values:
                if not name.startswith("base_"):
                    bus.values[name]["Torque_Enable"] = 1  # existing hold remains untouched
        assert robot.arm()["armed"]
        assert robot.command({"x.vel": 0.03, "theta.vel": 5})["command_accepted"]
        assert robot.stop()["stop_confirmed"]
        assert all(name.startswith("base_") for bus in buses.values() for _, name, _ in bus.writes)
    finally:
        robot.close()


def test_browser_key_state_machine():
    source = Path(__file__).parents[2] / "integrations/xlerobot_owner/src/embodirun_xlerobot_owner/web/drive.js"
    script = """
const assert = require('node:assert/strict');
const { DriveKeys } = require(process.argv[1]);
const k = new DriveKeys();
k.key('KeyW', true); assert.equal(k.keys.size, 0);
k.enable(); k.key('KeyW', true, true); assert.equal(k.keys.size, 0);
k.key('KeyW', true); k.key('KeyA', true); assert.deepEqual([...k.keys], ['KeyW','KeyA']);
k.key('KeyW', false); assert.deepEqual([...k.keys], ['KeyA']);
k.pause(); assert.equal(k.keys.size, 0); assert.equal(k.active, true);
k.key('KeyA', true, true); assert.equal(k.keys.size, 0);
k.key('KeyA', false); k.key('KeyA', true); assert.deepEqual([...k.keys], ['KeyA']);
k.stop(); assert.equal(k.keys.size, 0); assert.equal(k.active, false);
k.key('KeyW', true, true); assert.equal(k.keys.size, 0);
k.enable(); assert.equal(k.keys.size, 0); assert.equal(k.key('KeyX', true), false);
"""
    subprocess.run(["node", "-e", script, str(source)], check=True)


@pytest.mark.asyncio
async def test_delayed_keyboard_packets_pause_without_rearming_or_replaying(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await enable(p, ws)
        original_stamp = next(iter(p.frames.values()))[0].timestamp_ms
        await ws.send_json(packet(1, ("KeyW",)))
        await wait_for(lambda: p.robot.state["x.vel"] == 0.05)
        moving_stamp = next(iter(p.frames.values()))[0].timestamp_ms
        assert moving_stamp > original_stamp
        await asyncio.sleep(0.28)
        # Ordered but buffered key press: discarded, not an owner/session error.
        await ws.send_json(packet(2, ("KeyS",), timestamp_ms=moving_stamp + 1))
        await wait_for(lambda: p.delayed_keyboard_packets == 1)
        await wait_for(lambda: p.robot.state["x.vel"] == 0)
        assert p.armed and p.keyboard_paused and p.error is None
        # Even fresh held keys cannot restart movement after a pause.
        await ws.send_json(packet(3, ("KeyW",)))
        await asyncio.sleep(0.08)
        assert p.robot.state["x.vel"] == 0 and p.keyboard_paused
        await ws.send_json(packet(4))
        await wait_for(lambda: not p.keyboard_paused)
        await ws.send_json(packet(5, ("KeyW",)))
        await wait_for(lambda: p.robot.state["x.vel"] == 0.05)
        assert p.armed and not ws.closed
        await ws.close()


@pytest.mark.asyncio
async def test_keyboard_gap_zeros_at_old_timeout_and_expires_long_gap(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await enable(p, ws)
        await ws.send_json(packet(1, ("KeyW",)))
        await wait_for(lambda: p.robot.state["x.vel"] == 0.05)
        await wait_for(lambda: p.keyboard_paused, timeout=0.6)
        assert p.armed and p.robot.state["x.vel"] == 0
        await wait_for(lambda: not p.armed, timeout=2)
        assert p.error == "controller input timeout" and p.robot.state["x.vel"] == 0
        await ws.send_json(packet(2))
        await asyncio.sleep(0.08)
        assert not p.armed  # True expiration never auto-arms.
        await ws.close()


@pytest.mark.asyncio
async def test_observer_clock_errors_and_mode_requests_do_not_disturb_driver(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await enable(p, ws)
        await ws.send_json(packet(1, ("KeyW",)))
        await wait_for(lambda: p.robot.state["x.vel"] == 0.05)
        owner = p.owner
        observer = await c.ws_connect("/ws", headers=HEADERS)
        await observer.send_json({"type": "set_control_mode", "mode": "drive"})
        await observer.send_json(packet())
        await observer.send_json(packet())  # Observer sequence error.
        await asyncio.sleep(0.05)
        assert p.error is None and p.armed and p.owner is owner
        assert p.status()["input_status"]["keys"] == ["KeyW"]
        await observer.close()
        await ws.close()


@pytest.mark.parametrize(
    "key,signs",
    [
        ("KeyW", (-1, 1)),
        ("KeyS", (1, -1)),
        ("KeyA", (1, 1)),
        ("KeyD", (-1, -1)),
    ],
)
def test_verified_left_inversion_preserves_keyboard_axes(tmp_path, key, signs):
    from embodirun_xlerobot_owner.hardware import HardwareRobot

    config = _config(
        tmp_path, enabled_arms=[], enable_base=True, wheel_directions={"base_left_wheel": -1, "base_right_wheel": 1}
    )
    robot = HardwareRobot(config)
    try:
        robot.connect()
        assert robot.arm()["armed"]
        action = KeyboardFrame.parse(packet(keys=(key,))).action(MappingConfig(enable_base=True))
        assert robot.command(action)["command_accepted"]
        values = config["_buses"]["right"].values
        raw = [values[n]["Goal_Velocity"] for n in ("base_left_wheel", "base_right_wheel")]
        assert tuple(1 if v > 0 else -1 if v < 0 else 0 for v in raw) == signs
    finally:
        robot.close()


@pytest.mark.asyncio
async def test_drive_assets_and_auth(tmp_path):
    async with station(tmp_path) as (_, c):
        page = await c.get("/drive")
        assert page.status == 200
        assert (await page.text()).count("<video ") == 3
        assert (await c.get("/api/status")).status == 401
        assert (await c.get("/api/status", headers={**HEADERS, "Origin": "http://evil.invalid"})).status == 403


@pytest.mark.asyncio
async def test_connection_recovery_clears_old_transport_error_without_arming(tmp_path):
    class DisconnectRobot(DemoRobot):
        disconnected = False

        def read(self):
            if self.disconnected:
                raise RuntimeError("test wire disconnected")
            return super().read()

    robot = DisconnectRobot()
    async with station(tmp_path, robot=robot) as (p, _):
        robot.disconnected = True
        await wait_for(lambda: not p.connected)
        assert p.error == "test wire disconnected"
        robot.disconnected = False
        await wait_for(lambda: p.connected)
        assert p.error is None and not p.armed


@pytest.mark.asyncio
async def test_disconnect_cannot_cancel_stop_while_waiting_for_robot_io(tmp_path):
    async with station(tmp_path) as (p, _):
        p.robot.arm()
        p.armed = True
        p.robot.state["x.vel"] = 0.05
        await p.control_lock.acquire()
        await p.io_lock.acquire()
        stopping = asyncio.create_task(p.stop("test disconnect"))
        try:
            await wait_for(lambda: not p.armed)
            stopping.cancel()
        finally:
            p.io_lock.release()
        try:
            with pytest.raises(asyncio.CancelledError):
                await stopping
            assert not p.robot.armed
            assert p.robot.state["x.vel"] == 0
            assert p.last_feedback["stop_confirmed"] is True
        finally:
            p.control_lock.release()
