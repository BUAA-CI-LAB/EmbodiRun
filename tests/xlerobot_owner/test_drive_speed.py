import asyncio

import pytest
from tests.xlerobot_owner.test_hardware import _config
from tests.xlerobot_owner.test_keyboard import enable, packet
from tests.xlerobot_owner.test_server import HEADERS, station, wait_for

from embodirun_xlerobot_owner.control import MappingConfig
from embodirun_xlerobot_owner.hardware import HardwareRobot
from embodirun_xlerobot_owner.keyboard import KeyboardFrame


async def set_speed(ws, linear=0.1, angular=20):
    await ws.send_json({"type": "set_drive_speed", "linear_m_s": linear, "angular_deg_s": angular})
    async with asyncio.timeout(1):
        while True:
            reply = await ws.receive_json()
            if reply.get("type") == "drive_speed_result":
                return reply


@pytest.mark.parametrize(
    "keys,expected",
    [
        (("KeyW",), (0.12, 0)),
        (("KeyS",), (-0.12, 0)),
        (("KeyA",), (0, 24)),
        (("KeyD",), (0, -24)),
        (("KeyW", "KeyD"), (0.12, -24)),
        ((), (0, 0)),
    ],
)
def test_selected_speed_maps_both_directions_without_arm_fields(keys, expected):
    action = KeyboardFrame.parse(packet(keys=keys)).action(
        MappingConfig(enable_base=True, max_linear_m_s=0.15, max_angular_deg_s=30),
        linear_m_s=0.12,
        angular_deg_s=24,
    )
    assert action == dict(zip(("x.vel", "theta.vel"), expected))


@pytest.mark.asyncio
async def test_speed_applies_without_arming_and_is_per_browser(tmp_path):
    mapping = MappingConfig(enable_base=True, max_linear_m_s=0.15, max_angular_deg_s=30)
    async with station(tmp_path, mapping=mapping) as (p, c):
        a = await c.ws_connect("/ws", headers=HEADERS)
        b = await c.ws_connect("/ws", headers=HEADERS)
        await a.send_json({"type": "set_control_mode", "mode": "drive"})
        await wait_for(lambda: p.mapper.control_mode == "drive")
        assert (await set_speed(a))["ok"]
        assert not p.armed and p.last_action is None and p.robot.state["x.vel"] == 0
        assert sorted(v["linear_m_s"] for v in p.keyboard_speeds.values()) == [0.1]
        assert p.status()["drive_speed"] == {"linear_m_s": 0.05, "angular_deg_s": 10}
        assert p.mapper.config.max_linear_m_s == 0.15  # Slider cannot rewrite the cap.
        await a.close()
        await wait_for(lambda: not p.keyboard_speeds)
        await b.close()


@pytest.mark.asyncio
async def test_release_then_adjust_while_enabled_and_observer_cannot_change_driver(tmp_path):
    mapping = MappingConfig(enable_base=True, max_linear_m_s=0.15, max_angular_deg_s=30)
    async with station(tmp_path, mapping=mapping) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await enable(p, ws)
        assert (await set_speed(ws))["ok"]
        await ws.send_json(packet(1, ("KeyW",)))
        await wait_for(lambda: p.last_action == {"x.vel": 0.1, "theta.vel": 0})
        assert not (await set_speed(ws, 0.15, 30))["ok"]  # Can't accelerate a held key.
        assert p.armed and p.error is None
        observer = await c.ws_connect("/ws", headers=HEADERS)
        assert not (await set_speed(observer, 0.02, 5))["ok"]
        assert p.keyboard_speed(p.owner)["linear_m_s"] == 0.1
        await ws.send_json(packet(2))
        await wait_for(lambda: p.last_action["x.vel"] == 0)
        assert (await set_speed(ws, 0.15, 30))["ok"]
        assert p.armed
        await ws.send_json(packet(3, ("KeyS", "KeyD")))
        await wait_for(lambda: p.last_action == {"x.vel": -0.15, "theta.vel": -30})
        await ws.close()
        await observer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "linear,angular", [(0.151, 10), (0.1, 31), (0, 10), (-1, 10), (True, 10), (".1", 10), (0.1, None)]
)
async def test_invalid_speed_never_changes_selection_or_enables(tmp_path, linear, angular):
    async with station(
        tmp_path, mapping=MappingConfig(enable_base=True, max_linear_m_s=0.15, max_angular_deg_s=30)
    ) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json({"type": "set_control_mode", "mode": "drive"})
        await wait_for(lambda: p.mapper.control_mode == "drive")
        assert not (await set_speed(ws, linear, angular))["ok"]
        assert not p.keyboard_speeds and not p.armed and p.error is None
        await ws.close()


@pytest.mark.asyncio
async def test_physical_caps_are_intersection_and_unknown_endpoint_stays_slow(tmp_path):
    async with station(
        tmp_path, mapping=MappingConfig(enable_base=True, max_linear_m_s=0.15, max_angular_deg_s=30)
    ) as (p, _):
        p.robot.metadata["source"] = "physical"
        assert p.keyboard_speed_limits() == {"linear_m_s": 0.05, "angular_deg_s": 10}
        p.robot.metadata["base_velocity_limits"] = {"linear_m_s": 0.1, "angular_deg_s": 90}
        assert p.keyboard_speed_limits() == {"linear_m_s": 0.1, "angular_deg_s": 30}
        client = object()
        p.keyboard_speeds[client] = {"linear_m_s": 0.15, "angular_deg_s": 30}
        assert p.keyboard_speed(client)["linear_m_s"] == 0.1


def test_hardware_exposes_actual_caps_and_still_clamps_at_motor_boundary(tmp_path):
    robot = HardwareRobot(
        _config(
            tmp_path,
            enabled_arms=[],
            enable_base=True,
            wheel_directions={"left": -1, "right": 1},
            max_linear_m_s=0.15,
            max_angular_deg_s=30,
        )
    )
    try:
        robot.connect()
        assert robot.metadata["base_velocity_limits"] == {"linear_m_s": 0.15, "angular_deg_s": 30}
        assert robot.arm()["armed"]
        feedback = robot.command({"x.vel": 0.9, "theta.vel": 90})
        assert feedback["command_accepted"]
        assert feedback["applied_action"] == {"x.vel": 0.15, "theta.vel": 30}
        assert max(abs(v) for v in feedback["raw_action"].values()) <= 3000
    finally:
        robot.close()
