import asyncio
import copy
import sys
import time
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer

from embodirun_xlerobot_owner.control import MappingConfig
from embodirun_xlerobot_owner.robot import DemoRobot, RemoteRobot
from embodirun_xlerobot_owner.server import Platform, create_app, xr_diagnostics

TOKEN = "test-token-not-a-real-secret"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def frame(seq=0, grip=False, position=0):
    c = {
        "position": [position, 1, -0.5],
        "orientation": [0, 0, 0, 1],
        "tracked": True,
        "grip": grip,
        "trigger": 0.0,
        "thumbstick": [0, 0],
    }
    return {
        "type": "input",
        "seq": seq,
        # Integration requests are event-driven, not guaranteed to arrive at
        # exactly 20 Hz. Model the browser's real clock; InputClock unit tests
        # separately cover synthetic late/reordered timestamps.
        "timestamp_ms": time.monotonic() * 1000,
        "controllers": {"left": copy.deepcopy(c), "right": copy.deepcopy(c)},
    }


@asynccontextmanager
async def station(
    tmp_path,
    robot=None,
    robot_api=False,
    mapping=None,
    browser_no_token_cidr=None,
    leader_command=None,
):
    robot = robot or DemoRobot()
    if isinstance(robot, DemoRobot) and type(robot).read is DemoRobot.read:
        # These are control/HTTP tests, not JPEG-rendering benchmarks. Keep
        # valid images while avoiding repeated image rendering starving the
        # real 350 ms watchdog under parallel desktop workloads.
        _, demo_images = robot.read()

        def read_test_demo():
            now = time.time_ns()
            return {
                "state": dict(robot.state),
                "source_timestamp_ns": now,
                "state_timestamp_ns": now,
                "camera_timestamps_ns": dict.fromkeys(demo_images, now),
                "metadata": robot.metadata,
                "armed": robot.armed,
                "control_state": robot.control_state(),
            }, dict(demo_images)

        robot.read = read_test_demo
    platform = Platform(
        robot,
        TOKEN,
        output=tmp_path,
        fps=20,
        robot_api=robot_api,
        mapping=mapping,
        browser_no_token_cidr=browser_no_token_cidr,
        leader_command=leader_command,
    )
    client = TestClient(TestServer(create_app(platform)))
    await client.start_server()
    for _ in range(100):
        if platform.connected:
            break
        await asyncio.sleep(0.01)
    try:
        yield platform, client
    finally:
        await client.close()


async def wait_for(predicate, timeout=2):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition timed out")
        await asyncio.sleep(0.01)


async def websocket_error(ws):
    async with asyncio.timeout(1):
        while True:
            message = await ws.receive_json()
            if message.get("type") == "error":
                return message["error"]


@pytest.mark.asyncio
async def test_explicit_web_start_runs_leader_and_owner_disconnect_stops_it(tmp_path):
    robot = DemoRobot()
    robot.mode = "remote"
    robot.scope = "base"
    robot.metadata.update(source="physical", allow_motion=True, enable_base=True)
    command = (
        sys.executable,
        "-u",
        "-c",
        "import time; print('同步完成；测试进程', flush=True); time.sleep(30)",
    )
    async with station(tmp_path, robot=robot, leader_command=command) as (platform, client):
        ws = await client.ws_connect("/ws", headers=HEADERS)
        await ws.receive_json()
        await ws.send_json({"type": "leader_start"})
        await wait_for(lambda: platform.leader.running)
        await wait_for(lambda: platform.leader.status()["state"] == "following")
        assert platform.leader_owner is not None

        await ws.close()
        await wait_for(lambda: not platform.leader.running)
        assert platform.leader.status()["state"] == "stopped"


@pytest.mark.asyncio
async def test_lan_without_token_keeps_origin_host_and_robot_auth_boundaries(tmp_path):
    async with station(tmp_path, browser_no_token_cidr="127.0.0.0/8") as (p, c):
        response = await c.get("/api/status")
        assert response.status == 200
        assert (await response.json())["browser_access"] == "trusted_lan"
        assert (await c.get("/api/cameras/front.jpg")).status == 200
        assert (await c.get("/api/status", headers={"Origin": "https://evil.invalid"})).status == 403
        assert (
            await c.get("/api/status", headers={"Host": "evil.invalid", "Origin": "http://evil.invalid"})
        ).status == 401
        assert (await c.get("/robot/status")).status == 401
        assert (await c.get("/robot/status", headers=HEADERS)).status == 404
        ws = await c.ws_connect("/ws")
        await ws.send_json(frame(grip=True))
        await wait_for(lambda: bool(p.frames))
        assert not p.armed and p.last_action is None  # connection/input alone cannot arm
        await ws.close()


@pytest.mark.asyncio
async def test_other_subnet_and_forwarded_headers_cannot_enable_token_free_access(tmp_path):
    async with station(tmp_path, browser_no_token_cidr="192.168.1.0/24") as (p, c):
        assert p.browser_address_allowed("192.168.1.77")
        assert not p.browser_address_allowed("192.168.2.77")
        assert not p.browser_address_allowed(None)
        assert not p.browser_address_allowed("invalid")
        assert (
            await c.get(
                "/api/status",
                headers={
                    "X-Forwarded-For": "192.168.1.77",
                    "Forwarded": "for=192.168.1.77",
                },
            )
        ).status == 401
        assert (await c.get("/api/status", headers=HEADERS)).status == 200


@pytest.mark.parametrize("network", ["0.0.0.0/0", "8.8.8.0/24", "::/0", "192.168.1.7/24"])
def test_no_token_network_must_be_explicit_private_subnet(tmp_path, network):
    with pytest.raises(ValueError):
        Platform(DemoRobot(), TOKEN, output=tmp_path, browser_no_token_cidr=network)


def test_robot_endpoint_cannot_disable_token_auth(tmp_path):
    with pytest.raises(ValueError, match="robot endpoint"):
        Platform(DemoRobot(), TOKEN, output=tmp_path, robot_api=True, browser_no_token_cidr="127.0.0.0/8")
    robot = DemoRobot()
    robot.mode = "hardware"
    with pytest.raises(ValueError, match="robot endpoint"):
        Platform(robot, TOKEN, output=tmp_path, browser_no_token_cidr="127.0.0.0/8")


@pytest.mark.asyncio
async def test_grip_activation_has_no_hold_timer_but_requires_explicit_request(tmp_path):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame(grip=True))
        await wait_for(lambda: bool(p.frames))
        assert not p.armed
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.error is not None)
        assert not p.armed  # legacy neutral activation still requires neutral
        await ws.send_json({"type": "arm", "activation": "grip"})
        await wait_for(lambda: p.armed, timeout=0.5)
        assert p.owner is not None
        await ws.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["trigger", "stick", "tracking", "no_grip", "stale", "stop"])
async def test_grip_activation_keeps_existing_interlocks(tmp_path, fault):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        packet = frame(grip=fault != "no_grip")
        right = packet["controllers"]["right"]
        if fault == "trigger":
            right["trigger"] = 1
        elif fault == "stick":
            right["thumbstick"] = [0.5, 0]
        elif fault == "tracking":
            right["tracked"] = False
        elif fault == "stop":
            p.stop_unconfirmed = True
        await ws.send_json(packet)
        await wait_for(lambda: bool(p.frames))
        if fault == "stale":
            await asyncio.sleep(0.37)
        await ws.send_json({"type": "arm", "activation": "grip"})
        await wait_for(lambda: p.error is not None)
        assert not p.armed and p.owner is None
        await ws.close()


@pytest.mark.asyncio
async def test_command_receipt_rebases_mapper_hold_in_live_tick(tmp_path):
    class QuantizedRobot(DemoRobot):
        def command(self, action):
            quantized = {k: round(v * 4) / 4 for k, v in action.items()}
            return super().command(quantized)

    async with station(tmp_path, robot=QuantizedRobot()) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame(grip=True))
        await ws.send_json({"type": "arm", "activation": "grip"})
        await wait_for(lambda: bool(p.mapper.anchors))
        await ws.send_json(frame(1, grip=True, position=0.0023))
        pan = "left_arm_shoulder_pan.pos"
        # State mutates in the worker before tick receives/accepts the receipt.
        await wait_for(lambda: p.last_feedback and p.last_feedback.get("applied_action", {}).get(pan, 0) > 0)
        assert p.mapper.last_targets[pan] == p.last_feedback["applied_action"][pan] == 0.25
        await ws.send_json(frame(2, grip=False, position=0.0023))
        await wait_for(lambda: not p.mapper.anchors)
        assert p.mapper.last_targets[pan] == p.robot.state[pan] == 0.25
        await ws.close()


def test_xr_diagnostics_are_bounded_and_do_not_echo_unknown_fields():
    result = xr_diagnostics(
        {
            "active": "true",
            "visibility": "visible",
            "arbitrary_secret": "never echo",
            "sources": [{"hand": "left", "gamepad": True, "profiles": ["x" * 500] * 8}] * 20,
        }
    )
    assert "active" not in result
    assert "arbitrary_secret" not in result
    assert len(result["sources"]) == 6
    assert len(result["sources"][0]["profiles"]) == 4
    assert len(result["sources"][0]["profiles"][0]) == 80
    assert xr_diagnostics(None) == {}


@pytest.mark.asyncio
async def test_partial_startup_tracking_is_observable_but_never_arms(tmp_path):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        packet = frame()
        packet["controllers"]["right"]["tracked"] = False
        packet["xr"] = {
            "active": True,
            "visibility": "visible",
            "document_hidden": True,
            "viewer_tracked": True,
            "safety_tripped": False,
        }
        await ws.send_json(packet)
        await wait_for(lambda: bool(p.frames))
        status = p.status()
        assert status["input_status"]["connected_browsers"] == 1
        assert status["input_status"]["tracked"] == {"left": True, "right": False}
        assert status["input_status"]["xr"]["document_hidden"] is True
        assert not p.armed and p.last_action is None
        await ws.send_json(frame(1))
        await wait_for(lambda: p.status()["input_status"]["seq"] == 1)
        assert not p.armed and p.last_action is None
        await ws.close()


@pytest.mark.asyncio
async def test_actual_control_stops_on_tracking_loss_and_does_not_auto_resume(tmp_path):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame())
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.armed)
        packet = frame(1)
        packet["controllers"]["left"]["tracked"] = False
        packet["xr"] = {"active": True, "visibility": "visible"}
        await ws.send_json(packet)
        # Stop clears armed before the asynchronous hardware stop returns;
        # an earlier command receipt is not a stop acknowledgment.
        await wait_for(lambda: not p.armed and (p.last_feedback or {}).get("stop_confirmed") is True)
        assert p.last_feedback["stop_confirmed"] is True
        assert "tracking lost" in p.error
        await ws.send_json(frame(2))
        await wait_for(lambda: p.status()["input_status"]["seq"] == 2)
        assert not p.armed and p.owner is None
        await ws.close()


@pytest.mark.asyncio
async def test_drive_mode_requires_both_configurations_and_stopped_state(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, _c):
        assert p.status()["drive_available"] is True
        p.set_control_mode("drive")
        assert p.status()["control_mode"] == "drive"
        p.robot.metadata.update(source="physical", allow_motion=True, enable_base=False)
        p.set_control_mode("arms")
        assert p.status()["drive_available"] is False
        with pytest.raises(RuntimeError, match="AGX"):
            p.set_control_mode("drive")
        p.robot.metadata["enable_base"] = True
        assert p.status()["drive_available"] is True
        p.set_control_mode("drive")
        assert p.selected_action({**p.robot.state, "x.vel": 0.03, "theta.vel": 5}) == {"x.vel": 0.03, "theta.vel": 5}
        p.set_control_mode("arms")
        p.stop_unconfirmed = True
        with pytest.raises(RuntimeError, match="unconfirmed"):
            p.set_control_mode("drive")
        p.stop_unconfirmed = False
        p.robot.armed = True
        with pytest.raises(RuntimeError, match="Stop"):
            p.set_control_mode("drive")
        p.robot.armed = False
        p.recorder.start("ongoing episode", p.robot.metadata)
        with pytest.raises(RuntimeError, match="recording"):
            p.set_control_mode("drive")
        p.recorder.finish(success=None)
        p.mapper.config.enable_base = False
        assert p.status()["drive_available"] is False
        with pytest.raises(RuntimeError, match="Mac"):
            p.set_control_mode("drive")


@pytest.mark.asyncio
async def test_driving_via_websocket_keeps_cameras_and_stops_on_deadman_release(tmp_path):
    async with station(tmp_path, mapping=MappingConfig(enable_base=True)) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        observer = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame())
        await wait_for(lambda: ws is not None and bool(p.frames))
        await ws.send_json({"type": "set_control_mode", "mode": "drive"})
        await wait_for(lambda: p.mapper.control_mode == "drive")
        assert not p.frames
        assert not p.armed
        assert (await c.get("/api/cameras/front.jpg", headers=HEADERS)).status == 200
        await ws.send_json(frame(1))
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.armed)
        before = dict(p.robot.state)
        # Continuous input is necessary while the real watchdog is active.
        # Simulate 20 Hz rather than relying on an isolated packet surviving
        # scheduler load while assertions await other state.
        sequence = 2
        driving = True

        sequence = 2

        async def timed_input_stream():
            nonlocal sequence
            while True:
                data = frame(sequence, grip=driving, position=0.7)
                data["controllers"]["right"]["thumbstick"] = [1, -1]
                await ws.send_json(data)
                sequence += 1
                await asyncio.sleep(0.05)

        task = asyncio.create_task(timed_input_stream())
        try:
            await wait_for(lambda: p.robot.state["x.vel"] > 0)
            assert p.robot.state["theta.vel"] < 0
            assert all(p.robot.state[name] == value for name, value in before.items() if name.endswith(".pos"))
            await observer.send_json({"type": "set_control_mode", "mode": "arms"})
            assert "Stop" in await websocket_error(observer)
            assert p.error is None
            assert p.mapper.control_mode == "drive"
            assert p.armed
            driving = False
            await wait_for(lambda: p.robot.state["x.vel"] == 0)
            assert p.robot.state["theta.vel"] == 0
            assert p.armed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await wait_for(lambda: not p.armed)
        await wait_for(lambda: p.last_feedback and p.last_feedback.get("stop_confirmed") is True)
        assert p.robot.state["x.vel"] == 0
        await ws.send_json({"type": "set_control_mode", "mode": "arms"})
        await wait_for(lambda: p.mapper.control_mode == "arms")
        assert not p.armed
        await observer.close()
        await ws.close()


@pytest.mark.asyncio
async def test_duplicate_observation_is_not_a_new_recorded_sample(tmp_path):
    from embodirun_xlerobot_owner.recording import validate_episode

    p = Platform(DemoRobot(), TOKEN, output=tmp_path)
    p.robot.connect()
    observation, images = p.robot.read()
    path = p.recorder.start("read-only duplicate check", p.robot.metadata)
    await p.record_frame(observation=observation, images=images, action=None)
    await p.record_frame(observation=observation, images=images, action=None)
    assert p.skipped_duplicate_samples == 1
    with pytest.raises(RuntimeError, match="clock regressed"):
        await p.record_frame(observation={**observation, "source_timestamp_ns": 1}, images=images, action=None)
    p.recorder.finish(success=None)
    assert validate_episode(path)["frame_count"] == 1


@pytest.mark.asyncio
async def test_pairing_code_single_use_expiry_lockout_and_robot_separation(tmp_path):
    async with station(tmp_path, robot_api=True) as (p, c):
        code = p.issue_pairing_code()
        assert len(code) == 6 and code.isdigit()
        assert (await c.get("/robot/status", headers={"Authorization": "Bearer " + code})).status == 401
        assert (await c.post("/api/session", json={"token": code})).status == 200
        assert (await c.post("/api/session", json={"token": code})).status == 401
        code = p.issue_pairing_code()
        p.pairing_expires_at = 0
        assert (await c.post("/api/session", json={"token": code})).status == 401
        code = p.issue_pairing_code()
        for _ in range(5):
            assert (await c.post("/api/session", json={"token": "wrong"})).status == 401
        assert (await c.post("/api/session", json={"token": code})).status == 401
        assert (await c.post("/api/session", json={"token": TOKEN})).status == 200


@pytest.mark.asyncio
async def test_auth_origin_and_camera_bytes(tmp_path):
    async with station(tmp_path) as (_p, c):
        assert (await c.get("/api/status")).status == 401
        assert (await c.post("/api/session", json={"token": "wrong"})).status == 401
        assert (
            await c.post("/api/session", json={"token": TOKEN}, headers={"Origin": "https://evil.invalid"})
        ).status == 403
        response = await c.get("/api/cameras/front.jpg", headers=HEADERS)
        assert response.status == 200
        assert (await response.read()).startswith(b"\xff\xd8")
        assert response.headers["X-Camera-Timestamp-Ns"]
        assert (await c.ws_connect("/ws", headers=HEADERS)).closed is False


@pytest.mark.asyncio
async def test_remote_unavailable_at_start_recovers_without_arming(tmp_path):
    class RebootingRemote(DemoRobot):
        mode = "remote"
        up = False

        def connect(self):
            raise ConnectionError("AGX restarting")

        def read(self):
            if not self.up:
                raise ConnectionError("AGX restarting")
            return super().read()

    robot = RebootingRemote()
    platform = Platform(robot, TOKEN, output=tmp_path)
    client = TestClient(TestServer(create_app(platform)))
    await client.start_server()
    try:
        assert (await client.get("/")).status == 200
        assert not platform.connected
        robot.up = True
        await wait_for(lambda: platform.connected)
        assert not platform.armed
        assert platform.connection_error is None
        assert platform.error is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_controller_owner_timeout_and_observer_disconnect(tmp_path):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        observer = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame())
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.armed)
        await observer.close()
        assert p.armed
        await ws.send_json(frame(1, grip=True))
        await wait_for(lambda: "left" in p.mapper.anchors)
        await ws.send_json(frame(2, grip=True, position=0.01))
        await wait_for(lambda: p.robot.state["left_arm_shoulder_pan.pos"] > 0)
        await wait_for(lambda: not p.armed)
        await wait_for(lambda: p.last_feedback.get("stop_confirmed") is True)
        assert p.last_feedback["stop_confirmed"]
        assert "timeout" in p.error
        assert p.robot.state["x.vel"] == 0
        await ws.close()


@pytest.mark.asyncio
async def test_tracking_loss_and_old_frames_stop(tmp_path):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame())
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.armed)
        await ws.send_json(frame())
        await wait_for(lambda: not p.armed)
        assert "reordered" in p.error
        await ws.close()


@pytest.mark.asyncio
async def test_second_browser_cannot_take_control(tmp_path):
    async with station(tmp_path) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        other = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json(frame())
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.armed)
        owner = p.owner
        await other.send_json(frame())
        await other.send_json({"type": "arm"})
        assert "owned" in await websocket_error(other)
        assert p.error is None
        assert p.owner is owner
        assert p.armed
        await other.send_json({"type": "stop"})
        await wait_for(lambda: not p.armed)


@pytest.mark.asyncio
async def test_actual_http_mac_to_robot_chain(tmp_path):
    async with station(tmp_path / "robot", robot_api=True) as (robot_p, robot_client):
        url = str(robot_client.make_url("/")).rstrip("/")
        remote = RemoteRobot(url, TOKEN)
        async with station(tmp_path / "mac", remote) as (p, c):
            assert p.observation["state"] == robot_p.observation["state"]
            assert p.images["front"].startswith(b"\xff\xd8")
            ws = await c.ws_connect("/ws", headers=HEADERS)
            await ws.send_json(frame())
            await ws.send_json({"type": "arm"})
            await wait_for(lambda: p.armed)
            await ws.send_json(frame(1, grip=True))
            await wait_for(lambda: "left" in p.mapper.anchors)
            await ws.send_json(frame(2, grip=True, position=0.02))
            await wait_for(lambda: robot_p.robot.state["left_arm_shoulder_pan.pos"] > 0)
            await ws.send_json({"type": "stop"})
            await wait_for(lambda: not robot_p.robot.armed)
            assert not p.armed


@pytest.mark.asyncio
async def test_failed_stop_blocks_rearm(tmp_path):
    class UncertainRobot(DemoRobot):
        def stop(self):
            self.armed = False
            return {"stop_confirmed": False}

    async with station(tmp_path, UncertainRobot()) as (p, c):
        ws = await c.ws_connect("/ws", headers=HEADERS)
        await ws.send_json({"type": "stop"})
        await wait_for(lambda: p.stop_unconfirmed)
        await ws.send_json(frame())
        await ws.send_json({"type": "arm"})
        await wait_for(lambda: p.error is not None)
        assert not p.armed
        assert "unconfirmed" in p.error


@pytest.mark.asyncio
async def test_read_only_recording_retains_raw_robot_images(tmp_path):
    async with station(tmp_path) as (p, c):
        response = await c.post("/api/record/start", headers=HEADERS, json={"task": "observe without motion"})
        assert response.status == 200, await response.text()
        assert p.recorder.active
        await asyncio.sleep(0.2)
        response = await c.post("/api/record/stop", headers=HEADERS, json={"success": None})
        assert response.status == 200, await response.text()
        assert not p.recorder.active
        assert list(tmp_path.rglob("*.jpg"))
        assert not p.robot.armed
