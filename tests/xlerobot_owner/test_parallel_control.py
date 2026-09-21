import asyncio

import pytest
from tests.xlerobot_owner.test_server import HEADERS, TOKEN, station

from embodirun_xlerobot_owner.control import JOINT_NAMES
from embodirun_xlerobot_owner.robot import RemoteRobot


def headers(scope, owner):
    return {**HEADERS, "X-Teleop-Scope": scope, "X-Teleop-Owner": owner}


@pytest.mark.asyncio
async def test_remote_clients_share_one_robot_without_sharing_ownership(tmp_path):
    async with station(tmp_path, robot_api=True) as (p, c):
        arms = RemoteRobot(str(c.make_url("/")), TOKEN, scope="arms")
        base = RemoteRobot(str(c.make_url("/")), TOKEN, scope="base")
        observer = RemoteRobot(str(c.make_url("/")), TOKEN, scope="arms")
        for robot in (arms, base, observer):
            await asyncio.to_thread(robot.connect)
        await asyncio.to_thread(arms.arm)
        await asyncio.to_thread(base.arm)
        action = {name: 50.0 if name.endswith("gripper.pos") else 1.0 for name in JOINT_NAMES}
        await asyncio.to_thread(arms.command, action)
        await asyncio.to_thread(base.command, {"x.vel": 0.03, "theta.vel": 0})
        observed, _ = await asyncio.to_thread(observer.read)
        assert not observer.armed and not observed["control_owned"]
        assert observed["control_state"] == {"arms": True, "base": True}
        stopped = await asyncio.to_thread(base.stop)
        assert stopped["stop_confirmed"] and p.robot.control_state() == {"arms": True, "base": False}
        assert all(p.robot.state[n] == action[n] for n in JOINT_NAMES)
        await asyncio.to_thread(base.arm)
        await asyncio.to_thread(base.command, {"x.vel": 0.02, "theta.vel": 0})
        await asyncio.to_thread(arms.stop)
        assert p.robot.control_state() == {"arms": False, "base": True}
        assert p.robot.state["x.vel"] == 0.02
        await asyncio.to_thread(arms.arm)
        # Explicit whole-robot stop from either scoped client.
        await asyncio.to_thread(base.stop_all)
        await asyncio.to_thread(arms.read)
        assert not arms.armed and not any(p.robot.control_state().values())


@pytest.mark.asyncio
async def test_foreign_base_stop_preempts_navigation_without_stopping_arms(tmp_path):
    async with station(tmp_path, robot_api=True) as (p, c):
        arms = RemoteRobot(str(c.make_url("/")), TOKEN, scope="arms")
        navigation = RemoteRobot(str(c.make_url("/")), TOKEN, scope="base")
        human = RemoteRobot(str(c.make_url("/")), TOKEN, scope="base")
        for robot in (arms, navigation, human):
            await asyncio.to_thread(robot.connect)
        await asyncio.to_thread(arms.arm)
        await asyncio.to_thread(navigation.arm)

        stopped = await asyncio.to_thread(human.stop)
        assert stopped["stop_confirmed"]
        assert p.robot.control_state() == {"arms": True, "base": False}

        # The old navigation lease cannot command after the human arms a new
        # owner, and its late owner-scoped cleanup cannot release that owner.
        await asyncio.to_thread(human.arm)
        with pytest.raises(RuntimeError, match="does not own control"):
            await asyncio.to_thread(navigation.command, {"x.vel": 0.03, "theta.vel": 0})
        released = await asyncio.to_thread(navigation.release)
        assert released == {"released": False, "control_owned": False}
        await asyncio.to_thread(human.command, {"x.vel": 0.02, "theta.vel": 0})
        assert p.robot.control_state() == {"arms": True, "base": True}
        await asyncio.to_thread(arms.stop)
        await asyncio.to_thread(human.stop)


@pytest.mark.asyncio
async def test_scope_violations_legacy_exclusivity_and_expired_owner(tmp_path):
    async with station(tmp_path, robot_api=True) as (p, c):
        a, b, old = headers("arms", "a"), headers("base", "b"), headers("all", "old")
        assert (await c.post("/robot/arm", headers=a, json={})).status == 200
        assert (await c.post("/robot/arm", headers=old, json={})).status == 400
        assert (await c.post("/robot/arm", headers=b, json={})).status == 200
        before = dict(p.robot.state)
        response = await c.post("/robot/command", headers=b, json={"action": {"left_arm_gripper.pos": 0}})
        assert response.status == 400
        assert p.robot.state == before
        assert p.robot.control_state() == {"arms": True, "base": False}
        assert (await c.post("/robot/arm", headers=b, json={})).status == 200
        # Simulated hardware watchdog expiry: another owner's arm does not resurrect b.
        await p.call("stop", "base")
        newer = headers("base", "new")
        assert (await c.post("/robot/arm", headers=newer, json={})).status == 200
        response = await c.post("/robot/command", headers=b, json={"action": {"x.vel": 0.03, "theta.vel": 0}})
        assert response.status == 400 and p.robot.control_state()["base"]
        assert (await c.post("/robot/stop_all", headers=b, json={})).status == 200
        assert (await c.post("/robot/arm", headers=old, json={})).status == 200
        assert (await c.post("/robot/arm", headers=a, json={})).status == 400
        assert (await c.post("/robot/stop", headers=b, json={})).status == 400
        assert p.robot.armed
        assert (await c.post("/robot/stop_all", headers=b, json={})).status == 200


@pytest.mark.asyncio
async def test_cancelled_scoped_stop_still_finishes_without_stopping_arms(tmp_path):
    async with station(tmp_path, robot_api=True) as (p, c):
        for scope in ("arms", "base"):
            await c.post("/robot/arm", headers=headers(scope, scope), json={})
        async with p.control_lock:
            await p.io_lock.acquire()
            pending = asyncio.create_task(p.stop_scope("base"))
            await asyncio.sleep(0.01)
            pending.cancel()
            await asyncio.sleep(0.01)
            assert not pending.done()
            p.io_lock.release()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert p.robot.control_state() == {"arms": True, "base": False}
        assert p.robot_owners == {"arms": "arms"}


@pytest.mark.asyncio
async def test_two_video_readers_and_owner_scoped_release_never_stop_other_driver(tmp_path):
    async with station(tmp_path, robot_api=True) as (p, c):
        navigation = headers("base", "navigation")
        gamepad = headers("base", "gamepad")
        assert (await c.post("/robot/arm", headers=navigation, json={})).status == 200
        images = await asyncio.gather(
            c.get("/robot/observe", headers=navigation),
            c.get("/robot/observe", headers=gamepad),
        )
        first, second = [await response.json() for response in images]
        assert len(first["images"]) == len(second["images"]) == 3
        assert all(first["images"].values()) and all(second["images"].values())
        assert first["observation"]["control_owned"]
        assert not second["observation"]["control_owned"]
        assert (await c.post("/robot/arm", headers=gamepad, json={})).status == 400
        result = await (await c.post("/robot/release", headers=gamepad, json={})).json()
        assert result == {"released": False, "control_owned": False}
        assert p.robot.control_state()["base"]
        released = await (await c.post("/robot/release", headers=navigation, json={})).json()
        assert released["stop_confirmed"] and released["released"]
        assert (await c.post("/robot/arm", headers=gamepad, json={})).status == 200
        # A late cleanup from the former owner cannot stop the new owner.
        result = await (await c.post("/robot/release", headers=navigation, json={})).json()
        assert result["control_owned"] is False
        assert p.robot_owners["base"] == "gamepad"


@pytest.mark.asyncio
async def test_viewing_gateway_stop_does_not_stop_navigation_owner(tmp_path):
    async with station(tmp_path / "robot", robot_api=True) as (robot_platform, robot_client):
        navigation = headers("base", "navigation")
        await robot_client.post("/robot/arm", headers=navigation, json={})
        remote = RemoteRobot(str(robot_client.make_url("/")), TOKEN, scope="base")
        async with station(tmp_path / "gateway", robot=remote) as (gateway, _):
            assert gateway.status()["control_owner"] == "other"
            result = await gateway.stop("observer tab closed")
            assert result["control_owned"] is False
            assert not gateway.stop_unconfirmed
            assert robot_platform.robot.control_state()["base"]
