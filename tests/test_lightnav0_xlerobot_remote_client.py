from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace

import pytest

from embodirun.bindings.xlerobot.lightnav0 import LocalSegmentConfig
from embodirun.bindings.xlerobot.lightnav0.cli import (
    TeleopRunConfig,
    resolve_robot_factory,
    run_local_segment_loop,
)
from embodirun.robots.xlerobot.remote_client import (
    RemoteRobot,
    RemoteRobotError,
    build_remote_robot_from_env,
)


class _Response:
    def __init__(self, payload):
        self._stream = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self._stream

    def __exit__(self, *_args):
        self._stream.close()


class _Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(
            {
                "url": request.full_url,
                "authorization": request.get_header("Authorization"),
                "owner": request.get_header("X-teleop-owner"),
                "scope": request.get_header("X-teleop-scope"),
                "timeout": timeout,
                "body": None if request.data is None else json.loads(request.data),
            }
        )
        return _Response(next(self.responses))


def _responses():
    image = base64.b64encode(b"jpeg").decode()
    return [
        {"metadata": {"control_scopes": ["base"], "source": "physical"}},
        {"armed": True},
        {
            "observation": {
                "state": {"x.vel": 0.1, "theta.vel": 2.0},
                "state_timestamp_ns": 10,
                "state_cached": False,
                "camera_status": {"front": {"fresh": True, "timestamp_ns": 11}},
                "errors": [],
                "armed": True,
                "control_owned": True,
            },
            "images": {"front": image},
        },
        {"accepted": True, "command_accepted": True, "errors": []},
        {
            "stop_confirmed": True,
            "stationary_confirmed": True,
            "physical_outcome": "stopped",
            "errors": [],
        },
    ]


def test_remote_client_matches_http_control_sequence_without_hardware():
    robot = RemoteRobot("http://orin:8080", "secret", scope="base")
    opener = _Opener(_responses())
    robot.opener = opener

    robot.connect()
    assert robot.armed is False
    robot.arm()
    observation, images = robot.read()
    assert images == {"front": b"jpeg"}
    assert observation["timestamp_domains"]["camera"] == "robot"
    assert observation["state_timestamp_ns"] == 10
    command = robot.command({"x.vel": 0.1, "theta.vel": 2.0})
    assert command["command_accepted"] is True
    stop = robot.stop()
    assert stop["stop_confirmed"] is True
    assert robot.armed is False

    assert [request["url"].rsplit("/", 1)[-1] for request in opener.requests] == [
        "status",
        "arm",
        "observe",
        "command",
        "stop",
    ]
    assert opener.requests[0]["authorization"] == "Bearer secret"
    assert opener.requests[0]["scope"] == "base"
    assert opener.requests[3]["body"] == {"action": {"x.vel": 0.1, "theta.vel": 2.0}}


def test_env_factory_connects_but_does_not_arm_by_default(monkeypatch):
    monkeypatch.setenv("TEST_XLEROBOT_TOKEN", "secret")
    requests = []

    def fake_request(self, endpoint, data=None, *, timeout=None):
        del data, timeout
        requests.append(endpoint)
        return {"metadata": {"control_scopes": ["base"]}} if endpoint == "status" else {}

    monkeypatch.setattr(RemoteRobot, "_request", fake_request)
    robot = build_remote_robot_from_env(
        "http://orin:8080",
        token_env="TEST_XLEROBOT_TOKEN",
        authorize_motion=False,
    )
    assert robot.armed is False
    assert requests == ["status"]
    robot.close()


def test_env_factory_explicit_authorization_calls_arm(monkeypatch):
    monkeypatch.setenv("TEST_XLEROBOT_TOKEN", "secret")
    requests = []

    def fake_request(self, endpoint, data=None, *, timeout=None):
        del data, timeout
        requests.append(endpoint)
        if endpoint == "status":
            return {"metadata": {"control_scopes": ["base"]}}
        if endpoint == "arm":
            return {"armed": True}
        raise AssertionError(endpoint)

    monkeypatch.setattr(RemoteRobot, "_request", fake_request)
    robot = build_remote_robot_from_env(
        "http://orin:8080",
        token_env="TEST_XLEROBOT_TOKEN",
        authorize_motion=True,
    )
    assert robot.armed is True
    assert requests == ["status", "arm"]
    robot.armed = False
    robot.close()


def test_env_factory_requires_token_without_connecting(monkeypatch):
    monkeypatch.delenv("TEST_XLEROBOT_TOKEN", raising=False)
    with pytest.raises(RemoteRobotError, match="environment variable"):
        build_remote_robot_from_env(
            "http://orin:8080",
            token_env="TEST_XLEROBOT_TOKEN",
        )


def test_builtin_cli_factory_is_lazy_until_called(monkeypatch):
    calls = []
    sentinel = object()

    def fake_build(url, **kwargs):
        calls.append((url, kwargs))
        return sentinel

    monkeypatch.setattr(
        "embodirun.bindings.xlerobot.lightnav0.cli.build_remote_robot_from_env",
        fake_build,
    )
    args = type(
        "Args",
        (),
        {
            "robot_factory": None,
            "robot_url": "http://orin:8080",
            "robot_token_env": "TEST_XLEROBOT_TOKEN",
            "robot_timeout_s": 3.0,
            "robot_scope": "base",
            "authorize_motion": True,
        },
    )()
    factory = resolve_robot_factory(args)
    assert calls == []
    assert factory() is sentinel
    assert calls == [
        (
            "http://orin:8080",
            {
                "token_env": "TEST_XLEROBOT_TOKEN",
                "timeout": 3.0,
                "scope": "base",
                "authorize_motion": True,
            },
        )
    ]


class _LeaseAwareOpener:
    """Offline HTTP transport that revokes the lease only on stop."""

    def __init__(self, *, lose_lease_on_observe: int | None = None):
        self.armed = False
        self.observe_count = 0
        self.arm_count = 0
        self.stop_count = 0
        self.commands = []
        self.cached_observe_count = 0
        self.lose_lease_on_observe = lose_lease_on_observe
        self.last_observe_mono = None
        self.last_state_timestamp = 0
        self._image = self._jpeg()

    @staticmethod
    def _jpeg():
        from PIL import Image

        image = Image.new("RGB", (4, 4), "red")
        output = io.BytesIO()
        image.save(output, format="JPEG")
        return output.getvalue()

    def open(self, request, timeout):
        del timeout
        endpoint = request.full_url.rsplit("/", 1)[-1]
        if endpoint == "status":
            payload = {"metadata": {"control_scopes": ["base"]}}
        elif endpoint == "arm":
            self.armed = True
            self.arm_count += 1
            payload = {"armed": True}
        elif endpoint == "observe":
            self.observe_count += 1
            owned = self.armed
            if self.lose_lease_on_observe == self.observe_count:
                owned = False
                self.armed = False
            now = time.monotonic()
            state_cached = self.last_observe_mono is not None and now - self.last_observe_mono < 0.02
            if self.lose_lease_on_observe == self.observe_count:
                state_cached = False
            if not state_cached:
                self.last_state_timestamp += 1
            else:
                self.cached_observe_count += 1
            self.last_observe_mono = now
            payload = {
                "observation": {
                    "state": {"x.vel": 0.0, "theta.vel": 0.0},
                    "state_timestamp_ns": self.last_state_timestamp,
                    "state_cached": state_cached,
                    "camera_status": {"front": {"fresh": True, "timestamp_ns": self.observe_count}},
                    "errors": [],
                    "armed": self.armed,
                    "control_owned": owned,
                },
                "images": {"front": base64.b64encode(self._image).decode()},
            }
        elif endpoint == "command":
            action = json.loads(request.data)["action"]
            self.commands.append(action)
            payload = {
                "accepted": self.armed,
                "command_accepted": self.armed,
                "errors": [] if self.armed else ["lease lost"],
            }
        elif endpoint == "stop":
            self.stop_count += 1
            self.armed = False
            payload = {
                "stop_confirmed": True,
                "stationary_confirmed": True,
                "physical_outcome": "stopped",
                "errors": [],
            }
        else:
            raise AssertionError(endpoint)
        return _Response(payload)


class _TwoSegmentProvider:
    def __init__(self, *, delay_s=0.03):
        self.calls = 0
        self.delay_s = delay_s

    async def reset_session(self, _session_id):
        return None

    async def predict(self, rgb, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return SimpleNamespace(
            output=SimpleNamespace(
                actions=((0.05, 0.0, 0.0),) * 10,
                metadata={"stop": self.calls >= 3},
            )
        )


def _authorized_stub_robot(opener):
    robot = RemoteRobot("http://orin:8080", "secret", scope="base")
    robot.opener = opener
    robot.connect()
    robot.arm()
    return robot


def test_http_stub_keeps_lease_for_two_segments_and_disarms_finally():
    opener = _LeaseAwareOpener()
    robot = _authorized_stub_robot(opener)

    async def no_sleep(_duration):
        return None

    result = asyncio.run(
        run_local_segment_loop(
            _TwoSegmentProvider(),
            robot,
            TeleopRunConfig(
                instruction="go forward",
                camera="front",
                max_steps=3,
                segment=LocalSegmentConfig(duration_s=0.001),
            ),
            clock=lambda: 0.0,
            sleep=no_sleep,
        )
    )
    nonzero = [command for command in opener.commands if abs(command["x.vel"]) > 1e-9]
    assert len(nonzero) == 2
    assert opener.cached_observe_count > 0
    assert opener.arm_count == 1
    assert opener.stop_count >= 2
    assert result["model_stop"] is True
    assert result["final_stop_report"]["stop_confirmed"] is True
    assert (
        result["initial_stationary_feedback"]["state_timestamp_ns"]
        < result["steps"][0]["settled_feedback"]["state_timestamp_ns"]
    )
    assert robot.armed is False
    assert robot.closed is True


def test_http_stub_lease_loss_during_inference_fails_closed():
    # initial hold reads once; camera read is second; post-inference lease
    # check is third and reports that the server revoked ownership.
    opener = _LeaseAwareOpener(lose_lease_on_observe=3)
    robot = _authorized_stub_robot(opener)

    async def no_sleep(_duration):
        return None

    with pytest.raises(RuntimeError, match="lease"):
        asyncio.run(
            run_local_segment_loop(
                _TwoSegmentProvider(),
                robot,
                TeleopRunConfig(
                    instruction="go forward",
                    camera="front",
                    max_steps=2,
                    segment=LocalSegmentConfig(duration_s=0.001),
                ),
                clock=lambda: 0.0,
                sleep=no_sleep,
            )
        )
    assert opener.commands
    assert all(abs(command["x.vel"]) <= 1e-9 for command in opener.commands)
    assert opener.stop_count >= 1
    assert robot.armed is False
