from __future__ import annotations

import contextlib
import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from embodirun.robots import RobotAction, RobotPreparationRefused, robot_definition
from embodirun.robots.lerobot.xlerobot import (
    XLEROBOT_ACTION_SPACE,
    XLeRobotAdapter,
    XLeRobotAdapterError,
)
from embodirun.robots.lerobot.xlerobot.config import XLeRobotConfig
from embodirun.services.control.contracts import ControlServiceConfig
from embodirun.services.control.devices import (
    DeviceCloseError,
    DeviceManager,
    DeviceResource,
    DeviceUncertainError,
    ResourceIdentity,
)
from embodirun.services.control.server import ControlHttpServer, ControlService

_ARM_KEYS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class _OwnerState:
    """Faithful loopback model of the legacy external-owner HTTP contract."""

    def __init__(self, *, scope: str = "arms", token: str = "t") -> None:
        self.scope = scope
        self.token = token
        self.owner: str | None = None
        self.calls: list[tuple[str, str, str | None]] = []
        self.arm_owners: list[str] = []
        self.commands: list[dict[str, Any]] = []
        self.release_owners: list[str | None] = []
        self.drop_after: set[str] = set()
        self.arm_response: dict[str, Any] | None = None
        self.arm_attempts: list[str] = []
        self.next_command_response: dict[str, Any] | None = None
        self.command_rejected = threading.Event()
        self.block_release = False
        self.release_started = threading.Event()
        self.release_continue = threading.Event()
        self.release_finished = threading.Event()
        self.release_stop_confirmed = True
        self.release_status = 200
        self.omit_source_timestamp = False
        self.omit_state_timestamp = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self.owner is not None

    def force_owner_loss(self) -> None:
        with self._lock:
            self.owner = None

    def _check(self, handler: BaseHTTPRequestHandler) -> str:
        if handler.headers.get("Authorization") != f"Bearer {self.token}":
            raise _OwnerHTTPError(401, "invalid token")
        if handler.headers.get("X-Teleop-Scope") != self.scope:
            raise _OwnerHTTPError(403, "scope mismatch")
        owner = handler.headers.get("X-Teleop-Owner")
        if not owner:
            raise _OwnerHTTPError(400, "owner is required")
        return owner

    def status(self) -> dict[str, Any]:
        return {
            "metadata": {
                "control_scopes": [self.scope],
                "joint_unit": "degrees",
                "gripper_unit": "range_0_100",
                "source": "legacy-xlerobot-owner",
            }
        }

    def observation(self, request_owner: str) -> dict[str, Any]:
        with self._lock:
            owner = self.owner
            armed = owner is not None
        values = (
            {"x.vel": 0.0, "theta.vel": 0.0}
            if self.scope == "base"
            else {f"{side}_arm_{joint}.pos": 0.0 for side in ("left", "right") for joint in _ARM_KEYS}
        )
        now = time.time_ns()
        observation: dict[str, Any] = {
            "state": values,
            "armed": armed,
            "control_owned": armed and owner == request_owner,
            "raw": {"legacy": True},
            "raw_fields": {"owner_present": armed},
            "wheel_present_blocks": [],
            "errors": [],
            "metadata": {"scope": self.scope},
        }
        if not self.omit_source_timestamp:
            observation["source_timestamp_ns"] = now
        if not self.omit_state_timestamp:
            observation["state_timestamp_ns"] = now
        return {
            "observation": observation,
            "images": {},
        }


class _OwnerHTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class _OwnerHandler(BaseHTTPRequestHandler):
    server: _OwnerServer

    def log_message(self, *_args: object) -> None:
        pass

    @property
    def state(self) -> _OwnerState:
        return self.server.owner_state

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drop_response(self, endpoint: str) -> bool:
        with self.state._lock:
            drop = endpoint in self.state.drop_after
            if drop:
                self.state.drop_after.remove(endpoint)
        if drop:
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return True
        return False

    def _read_body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(size) or b"{}")
        if not isinstance(value, dict):
            raise _OwnerHTTPError(400, "body must be an object")
        return value

    def _dispatch_error(self, error: _OwnerHTTPError) -> None:
        self._send_json(error.status, {"error": str(error)})

    def do_GET(self) -> None:
        endpoint = self.path.removeprefix("/robot/")
        try:
            owner = self.state._check(self)
            self.state.calls.append(("GET", endpoint, owner))
            if endpoint == "status":
                value = self.state.status()
            elif endpoint == "observe":
                value = self.state.observation(owner)
            else:
                raise _OwnerHTTPError(404, "unknown endpoint")
            self._send_json(200, value)
        except _OwnerHTTPError as error:
            self._dispatch_error(error)

    def do_POST(self) -> None:
        endpoint = self.path.removeprefix("/robot/")
        try:
            owner = self.state._check(self)
            body = self._read_body()
            self.state.calls.append(("POST", endpoint, owner))
            if endpoint == "arm":
                if body:
                    raise _OwnerHTTPError(400, "arm body must be empty")
                configured = self.state.arm_response
                self.state.arm_response = None
                self.state.arm_attempts.append(owner)
                if configured is not None:
                    if configured.get("writes"):
                        with self.state._lock:
                            self.state.owner = owner
                    value = dict(configured)
                    dropped = self._drop_response(endpoint)
                    if not dropped:
                        self._send_json(200, value)
                    return
                with self.state._lock:
                    if self.state.owner is not None:
                        raise _OwnerHTTPError(409, "owner already armed")
                    self.state.owner = owner
                    self.state.arm_owners.append(owner)
                value = {"armed": True, "control_owned": True}
                dropped = self._drop_response(endpoint)
                if not dropped:
                    self._send_json(200, value)
                return
            if endpoint == "command":
                with self.state._lock:
                    if self.state.owner != owner:
                        raise _OwnerHTTPError(409, "command owner mismatch")
                action = body.get("action")
                if not isinstance(action, dict) or not action:
                    raise _OwnerHTTPError(400, "action is required")
                self.state.commands.append(dict(action))
                value = self.state.next_command_response or {
                    "command_accepted": True,
                    "applied_action": dict(action),
                    "raw_action": {"legacy": dict(action)},
                    "raw_fields": {"command_count": len(self.state.commands)},
                    "physical_outcome": "unknown",
                    "errors": [],
                }
                self.state.next_command_response = None
                if value.get("command_accepted") is False:
                    self.state.command_rejected.set()
                dropped = self._drop_response(endpoint)
                if not dropped:
                    self._send_json(200, value)
                return
            if endpoint == "release":
                if body:
                    raise _OwnerHTTPError(400, "release body must be empty")
                with self.state._lock:
                    self.state.release_owners.append(owner)
                    if self.state.owner != owner:
                        raise _OwnerHTTPError(409, "release owner mismatch")
                if self.state.block_release:
                    self.state.release_started.set()
                    self.state.release_continue.wait(2.0)
                with self.state._lock:
                    self.state.owner = None
                value = {
                    "released": True,
                    "control_owned": True,
                    "stop_confirmed": self.state.release_stop_confirmed,
                    "physical_outcome": "unknown",
                }
                dropped = self._drop_response(endpoint)
                try:
                    if not dropped:
                        self._send_json(self.state.release_status, value)
                finally:
                    self.state.release_finished.set()
                return
            raise _OwnerHTTPError(404, "unknown endpoint")
        except _OwnerHTTPError as error:
            self._dispatch_error(error)


class _OwnerServer(ThreadingHTTPServer):
    def __init__(self, state: _OwnerState):
        self.owner_state = state
        super().__init__(("127.0.0.1", 0), _OwnerHandler)


@pytest.fixture()
def owner():
    state = _OwnerState()
    server = _OwnerServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _adapter(url: str, *, scope: str = "arms", timeout_s: float = 1.0) -> XLeRobotAdapter:
    return XLeRobotAdapter(
        XLeRobotConfig.from_mapping(
            "xlerobot-test",
            {"url": url, "token": "t", "scope": scope, "timeout_s": timeout_s},
        )
    )


def _arm_action(value: float = 2.0, *, metadata: dict[str, Any] | None = None) -> RobotAction:
    return RobotAction(
        0.0,
        {"left_arm_shoulder_pan.pos": value},
        metadata
        or {
            "action_space": XLEROBOT_ACTION_SPACE,
            "units": {"left_arm_shoulder_pan.pos": "degrees"},
        },
    )


def _base_action(*, complete: bool = True) -> RobotAction:
    values: dict[str, float] = {"x.vel": 0.2}
    if complete:
        values["theta.vel"] = 3.0
    return RobotAction(
        0.0,
        values,
        {
            "action_space": XLEROBOT_ACTION_SPACE,
            "units": {
                "x.vel": "metres-per-sec",
                **({"theta.vel": "angular-degrees-per-sec"} if complete else {}),
            },
        },
    )


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _http_request(
    port: int,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    encoded = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if encoded is not None else {}
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read().decode() or "{}")
    connection.close()
    return response.status, payload


@pytest.fixture()
def control_http(owner, tmp_path):
    state, owner_url = owner

    def make(*, scope: str = "arms", external_owner: bool = True):
        state.scope = scope
        identity = "xlerobot-test-owner"
        resource = {
            "identity": f"local:robot:{identity}",
            "node": "local",
            "kind": "robot",
            "value": identity,
            "external_owner": external_owner,
        }
        config = ControlServiceConfig.device_only(
            runtime_id="xlerobot-http",
            bind="127.0.0.1",
            port=_port(),
            robot_id="xlerobot-http",
            robot_kind="lerobot.xlerobot",
            robot_options={"url": owner_url, "token": "t", "scope": scope},
            node_id="local",
            device_resources=(resource,),
        )
        manager = DeviceManager(
            "local",
            owner_id="xlerobot-http-test",
            lock_dir=tmp_path / f"locks-{scope}-{external_owner}",
            state_path=tmp_path / f"state-{scope}-{external_owner}.json",
        )
        service = ControlService(config, device_manager=manager)
        server = ControlHttpServer(service, state_dir=tmp_path / "jobs")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return state, service, server, thread

    yield make


def _close_control(service: ControlService, server: ControlHttpServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
    with contextlib.suppress(Exception):
        # Tests which intentionally leave a stop unknown still need to stop
        # their HTTP listener; the assertion owns the lifecycle error.
        service.close()


def test_definition_and_passive_prepare_release_new_explicit_task(owner):
    state, url = owner
    definition = robot_definition("lerobot.xlerobot")
    adapter = _adapter(url)
    adapter.connect(prepare=False)
    assert adapter.prepared is False
    observation = adapter.observe()
    assert observation.timestamp_s > 0
    assert observation.metadata["source"] == "xlerobot.external_owner"
    adapter.prepare()
    adapter.execute(_arm_action())
    adapter.stop()
    assert adapter.prepared is False
    adapter.prepare()
    adapter.stop()
    assert len(state.arm_owners) == 2
    assert state.arm_owners[0] != state.arm_owners[1]
    assert definition.adapter_type is XLeRobotAdapter


def test_fake_owner_reports_armed_and_control_owned_without_cross_owner_access(owner):
    state, url = owner
    first, second = _adapter(url), _adapter(url)
    first.connect()
    with pytest.raises(XLeRobotAdapterError, match="already armed"):
        second.connect()
    stale = second.observe()
    assert stale.metadata["source"] == "xlerobot.external_owner"
    assert state.armed is True
    first.stop()


def test_concurrent_arm_is_rejected_and_does_not_replace_lease(owner):
    state, url = owner
    first, second = _adapter(url), _adapter(url)
    first.connect(prepare=False)
    second.connect(prepare=False)
    first.prepare()
    with pytest.raises(XLeRobotAdapterError, match="owner already armed"):
        second.prepare()
    assert state.owner == first.owner
    assert state.arm_owners == [first.owner]
    first.stop()


def test_requested_scope_must_be_advertised_by_owner(owner):
    state, url = owner
    adapter = _adapter(url, scope="base")
    with pytest.raises(XLeRobotAdapterError, match="scope mismatch"):
        adapter.connect(prepare=False)
    assert state.calls == []


def test_stale_release_cannot_stop_new_owner(owner):
    state, url = owner
    old, new = _adapter(url), _adapter(url)
    old.connect()
    state.force_owner_loss()
    new.connect(prepare=False)
    new.prepare()
    with pytest.raises(XLeRobotAdapterError, match="release owner mismatch"):
        old.stop()
    assert state.owner == new.owner
    new.execute(_arm_action(4.0))
    new.stop()


def test_ownership_loss_never_auto_arms_or_reuses_prepared_lease(owner):
    state, url = owner
    adapter = _adapter(url)
    adapter.connect()
    assert adapter.prepared
    state.force_owner_loss()
    adapter.observe()
    assert adapter.prepared is False
    assert len(state.arm_owners) == 1
    adapter.observe()
    assert len(state.arm_owners) == 1
    with pytest.raises(XLeRobotAdapterError, match="not prepared"):
        adapter.execute(_arm_action())
    with pytest.raises(XLeRobotAdapterError, match="stop is unconfirmed"):
        adapter.prepare()
    assert len(state.arm_owners) == 1


def test_observation_without_foreign_timestamps_does_not_invent_local_time(owner):
    state, url = owner
    state.omit_source_timestamp = True
    state.omit_state_timestamp = True
    adapter = _adapter(url)
    adapter.connect(prepare=False)
    with pytest.raises(XLeRobotAdapterError, match="state_timestamp_ns is invalid"):
        adapter.observe()


@pytest.mark.parametrize("endpoint", ["arm", "command", "release"])
def test_transport_drop_after_remote_acceptance_is_not_retried(owner, endpoint):
    state, url = owner
    state.drop_after.add(endpoint)
    adapter = _adapter(url, timeout_s=0.4)
    adapter.connect(prepare=False)
    if endpoint == "arm":
        with pytest.raises(XLeRobotAdapterError):
            adapter.prepare()
        assert len(state.arm_owners) == 1
        assert adapter.stop_unconfirmed is True
        assert len([item for item in state.calls if item[1] == "arm"]) == 1
        return
    adapter.prepare()
    if endpoint == "command":
        with pytest.raises(XLeRobotAdapterError):
            adapter.execute(_arm_action())
        assert len(state.commands) == 1
        assert len([item for item in state.calls if item[1] == "command"]) == 1
        assert adapter.stop_unconfirmed is True
    else:
        with pytest.raises(XLeRobotAdapterError):
            adapter.stop()
        assert adapter.stop_unconfirmed is True
        assert len(state.release_owners) == 1


def test_stop_unknown_stays_unknown_after_repeated_release_attempts(owner):
    state, url = owner
    state.release_stop_confirmed = False
    adapter = _adapter(url)
    adapter.connect()
    with pytest.raises(XLeRobotAdapterError, match="did not confirm"):
        adapter.stop()
    assert adapter.stop_unconfirmed is True
    with pytest.raises(XLeRobotAdapterError):
        adapter.stop()
    assert adapter.stop_unconfirmed is True
    assert state.owner is None


def test_prepare_refusal_without_writes_can_be_retried_after_remote_clear(owner):
    state, url = owner
    state.arm_response = {
        "armed": False,
        "status": "refused",
        "writes": [],
        "errors": ["deadman is not active"],
        "physical_outcome": "unknown",
    }
    adapter = _adapter(url)
    adapter.connect(prepare=False)
    with pytest.raises(RobotPreparationRefused, match="before any write"):
        adapter.prepare()
    assert adapter.stop_unconfirmed is False
    assert state.owner is None
    adapter.prepare()
    assert len(state.arm_attempts) == 2
    adapter.stop()


def test_partial_prepare_failure_remains_uncertain_even_when_owner_is_cleared(owner):
    state, url = owner
    state.arm_response = {
        "armed": False,
        "status": "error",
        "writes": [{"actuator": "left_arm", "enabled": True}],
        "errors": ["right bus unavailable"],
        "physical_outcome": "unknown",
    }
    adapter = _adapter(url)
    adapter.connect(prepare=False)
    with pytest.raises(XLeRobotAdapterError, match="did not confirm"):
        adapter.prepare()
    assert adapter.stop_unconfirmed is True
    state.force_owner_loss()
    with pytest.raises(XLeRobotAdapterError, match="stop is unconfirmed"):
        adapter.prepare()
    assert len(state.arm_attempts) == 1


def test_unsupported_unit_is_rejected_before_http_command(owner):
    state, url = owner
    adapter = _adapter(url)
    adapter.connect()
    with pytest.raises(ValueError, match="unsupported unit"):
        adapter.execute(
            _arm_action(
                metadata={
                    "action_space": XLEROBOT_ACTION_SPACE,
                    "units": {"left_arm_shoulder_pan.pos": "radians"},
                }
            )
        )
    assert state.commands == []
    adapter.stop()


def test_gripper_accepts_native_legacy_range_unit(owner):
    state, url = owner
    adapter = _adapter(url)
    adapter.connect()
    adapter.execute(
        RobotAction(
            0.0,
            {"left_arm_gripper.pos": 35.0},
            {
                "action_space": XLEROBOT_ACTION_SPACE,
                "units": {"left_arm_gripper.pos": "range_0_100"},
            },
        )
    )
    assert state.commands == [{"left_arm_gripper.pos": 35.0}]
    adapter.stop()


def test_http_direct_first_returns_applied_raw_receipt_and_confirmed_hold(control_http):
    state, service, server, thread = control_http()
    try:
        status, payload = _http_request(
            server.server_port,
            "POST",
            "/v1/execute",
            {
                "request_id": "direct-first",
                "action": {
                    "timestamp_s": 1.0,
                    "values": {"left_arm_shoulder_pan.pos": 2.0},
                    "metadata": {
                        "action_space": XLEROBOT_ACTION_SPACE,
                        "units": {"left_arm_shoulder_pan.pos": "degrees"},
                    },
                },
                "wait": True,
            },
        )
        assert status == 200, payload
        assert payload["status"] == "completed"
        receipt = payload["result"]["outcomes"][0]["driver_receipt"]
        assert receipt["command_accepted"] is True
        assert receipt["applied_action"]["left_arm_shoulder_pan.pos"] == 2.0
        assert receipt["raw_action"]["legacy"]
        assert state.arm_owners and state.owner is None
        assert len(state.commands) == 1
    finally:
        _close_control(service, server, thread)


def test_http_passive_observe_first_then_direct_prepares_once(control_http):
    state, service, server, thread = control_http()
    try:
        status, observed = _http_request(server.server_port, "GET", "/v1/observe?include_robot=true")
        assert status == 200, observed
        robot = observed["robot"]
        assert robot["values"]["left_arm_shoulder_pan.pos"] == 0.0
        assert robot["metadata"]["raw_fields"]["owner_present"] is False
        assert robot["metadata"]["remote_errors"] == []
        assert state.arm_owners == []
        status, result = _http_request(
            server.server_port,
            "POST",
            "/v1/execute",
            {
                "request_id": "after-observe",
                "action": {
                    "timestamp_s": 1.0,
                    "values": {"left_arm_shoulder_pan.pos": 3.0},
                    "metadata": {
                        "action_space": XLEROBOT_ACTION_SPACE,
                        "units": {"left_arm_shoulder_pan.pos": "degrees"},
                    },
                },
                "wait": True,
            },
        )
        assert status == 200, result
        assert len(state.arm_owners) == 1
    finally:
        _close_control(service, server, thread)


def test_http_valid_base_action_reaches_owner(control_http):
    state, service, server, thread = control_http(scope="base")
    try:
        action = _base_action()
        status, payload = _http_request(
            server.server_port,
            "POST",
            "/v1/execute",
            {
                "request_id": "valid-base",
                "action": {
                    "timestamp_s": action.timestamp_s,
                    "values": action.values,
                    "metadata": action.metadata,
                },
                "wait": True,
            },
        )
        assert status == 200, payload
        assert state.commands == [{"x.vel": 0.2, "theta.vel": 3.0}]
    finally:
        _close_control(service, server, thread)


def test_http_two_successive_segments_use_fresh_owner_per_prepare(control_http):
    state, service, server, thread = control_http()
    try:
        for index in (1, 2):
            status, payload = _http_request(
                server.server_port,
                "POST",
                "/v1/execute",
                {
                    "request_id": f"segment-{index}",
                    "action": {
                        "timestamp_s": float(index),
                        "values": {"left_arm_shoulder_pan.pos": float(index)},
                        "metadata": {
                            "action_space": XLEROBOT_ACTION_SPACE,
                            "units": {"left_arm_shoulder_pan.pos": "degrees"},
                        },
                    },
                    "wait": True,
                },
            )
            assert status == 200, payload
        assert len(state.arm_owners) == 2
        assert state.arm_owners[0] != state.arm_owners[1]
        assert len(state.commands) == 2
        assert len(state.release_owners) == 2
    finally:
        _close_control(service, server, thread)


@pytest.mark.parametrize(
    "action",
    [
        {
            "timestamp_s": 1.0,
            "values": {"left_arm_shoulder_pan.pos": 2.0},
            "metadata": {"units": {"left_arm_shoulder_pan.pos": "degrees"}},
        },
        {
            "timestamp_s": 1.0,
            "values": {"left_arm_shoulder_pan.pos": 2.0},
            "metadata": {
                "action_space": "wrong.action.space",
                "units": {"left_arm_shoulder_pan.pos": "degrees"},
            },
        },
        {
            "timestamp_s": 1.0,
            "values": {"left_arm_shoulder_pan.pos": 2.0},
            "metadata": {"action_space": XLEROBOT_ACTION_SPACE, "units": {}},
        },
    ],
)
def test_http_wrong_or_missing_action_space_is_rejected_without_command(control_http, action):
    state, service, server, thread = control_http()
    try:
        status, payload = _http_request(
            server.server_port,
            "POST",
            "/v1/execute",
            {"request_id": "bad-action-space", "action": action, "wait": True},
        )
        assert status == 502, payload
        assert state.commands == []
    finally:
        _close_control(service, server, thread)


def test_http_incomplete_base_action_is_rejected_without_command(control_http):
    state, service, server, thread = control_http(scope="base")
    try:
        action = _base_action(complete=False)
        status, payload = _http_request(
            server.server_port,
            "POST",
            "/v1/execute",
            {
                "request_id": "incomplete-base",
                "action": {
                    "timestamp_s": action.timestamp_s,
                    "values": action.values,
                    "metadata": action.metadata,
                },
                "wait": True,
            },
        )
        assert status == 502, payload
        assert state.commands == []
    finally:
        _close_control(service, server, thread)


def test_http_200_command_rejected_is_failed_once_and_released(control_http):
    state, service, server, thread = control_http()
    state.block_release = True
    state.next_command_response = {
        "command_accepted": False,
        "applied_action": {},
        "raw_action": {},
        "errors": ["servo rejected command"],
    }
    request_done = threading.Event()
    response: list[tuple[int, dict[str, Any]]] = []
    errors: list[BaseException] = []

    def submit() -> None:
        try:
            response.append(
                _http_request(
                    server.server_port,
                    "POST",
                    "/v1/execute",
                    {
                        "request_id": "rejected-command",
                        "action": {
                            "timestamp_s": 1.0,
                            "values": {"left_arm_shoulder_pan.pos": 2.0},
                            "metadata": {
                                "action_space": XLEROBOT_ACTION_SPACE,
                                "units": {"left_arm_shoulder_pan.pos": "degrees"},
                            },
                        },
                        "wait": True,
                    },
                )
            )
        except BaseException as error:  # pragma: no cover - diagnostic assertion below
            errors.append(error)
        finally:
            request_done.set()

    request_thread = threading.Thread(target=submit)
    request_thread.start()
    try:
        assert state.command_rejected.wait(1.0)
        assert state.release_started.wait(1.0)
        assert not request_done.is_set()

        state.release_continue.set()
        assert state.release_finished.wait(1.0)
        assert request_done.wait(1.0)
        request_thread.join(1.0)
        assert errors == []
        assert len(response) == 1
        status, payload = response[0]
        assert status == 502, payload
        assert len(state.commands) == 1
        assert len([item for item in state.calls if item[1] == "command"]) == 1
        assert len(state.release_owners) == 1
        assert state.owner is None
    finally:
        state.release_continue.set()
        request_thread.join(1.0)
        _close_control(service, server, thread)


def test_xlerobot_without_external_owner_resource_is_rejected(control_http):
    state, service, server, thread = control_http(external_owner=False)
    try:
        status, payload = _http_request(
            server.server_port,
            "POST",
            "/v1/execute",
            {
                "request_id": "false-owner",
                "action": {
                    "timestamp_s": 1.0,
                    "values": {"left_arm_shoulder_pan.pos": 2.0},
                    "metadata": {
                        "action_space": XLEROBOT_ACTION_SPACE,
                        "units": {"left_arm_shoulder_pan.pos": "degrees"},
                    },
                },
                "wait": True,
            },
        )
        assert status == 502, payload
        assert state.calls == []
    finally:
        _close_control(service, server, thread)


def test_failed_close_remains_uncertain_after_manager_restart(tmp_path: Path):
    state = _OwnerState()
    state.release_stop_confirmed = False
    server = _OwnerServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    identity = ResourceIdentity("local", "robot", "failed-close")
    try:

        def make_adapter() -> XLeRobotAdapter:
            adapter = _adapter(url)
            adapter.connect(prepare=False)
            return adapter

        manager = DeviceManager(
            "local",
            owner_id="failed-close-test",
            lock_dir=tmp_path / "locks",
            state_path=tmp_path / "state.json",
        )
        lease = manager.acquire(DeviceResource(identity, make_adapter, external_owner=True), role="control")
        lease.value.prepare()
        with pytest.raises(DeviceCloseError, match="did not confirm a stop"):
            lease.close()
        assert len(state.arm_owners) == 1

        restarted = DeviceManager(
            "local",
            owner_id="restarted-manager",
            lock_dir=tmp_path / "locks-2",
            state_path=tmp_path / "state.json",
        )
        with pytest.raises(DeviceUncertainError, match="unresolved lifecycle state"):
            restarted.acquire(
                DeviceResource(identity, make_adapter, external_owner=True),
                role="control",
                prepare=True,
            )
        assert len(state.arm_owners) == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
