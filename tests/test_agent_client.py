from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from embodirun.client import ControlClient, ControlHTTPError
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.application import ControlApplication
from embodirun.services.control.arbitration import RobotControlArbiter
from embodirun.services.control.contracts import ControlServiceConfig
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.http_api import ControlHTTPAPI
from embodirun.services.control.io import IOResult, IOStatus
from embodirun.services.control.observation_store import ObservationStore
from embodirun.services.control.observation_values import ObservationSnapshot
from embodirun.services.control.proposals import ProposalError, generate_proposal
from embodirun.services.control.server import ControlHttpServer, ControlService
from embodirun.services.inference import (
    ImagePayload,
    PolicyAction,
    PolicyObservation,
    PolicyResult,
    Session,
)


class _Port:
    robot_id = "fake"

    def __init__(self) -> None:
        self.actions: list[RobotAction] = []

    def execute(self, action: RobotAction, cancel_event: threading.Event) -> IOResult:
        if cancel_event.is_set():
            return IOResult("execute", IOStatus.CANCELLED, requested=action)
        self.actions.append(action)
        return IOResult(
            "execute",
            IOStatus.COMPLETED,
            requested=action,
            driver_returned=True,
            driver_value={"accepted": len(self.actions)},
        )

    def hold(self) -> IOResult:
        return IOResult("stop", IOStatus.COMPLETED, stop_confirmed=True)

    emergency_stop = hold

    def close(self) -> None:
        return None


class _Service:
    def __init__(self, port: _Port | None = None) -> None:
        self.count = 0
        self.port = port

    def describe(self) -> dict[str, Any]:
        return {"status": "ok", "robot_id": "fake", "capabilities": {"execute": True}}

    def observe(self, *, runtime_id: str | None, include_robot: bool) -> dict[str, Any]:
        self.count += 1
        return {
            "status": "ok",
            "observation_id": f"obs-{self.count}",
            "runtime_id": runtime_id,
            "robot": (
                {"j0": float(len(self.port.actions) if self.port is not None else 0.0)} if include_robot else None
            ),
        }


class _JSONServer(ThreadingHTTPServer):
    daemon_threads = True


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _Handler(BaseHTTPRequestHandler):
    api: ControlHTTPAPI

    def log_message(self, *_args: object) -> None:
        return None

    def _call(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else None
        result = self.server.api.dispatch(method, self.path, body=body, headers=dict(self.headers.items()))
        encoded = json.dumps(result.payload).encode("utf-8")
        self.send_response(result.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        self._call("GET")

    def do_POST(self) -> None:
        self._call("POST")


def _client_server() -> tuple[ControlClient, _JSONServer, _Port]:
    port = _Port()
    service = _Service(port)
    arbiter = RobotControlArbiter(port)
    application = ControlApplication(service, arbiter_provider=lambda: arbiter)
    server = _JSONServer(("127.0.0.1", 0), _Handler)
    server.api = ControlHTTPAPI(application)
    server.application = application
    server.arbiter = arbiter
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.thread = thread
    return (
        ControlClient(
            f"http://127.0.0.1:{server.server_port}",
            caller_id="agent",
            session_id="session",
        ),
        server,
        port,
    )


def _close(server: _JSONServer) -> None:
    server.shutdown()
    server.server_close()
    server.application.close()
    server.arbiter.close(hold=False)


def test_client_uses_real_http_observe_execute_and_job_lookup() -> None:
    client, server, port = _client_server()
    try:
        assert client.describe()["robot_id"] == "fake"
        observation = client.observe()
        assert observation.robot == {"j0": 0.0}
        result = client.execute(
            [{"timestamp_s": 0.0, "values": {"j0": 1.0}}],
            request_id="agent-http-1",
            observation_id=observation.observation_id,
            wait=True,
        )
        assert result["status"] == "completed"
        assert client.inspect("agent-http-1")["request_id"] == "agent-http-1"
        assert port.actions[0].values == {"j0": 1.0}
        after = client.observe()
        assert after.observation_id != observation.observation_id
        assert after.robot == {"j0": 1.0}
    finally:
        _close(server)


def test_client_rejects_credentials_and_preserves_http_error() -> None:
    with pytest.raises(ValueError, match="without credentials"):
        ControlClient(
            "https://user:pass@example.invalid",
            caller_id="agent",
            session_id="session",
        )

    client, server, _ = _client_server()
    try:
        with pytest.raises(ControlHTTPError) as caught:
            client.inspect("missing")
        assert caught.value.status == 404
        assert caught.value.unknown is False
    finally:
        _close(server)


def test_real_http_propose_consumes_exact_snapshot_and_closes_only_session() -> None:
    now_ns = 10_000_000
    store = ObservationStore(service_instance_id="proposal-service")
    observation_id, generation, sequence = store.next_observation_id()
    snapshot = store.publish(
        ObservationSnapshot(
            observation_id=observation_id,
            service_instance_id=store.service_instance_id,
            generation=generation,
            sequence=sequence,
            state={"state": [0.0] * 6},
            cameras=(
                CameraFrame(
                    "front",
                    "image/png",
                    b"png",
                    captured_timestamp_ns=now_ns,
                    received_timestamp_ns=now_ns,
                    clock_domain="host_monotonic_ns",
                ),
            ),
            metadata={
                "state_metadata": {"timestamp_s": 1.0},
                "clock_domain": "host_monotonic_ns",
            },
            captured_timestamp_ns=now_ns,
            received_timestamp_ns=now_ns,
            published_timestamp_ns=now_ns,
            source_timestamps_ns={"front": now_ns},
            source_received_timestamps_ns={"front": now_ns},
            clock_domains={"front": "host_monotonic_ns", "state": "host_monotonic_ns"},
            skew_ns=0,
        )
    )

    class Mapper:
        policy_action_space = "fake.action.v1"

        def __init__(self):
            self.seen = None

        def map_observation(self, observation, **kwargs):
            self.seen = (observation, tuple(kwargs["frames"]))
            return PolicyObservation(
                session_id=kwargs["session_id"],
                request_id=kwargs["request_id"],
                step_id=kwargs["step_id"],
                instruction=kwargs["instruction"],
                state=observation.values,
                images=(ImagePayload("front", "image/png", b"png"),),
                metadata={"seen_observation": observation.metadata["observation_id"]},
            )

        def map_result(self, result):
            return (
                RobotAction(
                    timestamp_s=1.0,
                    values={"joint": 1.0},
                    metadata={"proposal": True},
                ),
            )

    mapper = Mapper()

    class FakeInference:
        def __init__(self):
            self.opened = self.closed = 0
            self.released = 0
            self.seen = None

        def open_session(self, **kwargs):
            self.opened += 1
            return Session("proposal-session", 0)

        def step(self, observation):
            self.seen = observation
            return PolicyResult(
                request_id=observation.request_id,
                session_id=observation.session_id,
                step_id=observation.step_id,
                session_revision=1,
                action_space="fake.action.v1",
                actions=(PolicyAction("action_chunk", {"data": [[1.0]]}),),
            )

        def close(self, session_id):
            assert session_id == "proposal-session"
            self.closed += 1

    inference = FakeInference()
    binding = SimpleNamespace(mapper_factory=lambda: mapper)
    profile = SimpleNamespace(runtime_id="runtime", robot_id="fake", inference_enabled=True)

    class Service:
        def proposal_context(self, observation_id, *, runtime_id=None):
            assert observation_id == snapshot.observation_id
            return (
                profile,
                binding,
                RobotObservation(
                    timestamp_s=1.0,
                    values={"state": [0.0] * 6},
                    metadata={"observation_id": observation_id},
                ),
                snapshot.cameras,
            )

        def _inference_client(self, _config, _timeout):
            return inference

        def _release_inference_client(self, client, _config):
            assert client is inference
            inference.released += 1

    app = ControlApplication(
        Service(),
        observation_store=store,
        clock_ns=lambda: now_ns,
        max_observation_age_ns=1,
    )
    server = _JSONServer(("127.0.0.1", 0), _Handler)
    server.api = ControlHTTPAPI(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(
        f"http://127.0.0.1:{server.server_port}",
        caller_id="agent",
        session_id="session",
    )
    try:
        result = client.propose(
            request_id="proposal-request",
            observation_id=snapshot.observation_id,
            instruction="move",
        )
        assert result["status"] == "proposed"
        assert result["observation_id"] == snapshot.observation_id
        assert result["actions"][0]["values"] == {"joint": 1.0}
        assert inference.opened == inference.closed == 1
        assert inference.released == 1
        assert inference.seen.metadata["observation_id"] == snapshot.observation_id
        assert inference.seen.images[0].data == b"png"
        assert mapper.seen[1][0].data == b"png"
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_proposal_rejects_mismatched_non_vvla_result_identity() -> None:
    class Mapper:
        policy_action_space = "sglang.action.v1"

        def map_observation(self, observation, **kwargs):
            return PolicyObservation(
                session_id=kwargs["session_id"],
                request_id=kwargs["request_id"],
                step_id=kwargs["step_id"],
                instruction=kwargs["instruction"],
                state=observation.values,
                images=(ImagePayload("front", "image/png", b"x"),),
            )

        def map_result(self, result):
            return (RobotAction(0.0, {"joint": 0.0}),)

    class Inference:
        def open_session(self, **_kwargs):
            return Session("sglang-session", 0)

        def step(self, observation):
            return PolicyResult(
                request_id="different-request",
                session_id=observation.session_id,
                step_id=observation.step_id,
                session_revision=1,
                action_space="sglang.action.v1",
                actions=(PolicyAction("chunk", {"data": [0.0]}),),
            )

        def close(self, _session_id):
            return None

    profile = SimpleNamespace(runtime_id="runtime", robot_id="fake", inference_enabled=True)
    binding = SimpleNamespace(mapper_factory=Mapper)
    inference = Inference()

    class Service:
        def proposal_context(self, _observation_id, *, runtime_id=None):
            return (
                profile,
                binding,
                RobotObservation(1.0, {"j0": 0.0}, {}),
                (),
            )

        def _inference_client(self, _config, _timeout):
            return inference

    with pytest.raises(ProposalError, match="does not match"):
        generate_proposal(
            Service(),
            observation_id="obs-identity",
            instruction="move",
            runtime_id=None,
            request_id="request-identity",
            timeout_s=1.0,
        )


class _FakeSglangHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, *_args: object) -> None:
        return None

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/actions/generations":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(payload)
        self._json(
            200,
            {
                "id": payload["request_id"],
                "model": "fake-sglang",
                "data": [{"action": {"values": [[0.25] * 6 for _ in range(50)]}}],
            },
        )


class _FakeSglangServer(ThreadingHTTPServer):
    daemon_threads = True


class _TwoFrameSource:
    def __init__(self, names: Sequence[str]) -> None:
        self._names = tuple(names)

    def capture(self) -> tuple[CameraFrame, ...]:
        now = time.monotonic_ns()
        return tuple(
            CameraFrame(
                name,
                "image/png",
                b"front-frame" if name.endswith("front") else b"wrist-frame",
                captured_timestamp_ns=now,
                received_timestamp_ns=now,
                clock_domain="host_monotonic_ns",
            )
            for name in self._names
        )

    def close(self) -> None:
        return None


def test_real_http_control_and_non_vvla_sglang_proposal_execute_reobserve(
    tmp_path,
) -> None:
    """Exercise config-built SGLang client through the public Control HTTP API."""

    _FakeSglangHandler.requests = []
    model_server = _FakeSglangServer(("127.0.0.1", 0), _FakeSglangHandler)
    model_thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    model_thread.start()
    inputs = (
        SensorInput("fake-front", "observation.images.front", "fake", {}),
        SensorInput("fake-wrist", "observation.images.wrist", "fake", {}),
    )
    feature_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    config = ControlServiceConfig(
        runtime_id="sim-runtime",
        binding_kind="simulated.policy_vector.pi05",
        bind="127.0.0.1",
        port=_free_port(),
        inference_transport="http",
        inference_endpoint=f"http://127.0.0.1:{model_server.server_port}",
        inference_options={
            "action_feature_names": feature_names,
            "output_action_dim": 6,
            "state_fields": ["state_native"],
        },
        inference_backend="sglang",
        robot_id="sim-robot",
        robot_kind="simulated.policy_vector",
        robot_options={},
        inputs=inputs,
        runtime_options={},
    )
    service = ControlService(
        config,
        camera_factory=lambda items: _TwoFrameSource(tuple(item.name for item in items)),
        device_manager=DeviceManager("local", lock_dir=tmp_path / "locks"),
    )
    api_server = ControlHttpServer(service)
    api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
    api_thread.start()
    client = ControlClient(
        f"http://127.0.0.1:{api_server.server_port}",
        caller_id="agent",
        session_id="session",
    )
    try:
        described = client.describe()
        assert described["capabilities"]["propose"] is True
        binding = described["binding"]
        assert binding["kind"] == "simulated.policy_vector.pi05"
        assert binding["maximum_chunk_steps"] > 0
        assert binding["action_feature_names"]
        before = client.observe(include_robot=True)
        assert before.robot["state_native"] == [0.0] * 6
        proposal = client.propose(
            request_id="sglang-proposal",
            observation_id=before.observation_id,
            instruction="move",
            max_age_ns=2_000_000_000,
        )
        assert proposal["status"] == "proposed"
        assert len(proposal["actions"]) == 50
        assert len(_FakeSglangHandler.requests) == 1
        model_input = _FakeSglangHandler.requests[0]["input"]["observation"]
        assert model_input["state"] == [0.0] * 6
        assert set(model_input["images"]) == {"front", "wrist"}
        # Proposal is compute-only: the simulated owner is unchanged until a
        # separate public execute request submits one bounded action.
        still_before = client.observe(
            observation_id=before.observation_id,
            include_robot=True,
            max_age_ns=2_000_000_000,
        )
        assert still_before.robot["state_native"] == [0.0] * 6
        executed = client.execute(
            proposal["actions"][0],
            request_id="sglang-execute",
            observation_id=before.observation_id,
            wait=True,
            max_age_ns=2_000_000_000,
        )
        assert executed["status"] == "completed"
        assert client.inspect("sglang-execute")["status"] == "completed"
        # The observation producer samples the robot on its own schedule, so a
        # new observation can still carry a sample taken before the action was
        # applied.  Wait for the reflected state, not merely for a new ID.
        after = client.observe(include_robot=True, max_age_ns=2_000_000_000)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if after.observation_id != before.observation_id and after.robot["state_native"] == [0.25] * 6:
                break
            time.sleep(0.05)
            after = client.observe(include_robot=True, max_age_ns=2_000_000_000)
        assert after.observation_id != before.observation_id
        assert after.robot["state_native"] == [0.25] * 6
    finally:
        api_server.shutdown()
        api_server.server_close()
        service.close()
        model_server.shutdown()
        model_server.server_close()
