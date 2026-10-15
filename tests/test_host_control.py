"""Focused tests for the Host application client and JSON commands."""

from __future__ import annotations

import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.services.control.application import ControlApplication
from embodirun.services.control.http_api import ControlHTTPAPI
from embodirun.services.host.cli import build_parser
from embodirun.services.host.cli.command.control import EXIT_UNKNOWN
from embodirun.services.host.config import config_digest
from embodirun.services.host.control import (
    ControlClientError,
    HostControlClient,
)
from embodirun.services.host.executor import JsonHttpResponse, LocalExecutor
from embodirun.services.host.plan import RuntimeSpec
from embodirun.services.host.state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)


class RecordingExecutor:
    def __init__(self, responder=None) -> None:
        self.calls = []
        self.responder = responder or self._default_response
        self.closed = False

    def request_json(self, method, url, payload, *, timeout_s, headers=None):
        self.calls.append((method, url, payload, timeout_s, headers))
        return self.responder(method, url, payload, headers)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _default_response(method, url, payload, headers):
        if url.endswith("/v1/execute"):
            return JsonHttpResponse(
                202,
                {"status": "accepted", "request_id": payload["request_id"]},
            )
        if url.endswith("/cancel") or url.endswith("/stop"):
            return JsonHttpResponse(202, {"status": "pending"})
        return JsonHttpResponse(200, {"status": "ok"})


def client(executor: RecordingExecutor) -> HostControlClient:
    return HostControlClient(
        executor,
        "http://127.0.0.1:8100",
        caller_id="caller-a",
        session_id="session-a",
        token="token-a",
    )


def test_host_control_client_routes_and_identity_headers() -> None:
    executor = RecordingExecutor()
    api = client(executor)

    assert api.describe() == {"status": "ok"}
    assert api.observe(
        runtime_id="fake-device",
        observation_id="obs one",
        include_robot=True,
        max_age_ns=10,
        max_skew_ns=20,
    ) == {"status": "ok"}
    accepted = api.execute(
        request_id="job-1",
        action={"timestamp_s": 0, "values": {"type": "stop"}},
        steps=1,
        wait=False,
    )
    assert accepted["status"] == "accepted"
    assert api.inspect("job-1") == {"status": "ok"}
    assert api.cancel("job-1") == {"status": "pending"}
    assert api.stop("job-1") == {"status": "pending"}
    assert api.media("obs one", frame="front camera") == {"status": "ok"}

    assert len(executor.calls) == 7
    headers = executor.calls[0][4]
    assert headers == {
        "X-EmbodiRun-Caller-Id": "caller-a",
        "X-EmbodiRun-Session-Id": "session-a",
        "X-EmbodiRun-Token": "token-a",
    }
    assert "/v1/observe?runtime_id=fake-device&observation_id=obs+one" in executor.calls[1][1]
    assert "include_robot=true" in executor.calls[1][1]
    assert executor.calls[2][2]["request_id"] == "job-1"
    assert executor.calls[2][2]["action"]["values"]["type"] == "stop"
    assert "jobs/job-1/cancel" in executor.calls[4][1]
    assert "media/obs%20one?frame=front+camera" in executor.calls[6][1]


def test_host_control_client_marks_lost_execute_reply_unknown_without_retry() -> None:
    executor = RecordingExecutor(
        responder=lambda _method, _url, _payload, _headers: (_ for _ in ()).throw(TimeoutError("socket timed out"))
    )
    api = client(executor)

    with pytest.raises(ControlClientError) as raised:
        api.execute(
            request_id="job-timeout",
            action={"timestamp_s": 0, "values": {"type": "stop"}},
        )

    assert raised.value.unknown is True
    assert raised.value.request_id == "job-timeout"
    assert len(executor.calls) == 1


def test_host_control_client_preserves_uncertain_service_result() -> None:
    executor = RecordingExecutor(
        responder=lambda _method, _url, _payload, _headers: JsonHttpResponse(
            409,
            {"status": "uncertain", "request_id": "job-uncertain"},
        )
    )
    api = client(executor)

    with pytest.raises(ControlClientError) as raised:
        api.inspect("job-uncertain")

    assert raised.value.unknown is True
    assert raised.value.status == 409
    assert raised.value.payload["status"] == "uncertain"


def test_host_control_client_reads_media_data_and_recording_routes() -> None:
    executor = RecordingExecutor(
        responder=lambda _method, _url, payload, _headers: JsonHttpResponse(
            200,
            {"status": "ok", "payload": payload or {}},
        )
    )
    api = client(executor)

    api.media(
        "host:123:obs-1",
        frame="front",
        runtime_id="fake-device",
        include_data=True,
    )
    api.recording_status()
    api.recording_start()
    api.recording_stop(recording_timeout_s=2.0)
    api.recording_get("host:123:obs-1")

    assert "media/host:123:obs-1?" in executor.calls[0][1]
    assert "include_data=true" in executor.calls[0][1]
    assert "runtime_id=fake-device" in executor.calls[0][1]
    assert executor.calls[1][1].endswith("/v1/recordings/status")
    assert executor.calls[2][1].endswith("/v1/recordings/start")
    assert executor.calls[3][1].endswith("/v1/recordings/stop")
    assert executor.calls[3][2] == {"timeout_s": 2.0}
    assert executor.calls[4][1].endswith("/v1/recordings/record/host:123:obs-1")


def test_host_control_client_uses_real_local_fake_http_boundary() -> None:
    """Exercise JSON serialization and headers without opening a device."""

    class Service:
        def describe(self):
            return {"service": "fake"}

    application = ControlApplication(Service())
    application.describe = lambda **_kwargs: {"status": "ok"}
    application.observe = lambda **_kwargs: {
        "status": "ok",
        "observation_id": "obs-1",
    }
    application.execute = lambda **kwargs: {
        "status": "accepted",
        "request_id": kwargs["request_id"],
    }
    application.inspect = lambda **kwargs: {
        "status": "completed",
        "request_id": kwargs["request_id"],
    }
    application.media = lambda **kwargs: {
        "observation_id": kwargs["observation_id"],
        "media": [],
    }
    application.recording_get = lambda **kwargs: {
        "observation_id": kwargs["observation_id"],
        "recorded": True,
    }
    api = ControlHTTPAPI(application)
    received_headers: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib handler API
            headers = dict(self.headers.items())
            received_headers.append(headers)
            response = api.dispatch("GET", self.path, headers=headers)
            _send_http_json(self, response.status, response.payload)

        def do_POST(self):  # noqa: N802 - stdlib handler API
            headers = dict(self.headers.items())
            received_headers.append(headers)
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            response = api.dispatch("POST", self.path, body=body, headers=headers)
            _send_http_json(self, response.status, response.payload)

        def log_message(self, _format, *_args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        api_client = HostControlClient(
            # LocalExecutor is exercised here, rather than a fake transport.
            LocalExecutor(),
            endpoint,
            caller_id="caller-http",
            session_id="session-http",
        )
        assert api_client.describe()["status"] == "ok"
        assert (
            api_client.execute(
                request_id="job-http",
                action={"timestamp_s": 0, "values": {"type": "stop"}},
            )["request_id"]
            == "job-http"
        )
        assert api_client.inspect("job-http")["status"] == "completed"
        assert (
            api_client.media(
                "host:123:obs-1",
                runtime_id="fake-device",
                include_data=True,
            )["observation_id"]
            == "host:123:obs-1"
        )
        assert api_client.recording_get("host:123:obs-1")["recorded"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        application.close()

    assert received_headers

    def _header_value(headers: dict[str, str], name: str) -> str | None:
        # Transport stacks may re-case header names (urllib title-cases each
        # dash-separated word); HTTP header names are case-insensitive.
        lowered = {key.lower(): value for key, value in headers.items()}
        return lowered.get(name.lower())

    assert all(_header_value(headers, "X-EmbodiRun-Caller-Id") == "caller-http" for headers in received_headers)
    assert all(_header_value(headers, "X-EmbodiRun-Session-Id") == "session-http" for headers in received_headers)


def test_json_execute_command_emits_only_result_and_keeps_original_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, executor = _fake_context(tmp_path)
    action_path = tmp_path / "action.json"
    action_path.write_text(
        json.dumps(
            {
                "timestamp_s": 0,
                "values": {"type": "stop"},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(tmp_path / "unused.yaml"),
            "execute",
            "--runtime",
            "fake-device",
            "--caller-id",
            "caller-a",
            "--session-id",
            "session-a",
            "--request-id",
            "job-cli",
            "--action",
            str(action_path),
            "--json",
        ]
    )
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)

    exit_code = args.command_handler(args, context)

    assert exit_code == 0
    assert json.loads(output.getvalue()) == {
        "request_id": "job-cli",
        "status": "accepted",
    }
    assert len(executor.calls) == 1
    assert executor.calls[0][2]["request_id"] == "job-cli"


def test_json_execute_timeout_has_distinct_exit_and_no_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, executor = _fake_context(tmp_path, timeout=True)
    action_path = tmp_path / "action.json"
    action_path.write_text('{"timestamp_s": 0, "values": {"type": "stop"}}')
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(tmp_path / "unused.yaml"),
            "execute",
            "--runtime",
            "fake-device",
            "--caller-id",
            "caller-a",
            "--session-id",
            "session-a",
            "--request-id",
            "job-timeout",
            "--action",
            str(action_path),
        ]
    )
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)

    exit_code = args.command_handler(args, context)

    result = json.loads(output.getvalue())
    assert exit_code == EXIT_UNKNOWN
    assert result["status"] == "unknown"
    assert result["request_id"] == "job-timeout"
    assert result["next"].startswith("inspect")
    assert len(executor.calls) == 1


class FakeContext:
    def __init__(self, config, deployment, state_path, executor) -> None:
        self.config = config
        self.deployment = deployment
        self.state_path = state_path
        self._executor = executor

    @contextmanager
    def executor(self, _node_id):
        yield self._executor


def _fake_context(tmp_path: Path, *, timeout: bool = False):
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text("fake: config\n", encoding="utf-8")
    config = SimpleNamespace(path=config_path)
    runtime = RuntimeSpec(
        runtime_id="fake-device",
        node="local",
        target_kind="robot",
        target_id="fake-arm",
        model=None,
        binding=None,
        environment_id="deploy-local-fake",
        model_endpoint=None,
        service_id="control-fake-arm",
        service_endpoint="http://127.0.0.1:8100",
    )
    deployment = SimpleNamespace(deploy_commit="main", runtimes=(runtime,))
    state_path = tmp_path / "state.json"
    StateStore(state_path).save(
        DeploymentState(
            name="fake",
            config_digest=config_digest(config),
            deploy_commit="main",
            inference_commit=None,
            nodes={
                "local": NodeState(
                    node_id="local",
                    home="/tmp",
                    root="/tmp/embodirun",
                    deploy_project="/tmp/embodirun/deploy",
                    inference_project="/tmp/embodirun/inference",
                    platform="test",
                    machine="test",
                    python="python3",
                    python_version="3.12",
                )
            },
            environments={
                "deploy-local-fake": EnvironmentState(
                    environment_id="deploy-local-fake",
                    node="local",
                    project="deploy",
                    group="simulated.joints",
                    path="/tmp/venv",
                    status="ready",
                )
            },
            services={
                "control-fake-arm": ServiceState(
                    service_id="control-fake-arm",
                    node="local",
                    status="running",
                    pid=1,
                    endpoint="http://127.0.0.1:8100",
                )
            },
        )
    )
    if timeout:
        executor = RecordingExecutor(
            responder=lambda _method, _url, _payload, _headers: (_ for _ in ()).throw(TimeoutError("socket timed out"))
        )
    else:
        executor = RecordingExecutor()
    return FakeContext(config, deployment, state_path, executor), executor


def _send_http_json(handler: BaseHTTPRequestHandler, status: int, payload) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
