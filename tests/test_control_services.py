"""Boundary tests for Host-to-Control contracts and control-node execution."""

from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import replace
from threading import Thread
from types import MappingProxyType, SimpleNamespace

import pytest

from embodirun.robots import RobotObservation
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.contracts import (
    ControlContractError,
    ControlServiceConfig,
    TaskRequest,
    TaskResult,
)
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.server import (
    ControlHttpServer,
    ControlService,
    ControlServiceError,
)
from embodirun.services.host.control import ControlClient
from embodirun.services.host.executor import LocalExecutor
from embodirun.services.inference import EmbodiInferWirelessClient
from embodirun.services.simulation.contracts import (
    EpisodeRequest,
    SimulationServiceConfig,
)
from embodirun.services.simulation.runtime import EpisodeOutcome
from embodirun.services.simulation.server import SimulationService


def control_config(*, port: int = 8100) -> ControlServiceConfig:
    return ControlServiceConfig(
        runtime_id="so101-runtime",
        binding_kind="lerobot.so101.pi05",
        bind="127.0.0.1",
        port=port,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:8000",
        inference_options={},
        robot_id="so101-1",
        robot_kind="lerobot.so101",
        robot_options={"port": "/dev/ttyACM0"},
        inputs=(
            SensorInput(
                "front-camera",
                "observation.images.front",
                "v4l2",
                {"device": "/dev/video0"},
            ),
        ),
        runtime_options={},
    )


def task_request() -> TaskRequest:
    return TaskRequest(
        request_id="task-1",
        runtime_id="so101-runtime",
        prompt="抓取黄色格子",
        chunk_steps=10,
        max_steps=2,
        control_hz=5.0,
        inference_timeout_s=60.0,
    )


def test_host_control_contract_contains_only_dynamic_task_fields() -> None:
    payload = task_request().to_payload()

    assert TaskRequest.from_payload(payload) == task_request()
    assert payload["schema"] == "rlinf.control.task.v1"
    assert "robot" not in payload
    assert "inputs" not in payload
    assert "binding" not in payload
    assert "inference" not in payload


def test_host_control_contract_rejects_unknown_fields() -> None:
    payload = task_request().to_payload()
    payload["robot"] = {"id": "must-not-cross-this-boundary"}

    with pytest.raises(ControlContractError, match="unknown fields: robot"):
        TaskRequest.from_payload(payload)


def test_control_service_config_round_trip_keeps_static_node_ownership() -> None:
    config = control_config()

    decoded = ControlServiceConfig.from_json(config.to_json())

    assert decoded == config
    assert decoded.robot_options == {"port": "/dev/ttyACM0"}
    assert decoded.inputs[0].sensor_id == "front-camera"


def test_control_service_rejects_non_loopback_task_listener() -> None:
    with pytest.raises(ValueError, match="loopback"):
        replace(control_config(), bind="0.0.0.0")


def test_host_client_rejects_non_loopback_control_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        ControlClient(LocalExecutor(), "http://192.168.10.10:8100")


def test_control_service_executes_chunks_and_releases_resources(monkeypatch, tmp_path) -> None:
    events: list[object] = []

    class Client:
        def health(self):
            return {"status": "ok"}

        def shutdown(self):
            events.append("client.shutdown")

    class Cameras:
        def capture(self):
            events.append("camera.capture")
            captured = time.monotonic_ns()
            return (
                CameraFrame(
                    "observation.images.front",
                    "image/jpeg",
                    b"frame",
                    captured_timestamp_ns=captured,
                    received_timestamp_ns=captured,
                    clock_domain="host_monotonic_ns",
                ),
            )

        def close(self):
            events.append("camera.close")

    class Robot:
        robot_id = "so101-1"

        def __init__(self, config):
            events.append(("robot.config", config))

        def connect(self):
            events.append("robot.connect")

        def observe(self):
            captured = time.monotonic_ns()
            return RobotObservation(
                1.0,
                {"state": 1},
                metadata={
                    "captured_timestamp_ns": captured,
                    "clock_domain": "host_monotonic_ns",
                },
            )

        def close(self):
            events.append("robot.close")

        def stop(self):
            events.append("robot.stop")

    class Runtime:
        def __init__(self, robot, client, **options):
            events.append(("runtime.options", options))

        def step(self, frames):
            events.append(("runtime.step", frames))

        def close(self):
            events.append("runtime.close")

    binding = SimpleNamespace(
        kind="lerobot.so101.pi05",
        maximum_chunk_steps=50,
        mapper_factory=lambda: "mapper",
    )
    robot = SimpleNamespace(
        config_factory=lambda robot_id, options: (robot_id, dict(options)),
        adapter_type=Robot,
    )
    monkeypatch.setattr(
        "embodirun.application.control_service._definitions",
        lambda _config: (binding, robot),
    )
    service = ControlService(
        control_config(),
        camera_factory=lambda _inputs: Cameras(),
        client_factory=lambda _config, _timeout: Client(),
        runtime_factory=Runtime,
        device_manager=DeviceManager("node-test", lock_dir=tmp_path / "locks"),
    )

    try:
        result = service.execute(task_request())

        assert result == TaskResult("task-1", "so101-runtime", 2)
        assert events.count("camera.capture") >= 2
        runtime_steps = [item for item in events if isinstance(item, tuple) and item[0] == "runtime.step"]
        assert len(runtime_steps) == 2
        assert all(frames and frames[0].data == b"frame" for _, frames in runtime_steps)
        assert events[-2:] == [
            "runtime.close",
            "client.shutdown",
        ]
        assert "robot.close" not in events
    finally:
        service.close()
    assert events.count("camera.close") == 1
    assert "robot.stop" in events
    assert events[-1] == "robot.close"
    runtime_options = next(
        value for item in events if isinstance(item, tuple) and item[0] == "runtime.options" for value in (item[1],)
    )
    assert runtime_options["instruction"] == "抓取黄色格子"
    assert runtime_options["chunk_steps"] == 10
    assert runtime_options["control_hz"] == 5.0


def test_control_service_serializes_robot_tasks() -> None:
    service = ControlService(control_config())
    service._task_lock.acquire()
    try:
        with pytest.raises(ControlServiceError, match="already executing"):
            service.execute(task_request())
    finally:
        service._task_lock.release()


def test_control_service_reuses_wireless_client_across_health_checks() -> None:
    created = 0
    shutdown = 0

    class Client:
        def health(self):
            return {"status": "ok"}

        def shutdown(self):
            nonlocal shutdown
            shutdown += 1

    def client_factory(_config, _timeout):
        nonlocal created
        created += 1
        return Client()

    service = ControlService(
        replace(
            control_config(),
            inference_transport="wireless",
            inference_endpoint="wireless://inference-thor",
            inference_options={
                "comm_config": "/etc/rlinf/control-wireless.yaml",
                "server_node_id": "inference-thor",
            },
        ),
        client_factory=client_factory,
    )

    assert service.health()["status"] == "ok"
    assert service.health()["status"] == "ok"
    assert created == 1
    assert shutdown == 0

    service.close()

    assert shutdown == 1


def test_host_client_submits_task_to_control_http_service() -> None:
    port = _unused_loopback_port()
    config = control_config(port=port)

    class Service:
        def __init__(self):
            self.config = config
            self.received = None

        def health(self):
            return {"status": "ok", "runtime_id": config.runtime_id}

        def execute(self, request):
            self.received = request
            return TaskResult(request.request_id, request.runtime_id, 2)

    service = Service()
    server = ControlHttpServer(service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = ControlClient(LocalExecutor(), f"http://127.0.0.1:{port}").run(task_request(), timeout_s=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert result == TaskResult("task-1", "so101-runtime", 2)
    assert service.received == task_request()
    assert not thread.is_alive()


@pytest.mark.parametrize("episode_fails", [False, True])
def test_simulation_keeps_wireless_connection_until_service_close(monkeypatch, episode_fails) -> None:
    events = []
    calls = []
    created = []

    class Transport:
        def request(self, method, payload, *, timeout_s):
            calls.append((method, timeout_s))
            if method == "health":
                return {"status": "ok"}
            if method == "open_session":
                return {"session_id": "session-1", "session_revision": 0}
            if method == "close":
                events.append("session.close")
                return {}
            raise AssertionError(method)

        def shutdown(self):
            events.append("transport.shutdown")

    def client_factory(_config, timeout_s):
        client = EmbodiInferWirelessClient(Transport(), timeout_s=timeout_s)
        created.append(client)
        return client

    class Simulator:
        def __init__(self, config):
            pass

        def close(self):
            events.append("simulator.close")

    class Runtime:
        def __init__(self, simulator, client, **options):
            self.client = client
            self.session = None

        def run(self, **options):
            self.session = self.client.open_session(robot_id="sim", action_space="pi05.action_chunk.v1")
            # A health request while an episode owns the connection must be safe.
            assert service.health()["status"] == "ok"
            assert self.client.timeout_s == 60.0
            if episode_fails:
                raise RuntimeError("episode failed")
            return EpisodeOutcome(1, 1, 0.0, True, False)

        def close(self):
            if self.session is not None:
                self.client.close(self.session.session_id)

    monkeypatch.setattr(
        "embodirun.application.simulation.service.simulator_definition",
        lambda _kind: SimpleNamespace(
            embodiment_kind="franka.panda.eef",
            config_factory=lambda _id, options: options,
            adapter_type=Simulator,
        ),
    )
    service = SimulationService(
        SimulationServiceConfig(
            runtime_id="sim-runtime",
            binding_kind="franka.panda.pi05",
            bind="127.0.0.1",
            port=8100,
            inference_transport="wireless",
            inference_endpoint="wireless://inference-1",
            inference_options={},
            simulator_id="sim",
            simulator_kind="vlabench",
            simulator_options={},
        ),
        client_factory=client_factory,
        runtime_factory=Runtime,
    )
    request = EpisodeRequest("episode-1", "sim-runtime", "pick up", None, 0, 1, 1, 60.0)
    try:
        assert service.health()["status"] == "ok"
        for _ in range(2):
            if episode_fails:
                with pytest.raises(RuntimeError, match="episode failed"):
                    service.execute(request)
            else:
                assert service.execute(request).policy_steps == 1
            assert service.health()["status"] == "ok"
        assert len(created) == 1
        assert events == ["session.close", "simulator.close"] * 2
        assert [timeout for method, timeout in calls if method == "open_session"] == [60.0] * 2
        assert calls.count(("health", 2.0)) == 5
        assert calls.count(("health", 60.0)) == 2
    finally:
        service.close()
    service.close()
    assert events.count("transport.shutdown") == 1


def test_public_observe_serializes_nested_immutable_robot_metadata() -> None:
    port = _unused_loopback_port()
    config = control_config(port=port)

    class Service:
        def __init__(self):
            self.config = config

        def observe(self):
            return {
                "status": "ok",
                "robot": {
                    "metadata": MappingProxyType(
                        {
                            "raw_fields": MappingProxyType({"owner_present": False}),
                            "errors": (),
                        }
                    )
                },
            }

    service = Service()
    server = ControlHttpServer(service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", "/v1/observe")
        response = connection.getresponse()
        payload = json.loads(response.read().decode())
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 200
    assert payload["robot"]["metadata"]["raw_fields"]["owner_present"] is False
    assert payload["robot"]["metadata"]["errors"] == []


def _unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
