from __future__ import annotations

import base64
import time
from threading import Event
from types import SimpleNamespace

import pytest

from embodirun.application import control_service as control_service_impl
from embodirun.robots import RobotAdapter, RobotDefinition, RobotObservation
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.contracts import (
    ControlServiceConfig,
    TaskRequest,
)
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.recordings import ActionEvent, ObservationRecorder
from embodirun.services.control.server import ControlService


def _wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class _Camera:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.capture_count = 0
        self.close_count = 0

    def capture(self) -> tuple[CameraFrame, ...]:
        self.capture_count += 1
        captured = time.monotonic_ns()
        self.events.append(("camera.capture", self.capture_count, captured))
        return (
            CameraFrame(
                "front",
                "image/jpeg",
                b"shared-camera-bytes",
                captured_timestamp_ns=captured,
                received_timestamp_ns=captured,
                clock_domain="host_monotonic_ns",
                profile={"width": 2, "height": 1},
            ),
        )

    def close(self) -> None:
        self.close_count += 1
        self.events.append("camera.close")


class _Robot(RobotAdapter):
    robot_id = "fake-arm"

    def __init__(self, _config, events: list[object]) -> None:
        self.events = events
        self.connect_calls: list[bool] = []
        self.stop_count = 0
        self.close_count = 0

    def connect(self, *, prepare: bool = True) -> None:
        self.connect_calls.append(prepare)
        self.events.append(("robot.connect", prepare))

    def prepare(self) -> None:
        self.events.append("robot.prepare")

    def observe(self) -> RobotObservation:
        captured = time.monotonic_ns()
        return RobotObservation(
            timestamp_s=1.0,
            values={"joints": [1.0, 2.0], "gripper": 0.25},
            metadata={
                "captured_timestamp_ns": captured,
                "clock_domain": "host_monotonic_ns",
            },
        )

    def execute(self, _action) -> None:
        return None

    def stop(self) -> None:
        self.stop_count += 1
        self.events.append("robot.stop")

    def close(self) -> None:
        self.close_count += 1
        self.events.append("robot.close")


def _config() -> ControlServiceConfig:
    sensor = SensorInput(
        "camera",
        "front",
        "fake",
        {"device": "fake-camera"},
    )
    return ControlServiceConfig(
        runtime_id="fake-runtime",
        binding_kind="fake.binding",
        bind="127.0.0.1",
        port=18130,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        robot_id="fake-arm",
        robot_kind="fake.robot",
        robot_options={"port": "fake-robot"},
        inputs=(sensor,),
        runtime_options={},
        node_id="node-test",
        device_resources=(
            {
                "identity": "node-test:sensor:fake-camera",
                "node": "node-test",
                "kind": "sensor",
                "value": "fake-camera",
            },
            {
                "identity": "node-test:robot:fake-robot",
                "node": "node-test",
                "kind": "robot",
                "value": "fake-robot",
            },
        ),
    )


def test_service_shares_one_camera_and_snapshot_across_four_consumers(monkeypatch, tmp_path) -> None:
    events: list[object] = []
    cameras: list[_Camera] = []
    robots: list[_Robot] = []
    model_inputs: list[tuple[RobotObservation, tuple[CameraFrame, ...]]] = []
    runtime_closed = Event()

    def camera_factory(_inputs):
        camera = _Camera(events)
        cameras.append(camera)
        return camera

    class Client:
        def health(self):
            return {"status": "ok"}

        def shutdown(self):
            events.append("client.shutdown")

    class Runtime:
        def __init__(self, _robot, _client, **options):
            self._observation_source = options["observation_source"]

        def step(self, frames):
            observation = self._observation_source()
            model_inputs.append((observation, tuple(frames)))

        def close(self):
            runtime_closed.set()

    binding = SimpleNamespace(
        kind="fake.binding",
        maximum_chunk_steps=50,
        mapper_factory=lambda: object(),
    )

    class Adapter(_Robot):
        def __init__(self, config):
            super().__init__(config, events)
            robots.append(self)

    definition = RobotDefinition(
        kind="fake.robot",
        config_factory=lambda _robot_id, _options: object(),
        adapter_type=Adapter,
        environment_group="test",
    )
    monkeypatch.setattr(
        control_service_impl,
        "_definitions",
        lambda _config: (binding, definition),
    )

    service = ControlService(
        _config(),
        camera_factory=camera_factory,
        client_factory=lambda _config, _timeout: Client(),
        runtime_factory=Runtime,
        device_manager=DeviceManager("node-test", lock_dir=tmp_path / "locks"),
    )
    recorder = ObservationRecorder(
        service.observation_store,
        tmp_path,
        "multi-consumer",
        queue_size=1,
        require_state=True,
        clock_ns=time.monotonic_ns,
    )
    service.attach_recorder(recorder)
    subscription = service.subscribe(max_queue=1)
    release_recorder = Event()
    recorder_entered = Event()

    try:
        recorder.start()
        first = service.observe(include_robot=True)
        first_id = first["snapshot_id"]
        _wait_until(lambda: recorder.get_record(first_id) is not None)

        # Pinning the same ID is the preview path: it must reuse the exact
        # encoded bytes and state selected by the agent observation.
        preview = service.get_snapshot(first_id, include_robot=True)
        assert preview["snapshot_id"] == first_id
        assert preview["frames"] == first["frames"]
        assert preview["robot"] == first["robot"]

        original_write = recorder._write_snapshot

        def slow_write(snapshot) -> None:
            if not recorder_entered.is_set():
                recorder_entered.set()
                assert release_recorder.wait(2.0)
            original_write(snapshot)

        recorder._write_snapshot = slow_write
        result = service.execute(
            TaskRequest(
                request_id="fake-task",
                runtime_id="fake-runtime",
                prompt="test",
                chunk_steps=1,
                max_steps=3,
                control_hz=5.0,
                inference_timeout_s=2.0,
            )
        )
        assert result.completed_steps == 3
        assert runtime_closed.is_set()
        assert recorder_entered.wait(1.0)
        assert len(model_inputs) == 3
        model_observation, model_frames = model_inputs[0]
        model_id = model_observation.metadata["observation_id"]
        assert model_id != first_id
        assert model_frames[0].data == b"shared-camera-bytes"
        assert model_observation.values == {
            "joints": [1.0, 2.0],
            "gripper": 0.25,
        }
        assert cameras[0].close_count == 0

        # The slow preview subscriber and recorder both expose bounded loss;
        # neither loss path closes the source or blocks model execution.
        assert subscription.dropped_count >= 1
        assert recorder.status().dropped_observations >= 1
        recorder.record_action(
            ActionEvent(
                action_id="proposal-1",
                source="model",
                stage="proposal",
                executed=False,
                observation_id=model_id,
            )
        )
        release_recorder.set()
        _wait_until(lambda: recorder.status().observation_count >= 2)
        recorder.stop(timeout_s=1.0)

        model_view = service.get_snapshot(model_id, include_robot=True)
        assert model_view["frames"][0]["data"] == base64.b64encode(b"shared-camera-bytes").decode("ascii")
        assert model_view["robot"]["values"] == model_observation.values
        record = recorder.get_record(first_id)
        assert record is not None
        media_path = tmp_path / "multi-consumer" / record["cameras"][0]["path"]
        assert media_path.read_bytes() == b"shared-camera-bytes"
        actions = list(recorder.iter_actions())
        assert actions and actions[-1]["executed"] is False
        assert actions[-1]["observation_id"] == model_id

        subscription.close()
        assert service.get_snapshot(first_id)["frames"] == first["frames"]
        assert len(cameras) == 1
        assert len(robots) == 1
    finally:
        release_recorder.set()
        subscription.close()
        service.close()

    assert cameras[0].close_count == 1
    assert robots[0].close_count == 1
    assert robots[0].stop_count >= 1
    assert service._shared_observations.producer.running is False


@pytest.mark.parametrize("consumer", ["subscription", "recorder"])
def test_consumer_exit_does_not_release_shared_camera(consumer, tmp_path) -> None:
    events: list[object] = []
    cameras: list[_Camera] = []

    def factory(_inputs):
        camera = _Camera(events)
        cameras.append(camera)
        return camera

    service = ControlService(
        _config(),
        camera_factory=factory,
        device_manager=DeviceManager("node-test", lock_dir=tmp_path / "locks"),
    )
    recorder = None
    subscription = None
    try:
        if consumer == "subscription":
            subscription = service.subscribe(max_queue=1)
            service.observe()
            subscription.close()
        else:
            recorder = ObservationRecorder(service.observation_store, tmp_path, "exit")
            service.attach_recorder(recorder)
            recorder.start()
            service.observe()
            recorder.stop(timeout_s=1.0)
        assert cameras[0].close_count == 0
        service.observe()
        assert cameras[0].capture_count >= 1
    finally:
        if subscription is not None:
            subscription.close()
        service.close()
    assert cameras[0].close_count == 1
    assert service._shared_observations.producer.running is False
