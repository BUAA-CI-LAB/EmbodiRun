"""Service-level cancellation and resource ownership, without physical devices."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.arbitration import CommandCancelled
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.server import ControlHttpServer, ControlService, ControlTaskRejected
from embodirun.services.control.teleop import ControlHttpTeleopClient
from test_control_services import _unused_loopback_port, control_config, task_request


@pytest.fixture
def controlled_service(monkeypatch, tmp_path):
    started, release, moved = threading.Event(), threading.Event(), threading.Event()
    actions, stops, closed = [], [], []

    class Robot:
        robot_id = "so101-1"

        def __init__(self, config):
            pass

        def connect(self):
            pass

        def observe(self):
            captured = time.monotonic_ns()
            return RobotObservation(
                0,
                {},
                metadata={
                    "captured_timestamp_ns": captured,
                    "clock_domain": "host_monotonic_ns",
                },
            )

        def execute(self, action):
            if action.values.get("fail"):
                raise RuntimeError("motor feedback lost")
            actions.append(action)
            moved.set()

        def stop(self):
            stops.append(True)

        def close(self):
            closed.append(True)

    class Client:
        def health(self):
            return {"status": "ok"}

        def open_session(self, **kwargs):
            return SimpleNamespace(session_id="s")

        def step(self, request):
            started.set()
            assert release.wait(3)
            return object()

        def close(self, session_id):
            pass

        def shutdown(self):
            pass

    mapper = SimpleNamespace(
        policy_action_space="test",
        map_observation=lambda *args, **kwargs: SimpleNamespace(metadata={}),
        map_result=lambda result: (RobotAction(0, {"model": True}),),
    )
    monkeypatch.setattr(
        "embodirun.application.control_service._definitions",
        lambda config: (
            SimpleNamespace(maximum_chunk_steps=1, mapper_factory=lambda: mapper),
            SimpleNamespace(config_factory=lambda *args: None, adapter_type=Robot),
        ),
    )
    service = ControlService(
        control_config(),
        camera_factory=lambda inputs: SimpleNamespace(
            capture=lambda: (
                CameraFrame(
                    "observation.images.front",
                    "image/jpeg",
                    b"takeover-frame",
                    captured_timestamp_ns=time.monotonic_ns(),
                    received_timestamp_ns=time.monotonic_ns(),
                    clock_domain="host_monotonic_ns",
                ),
            ),
            close=lambda: None,
        ),
        client_factory=lambda *args: Client(),
        device_manager=DeviceManager("node-takeover", lock_dir=tmp_path / "locks"),
    )
    yield service, started, release, moved, actions, stops, closed
    release.set()
    service.close()


@pytest.mark.parametrize("release_manual", [False, True])
def test_takeover_cancels_late_inference_and_task_cleanup_keeps_robot(controlled_service, release_manual):
    service, started, release, moved, actions, stops, closed = controlled_service
    errors = []

    def run():
        try:
            service.execute(replace(task_request(), chunk_steps=1, max_steps=1))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(2)
    service.acquire_manual()
    if release_manual:
        service.release_manual()
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert errors and "cancel" in str(errors[0])
    assert not actions and not closed
    if not release_manual:
        service.set_manual_deadman(True)
        service.submit_manual_action({"values": {"manual": True}})
        assert moved.wait(2)
        assert service.control_snapshot()["authority"] == "manual"


def test_idle_estop_blocks_tasks_and_reset_does_not_resurrect_old_work(
    controlled_service,
):
    service, _, _, _, actions, _, _ = controlled_service
    assert service.emergency_stop()["authority"] == "estop_latched"
    with pytest.raises(ControlTaskRejected, match="latched"):
        service.execute(task_request())
    assert not actions
    service.reset_emergency_stop()
    service.acquire_manual()
    service.emergency_stop()
    assert service.reset_emergency_stop()["authority"] == "manual"
    assert service.control_snapshot()["deadman_active"] is False


def test_model_sink_rejects_pre_takeover_result_even_after_release(controlled_service):
    service, *_ = controlled_service
    service.acquire_manual()
    service.release_manual()
    arbiter = service._current_arbiter
    cancel = arbiter.begin_model_task()
    service.acquire_manual()
    service.release_manual()
    with pytest.raises(CommandCancelled):
        arbiter.submit_model(RobotAction(0, {}), task_cancel=cancel)


def test_manual_execution_failure_is_visible_and_latched(controlled_service):
    service, *_ = controlled_service
    service.acquire_manual()
    service.set_manual_deadman(True)
    arbiter = service._current_arbiter
    ticket = arbiter.submit_manual(RobotAction(0, {"fail": True}))
    ticket.wait(2)
    # Synchronize on the worker's following hold rather than assuming ticket
    # completion and fault reporting are a single thread scheduling instant.
    with arbiter._condition:
        assert arbiter._condition.wait_for(lambda: arbiter._authority.value == "estop_latched", timeout=2)
    state = service.control_snapshot()
    assert state["authority"] == "estop_latched"
    assert state["last_error"] == "motor feedback lost"


def test_normal_task_completion_holds_without_disconnecting_robot(controlled_service):
    service, _, release, _, actions, stops, closed = controlled_service
    release.set()
    result = service.execute(replace(task_request(), chunk_steps=1, max_steps=1))
    assert result.completed_steps == 1
    assert len(actions) == 1
    assert stops and not closed


def test_http_manual_control_and_emergency_stop(controlled_service):
    service, _, _, moved, actions, _, _ = controlled_service
    service.config = replace(service.config, port=_unused_loopback_port())
    server = ControlHttpServer(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControlHttpTeleopClient(f"http://127.0.0.1:{server.server_port}")
    try:
        client.acquire_manual()
        client.set_deadman(True)
        client.submit_manual(RobotAction(0, {"manual": True}))
        assert moved.wait(2)
        assert len(actions) == 1
        client.emergency_stop()
        assert client.snapshot()["authority"] == "estop_latched"
        with pytest.raises(RuntimeError, match="latched"):
            client.submit_manual(RobotAction(0, {"manual": True}))
        client.reset_emergency_stop()
        assert client.snapshot()["authority"] == "manual"
        assert client.snapshot()["deadman_active"] is False
        client.release_manual()
        assert client.snapshot()["authority"] == "model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
