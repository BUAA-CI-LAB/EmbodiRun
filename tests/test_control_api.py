from __future__ import annotations

import base64
import http.client
import json
import socket
import threading

import pytest

from embodirun.robots import RobotAction
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.application import (
    ApplicationStaleObservation,
    ControlApplication,
)
from embodirun.services.control.arbitration import (
    CommandSource,
    RobotControlArbiter,
)
from embodirun.services.control.auth import AuthPolicy
from embodirun.services.control.contracts import (
    TASK_REQUEST_SCHEMA,
    ControlServiceConfig,
    TaskResult,
)
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.direct_execution import DirectExecutionError, DirectExecutionRunner
from embodirun.services.control.http_api import ControlHTTPAPI
from embodirun.services.control.io import IOResult, IOStatus
from embodirun.services.control.jobs import JobConflict, JobRegistry
from embodirun.services.control.observation_store import ObservationStore
from embodirun.services.control.observation_values import ObservationSnapshot
from embodirun.services.control.server import (
    ControlHttpServer,
    ControlService,
    ControlServiceError,
    ControlTaskRejected,
    _load_auth_policy,
    _sensor_identity,
)


def _action(value: float) -> RobotAction:
    return RobotAction(timestamp_s=value, values={"joint": value})


class GenericPort:
    robot_id = "fake-arm"

    def __init__(self) -> None:
        self.actions: list[RobotAction] = []
        self.holds = 0

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
        self.holds += 1
        return IOResult("stop", IOStatus.COMPLETED, stop_confirmed=True)

    def emergency_stop(self) -> IOResult:
        return self.hold()

    def close(self) -> None:
        return None


class FakeService:
    def __init__(self) -> None:
        self.observe_calls = 0
        self.execute_calls = 0
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.release = threading.Event()

    def describe(self) -> dict[str, object]:
        return {"status": "ok", "robot_id": "fake-arm"}

    def observe(self, *, runtime_id: str | None, include_robot: bool) -> dict[str, object]:
        self.observe_calls += 1
        return {
            "status": "ok",
            "runtime_id": runtime_id,
            "robot": {"joint": 99} if include_robot else None,
        }

    def execute(self, request, *, cancel_event: threading.Event | None = None) -> TaskResult:
        self.execute_calls += 1
        self.started.set()
        if cancel_event is not None:
            while not (self.release.is_set() or cancel_event.wait(0.01)):
                pass
            if cancel_event.is_set():
                self.cancelled.set()
        return TaskResult(request.request_id, request.runtime_id, 1)

    def cancel_task(self, cancel_event: threading.Event) -> dict[str, object]:
        cancel_event.set()
        return {"physical_status": "stop_requested", "stop_confirmed": False}


def _snapshot(store: ObservationStore, now_ns: int) -> ObservationSnapshot:
    observation_id, generation, sequence = store.next_observation_id()
    return store.publish(
        ObservationSnapshot(
            observation_id=observation_id,
            service_instance_id=store.service_instance_id,
            generation=generation,
            sequence=sequence,
            state={"joint": 3, "nested": {"same": True}},
            cameras=(
                CameraFrame(
                    "front",
                    "image/jpeg",
                    b"jpeg",
                    captured_timestamp_ns=now_ns - 100,
                    received_timestamp_ns=now_ns - 50,
                    clock_domain="host_monotonic_ns",
                ),
            ),
            metadata={"clock_domain": "host_monotonic_ns", "source": "fake"},
            captured_timestamp_ns=now_ns - 100,
            received_timestamp_ns=now_ns - 50,
            published_timestamp_ns=now_ns,
            source_timestamps_ns={"front": now_ns - 100},
            source_received_timestamps_ns={"front": now_ns - 50},
            clock_domains={"front": "host_monotonic_ns"},
            skew_ns=0,
        )
    )


def test_shared_observation_is_consistent_and_direct_segment_holds_owned_token() -> None:
    service = FakeService()
    now_ns = 10_000_000
    store = ObservationStore(service_instance_id="fake-service")
    snapshot = _snapshot(store, now_ns)
    port = GenericPort()
    arbiter = RobotControlArbiter(port)
    app = ControlApplication(
        service,
        observation_store=store,
        arbiter_provider=lambda: arbiter,
        clock_ns=lambda: now_ns,
    )
    try:
        observed = app.observe(
            caller_id="caller",
            session_id="session",
            observation_id=snapshot.observation_id,
            include_robot=True,
        )
        assert observed["observation_id"] == snapshot.observation_id
        assert observed["robot"] == {"joint": 3, "nested": {"same": True}}
        assert observed["media"][0]["media_ref"].startswith("observation://")
        assert service.observe_calls == 0
        media_response = ControlHTTPAPI(app).dispatch(
            "GET",
            f"/v1/media/{snapshot.observation_id}?frame=front&include_data=true",
        )
        assert media_response.status == 200
        assert media_response.payload["media"][0]["data_base64"] == base64.b64encode(b"jpeg").decode("ascii")

        record = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="direct-1",
            action={"timestamp_s": 1.0, "values": {"joint": 7}},
            source=CommandSource.AGENT,
            observation_id=snapshot.observation_id,
            wait=True,
        )
        assert record["status"] == "completed"
        assert record["inference_status"] == "not_requested"
        assert record["result"]["business_success"] is None
        assert record["result"]["physical_status"] == "stop_requested"
        assert [item.values["joint"] for item in port.actions] == [7]
        assert port.holds >= 1
    finally:
        app.close()
        arbiter.close(hold=False)


def test_direct_idempotency_conflict_and_manual_source_rejection() -> None:
    service = FakeService()
    port = GenericPort()
    arbiter = RobotControlArbiter(port)
    app = ControlApplication(service, arbiter_provider=lambda: arbiter)
    try:
        first = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="same",
            action={"timestamp_s": 1.0, "values": {"joint": 1}},
            wait=True,
        )
        duplicate = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="same",
            action={"timestamp_s": 1.0, "values": {"joint": 1}},
            wait=True,
        )
        assert duplicate["run_id"] == first["run_id"]
        assert len(port.actions) == 1
        with pytest.raises(JobConflict):
            app.execute(
                caller_id="caller",
                session_id="session",
                request_id="same",
                action={"timestamp_s": 1.0, "values": {"joint": 2}},
                wait=False,
            )
        with pytest.raises(ValueError, match="agent.*replay"):
            app.execute(
                caller_id="caller",
                session_id="session",
                request_id="manual",
                action={"timestamp_s": 1.0, "values": {"joint": 2}},
                source="manual",
            )
    finally:
        app.close()
        arbiter.close(hold=False)


def test_duplicate_explicit_observation_reuses_result_after_snapshot_expires() -> None:
    service = FakeService()
    now_ns = 10_000_000
    store = ObservationStore(max_retained=1, service_instance_id="expired-idempotency")
    first_snapshot = _snapshot(store, now_ns)
    port = GenericPort()
    arbiter = RobotControlArbiter(port)
    app = ControlApplication(
        service,
        observation_store=store,
        arbiter_provider=lambda: arbiter,
        clock_ns=lambda: now_ns,
    )
    try:
        first = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="expired-explicit",
            action={"timestamp_s": 1.0, "values": {"joint": 1}},
            observation_id=first_snapshot.observation_id,
            wait=True,
        )
        _snapshot(store, now_ns)
        duplicate = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="expired-explicit",
            action={"timestamp_s": 1.0, "values": {"joint": 1}},
            observation_id=first_snapshot.observation_id,
            wait=True,
        )
        assert duplicate["run_id"] == first["run_id"]
        assert len(port.actions) == 1
    finally:
        app.close()
        arbiter.close(hold=False)


def test_duplicate_without_observation_binding_reuses_result_after_latest_changes() -> None:
    service = FakeService()
    now_ns = 10_000_000
    store = ObservationStore(max_retained=2, service_instance_id="latest-idempotency")
    _snapshot(store, now_ns)
    port = GenericPort()
    arbiter = RobotControlArbiter(port)
    app = ControlApplication(
        service,
        observation_store=store,
        arbiter_provider=lambda: arbiter,
        clock_ns=lambda: now_ns,
    )
    try:
        first = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="latest-omitted",
            action={"timestamp_s": 1.0, "values": {"joint": 2}},
            wait=True,
        )
        _snapshot(store, now_ns)
        duplicate = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="latest-omitted",
            action={"timestamp_s": 1.0, "values": {"joint": 2}},
            wait=True,
        )
        assert duplicate["run_id"] == first["run_id"]
        assert len(port.actions) == 1
    finally:
        app.close()
        arbiter.close(hold=False)


def test_configured_token_binds_caller_and_describe_does_not_open_provider() -> None:
    service = FakeService()
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        raise AssertionError("describe must not resolve the arbiter")

    auth = AuthPolicy(
        {
            "token-a": {
                "role": "controller",
                "caller_id": "caller-a",
                "session_id": "session-a",
            },
            "token-b": {
                "role": "controller",
                "caller_id": "caller-b",
                "session_id": "session-b",
            },
        }
    )
    app = ControlApplication(service, arbiter_provider=provider, auth_policy=auth)
    api = ControlHTTPAPI(app)
    assert (
        api.dispatch(
            "GET",
            "/v1/describe",
            headers={
                "Authorization": "Bearer token-a",
                "X-EmbodiRun-Caller-ID": "caller-a",
                "X-EmbodiRun-Session-ID": "session-a",
            },
        ).status
        == 200
    )
    assert calls == 0
    spoofed = api.dispatch(
        "GET",
        "/v1/describe",
        headers={
            "Authorization": "Bearer token-a",
            "X-EmbodiRun-Caller-ID": "caller-b",
            "X-EmbodiRun-Session-ID": "session-a",
        },
    )
    assert spoofed.status == 403
    assert spoofed.payload["code"] == "forbidden"
    assert "token-a" not in str(spoofed.payload)
    assert api.dispatch("GET", "/v1/observe").status == 401
    app.close()


def test_legacy_rlinf_identity_headers_still_authenticate() -> None:
    """Deprecated ``X-RLinf-*`` identity headers keep authenticating during migration."""
    service = FakeService()
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        raise AssertionError("describe must not resolve the arbiter")

    auth = AuthPolicy(
        {
            "token-a": {
                "role": "controller",
                "caller_id": "caller-a",
                "session_id": "session-a",
            },
        }
    )
    app = ControlApplication(service, arbiter_provider=provider, auth_policy=auth)
    api = ControlHTTPAPI(app)
    try:
        legacy = api.dispatch(
            "GET",
            "/v1/describe",
            headers={
                "Authorization": "Bearer token-a",
                "X-RLinf-Caller-ID": "caller-a",
                "X-RLinf-Session-ID": "session-a",
            },
        )
        assert legacy.status == 200
        assert calls == 0
        spoofed = api.dispatch(
            "GET",
            "/v1/describe",
            headers={
                "Authorization": "Bearer token-a",
                "X-RLinf-Caller-ID": "caller-b",
                "X-RLinf-Session-ID": "session-a",
            },
        )
        assert spoofed.status == 403
        assert spoofed.payload["code"] == "forbidden"
    finally:
        app.close()


def test_http_legacy_task_keeps_task_result_and_request_identity() -> None:
    service = FakeService()
    service.release.set()
    app = ControlApplication(service)
    api = ControlHTTPAPI(app)
    payload = {
        "schema": TASK_REQUEST_SCHEMA,
        "request_id": "legacy-1",
        "runtime_id": "runtime",
        "prompt": "move",
        "chunk_steps": 1,
        "max_steps": 1,
        "control_hz": 10.0,
        "inference_timeout_s": 1.0,
        "wait": True,
    }
    try:
        response = api.dispatch("POST", "/v1/tasks", body=payload)
        assert response.status == 200
        assert TaskResult.from_payload(response.payload).request_id == "legacy-1"
        duplicate = api.dispatch("POST", "/v1/tasks", body=payload)
        assert duplicate.status == 200
        assert service.execute_calls == 1
    finally:
        app.close()


def test_http_recording_routes_use_optional_service_recorder_and_safe_snapshot_id() -> None:
    service = FakeService()
    service.start_recording = lambda: {"recording_id": "r1", "state": "running"}
    service.stop_recording = lambda timeout_s=1.0: {
        "recording_id": "r1",
        "state": "stopped",
        "timeout_s": timeout_s,
    }
    service.recording_status = lambda: {"recording_id": "r1", "state": "running"}
    service.get_record = lambda observation_id: {
        "observation_id": observation_id,
        "media": [],
    }
    app = ControlApplication(service)
    api = ControlHTTPAPI(app)
    try:
        assert api.dispatch("POST", "/v1/recordings/start").payload["state"] == "running"
        assert api.dispatch("GET", "/v1/recordings/status").status == 200
        record = api.dispatch("GET", "/v1/recordings/record/obs-1")
        assert record.status == 200
        assert record.payload["observation_id"] == "obs-1"
        stopped = api.dispatch(
            "POST",
            "/v1/recordings/stop",
            body={"timeout_s": 0.0},
        )
        assert stopped.payload["timeout_s"] == 0.0
    finally:
        app.close()


def test_legacy_cancel_callback_uses_old_event_and_does_not_cancel_new_event() -> None:
    service = FakeService()
    app = ControlApplication(service)
    old_event = threading.Event()
    new_event = threading.Event()
    result = app._cancel_legacy_task(old_event)
    assert result["physical_status"] == "stop_requested"
    assert old_event.is_set()
    assert not new_event.is_set()
    app.close()


def test_observation_freshness_rejects_unknown_or_stale_timing() -> None:
    service = FakeService()
    store = ObservationStore(service_instance_id="freshness")
    snapshot = _snapshot(store, 100)
    app = ControlApplication(
        service,
        observation_store=store,
        clock_ns=lambda: 1_000_000,
        max_observation_age_ns=10,
    )
    try:
        with pytest.raises(ApplicationStaleObservation):
            app.execute(
                caller_id="caller",
                session_id="session",
                request_id="stale",
                action=_action(1),
                observation_id=snapshot.observation_id,
            )
    finally:
        app.close()


def test_direct_segment_rejects_schedule_longer_than_bounded_duration_before_io() -> None:
    port = GenericPort()
    arbiter = RobotControlArbiter(port)
    runner = DirectExecutionRunner(
        lambda: arbiter,
        max_duration_s=0.01,
    )
    try:
        with pytest.raises(DirectExecutionError, match="max_duration_s"):
            runner.run(
                [_action(1), _action(2)],
                source="agent",
                cancel_event=threading.Event(),
                control_hz=1.0,
            )
        assert port.actions == []
    finally:
        arbiter.close(hold=False)


def test_arbiter_event_callback_preserves_source_observation_and_driver_receipt() -> None:
    port = GenericPort()
    events: list[dict[str, object]] = []
    arbiter = RobotControlArbiter(port, event_callback=events.append)
    try:
        command = RobotAction(
            timestamp_s=1.0,
            values={"joint": 5},
            metadata={"observation_id": "obs-1"},
        )
        ticket = arbiter.submit_replay(command, wait=True)
        assert ticket.status.value == "executed"
        assert any(item["stage"] == "queued" for item in events)
        assert any(
            item["source"] == "replay"
            and item["observation_id"] == "obs-1"
            and item["executed"] is True
            and item["payload"]["driver_receipt"] == {"accepted": 1}
            for item in events
        )
    finally:
        arbiter.close(hold=False)


def test_event_callback_failure_does_not_deadlock_or_change_command_status() -> None:
    port = GenericPort()

    def broken_callback(event: dict[str, object]) -> None:
        raise RuntimeError("recorder queue failed")

    arbiter = RobotControlArbiter(port, event_callback=broken_callback)
    try:
        ticket = arbiter.submit_agent(_action(1), wait=True, timeout_s=1.0)
        assert ticket.status.value == "executed"
        assert arbiter.snapshot()["last_event_error"] == "recorder queue failed"
    finally:
        arbiter.close(hold=False)


def test_device_only_direct_and_manual_commands_share_prepared_arbiter(tmp_path) -> None:
    config = ControlServiceConfig.device_only(
        runtime_id="fake-device",
        bind="127.0.0.1",
        port=8123,
        robot_id="fake-arm",
        robot_kind="simulated.joints",
        robot_options={
            "initial_joint_positions_deg": [0, 0, 0, 0, 0],
            "initial_gripper_position": 0,
            "step_limit_mode": "clip",
        },
    )
    manager = DeviceManager(
        "local",
        owner_id="control-api-test",
        lock_dir=tmp_path / "locks",
        state_path=tmp_path / "state.json",
    )
    service = ControlService(config, device_manager=manager)
    app = ControlApplication(
        service,
        registry=JobRegistry(tmp_path / "jobs.sqlite"),
    )
    try:
        assert app.describe(caller_id="caller", session_id="session")["application"]["execute"]
        action = {
            "timestamp_s": 1.0,
            "values": {
                "type": "joint_position",
                "joint_positions_deg": [1, 2, 3, 4, 5],
                "gripper_position": 20,
            },
        }
        result = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="fake-direct",
            action=action,
            wait=True,
        )
        assert result["status"] == "completed"
        assert result["result"]["simulated"] is True
        service.acquire_manual()
        service.set_manual_deadman(True)
        snapshot = service.submit_manual_action(action)
        assert snapshot["ticket_status"] in {"queued", "running", "executed"}
        service.release_manual()
    finally:
        app.close()
        service.close()


def test_manual_scope_rejects_late_old_owner_without_touching_new_owner(
    tmp_path,
) -> None:
    config = ControlServiceConfig.device_only(
        runtime_id="manual-scope",
        bind="127.0.0.1",
        port=8124,
        robot_id="manual-arm",
        robot_kind="simulated.joints",
        robot_options={
            "initial_joint_positions_deg": [0, 0, 0, 0, 0],
            "initial_gripper_position": 0,
            "step_limit_mode": "clip",
        },
    )
    service = ControlService(
        config,
        device_manager=DeviceManager(
            "local",
            owner_id="manual-scope-test",
            lock_dir=tmp_path / "locks",
            state_path=tmp_path / "state.json",
        ),
    )
    try:
        service.acquire_manual(caller_id="caller-a", session_id="session-a")
        service.release_manual(caller_id="caller-a", session_id="session-a")
        service.acquire_manual(caller_id="caller-b", session_id="session-b")
        service.set_manual_deadman(
            True,
            caller_id="caller-b",
            session_id="session-b",
        )

        with pytest.raises(ControlTaskRejected, match="owned by another caller"):
            service.release_manual(caller_id="caller-a", session_id="session-a")
        with pytest.raises(ControlTaskRejected, match="owned by another caller"):
            service.set_manual_deadman(
                False,
                caller_id="caller-a",
                session_id="session-a",
            )
        with pytest.raises(ControlTaskRejected, match="owned by another caller"):
            service.submit_manual_action(
                {
                    "timestamp_s": 1.0,
                    "values": {
                        "type": "joint_position",
                        "joint_positions_deg": [1, 2, 3, 4, 5],
                        "gripper_position": 10,
                    },
                },
                caller_id="caller-a",
                session_id="session-a",
            )

        snapshot = service.control_snapshot()
        assert snapshot["authority"] == "manual"
        assert snapshot["deadman_active"] is True
        service.release_manual(caller_id="caller-b", session_id="session-b")
    finally:
        service.close()


def test_manual_transition_waits_for_inflight_old_owner_operation(tmp_path, monkeypatch) -> None:
    config = ControlServiceConfig.device_only(
        runtime_id="manual-race",
        bind="127.0.0.1",
        port=8125,
        robot_id="manual-race-arm",
        robot_kind="simulated.joints",
        robot_options={
            "initial_joint_positions_deg": [0, 0, 0, 0, 0],
            "initial_gripper_position": 0,
            "step_limit_mode": "clip",
        },
    )
    service = ControlService(
        config,
        device_manager=DeviceManager(
            "local",
            owner_id="manual-race-test",
            lock_dir=tmp_path / "locks",
            state_path=tmp_path / "state.json",
        ),
    )
    submitted = threading.Event()
    continue_submit = threading.Event()
    try:
        service.acquire_manual(caller_id="caller-a", session_id="session-a")
        service.set_manual_deadman(
            True,
            caller_id="caller-a",
            session_id="session-a",
        )
        arbiter = service._current_arbiter
        assert arbiter is not None
        original_submit = arbiter.submit_manual

        def paused_submit(action, *, wait=False, timeout_s=None, deadline_s=None):
            submitted.set()
            assert continue_submit.wait(2)
            return original_submit(
                action,
                wait=wait,
                timeout_s=timeout_s,
                deadline_s=deadline_s,
            )

        monkeypatch.setattr(arbiter, "submit_manual", paused_submit)
        action = {
            "timestamp_s": 1.0,
            "values": {
                "type": "joint_position",
                "joint_positions_deg": [1, 2, 3, 4, 5],
                "gripper_position": 10,
            },
        }
        submit_errors: list[BaseException] = []

        def submit_old_owner() -> None:
            try:
                service.submit_manual_action(
                    action,
                    caller_id="caller-a",
                    session_id="session-a",
                )
            except BaseException as error:  # pragma: no cover - diagnostic only
                submit_errors.append(error)

        submit_thread = threading.Thread(target=submit_old_owner)
        submit_thread.start()
        assert submitted.wait(2)

        release_thread = threading.Thread(
            target=service.release_manual,
            kwargs={"caller_id": "caller-a", "session_id": "session-a"},
        )
        release_thread.start()
        # The release cannot overtake the paused old-owner operation.  This
        # is the race that would otherwise let the action reach a new owner.
        release_thread.join(0.05)
        assert release_thread.is_alive()
        continue_submit.set()
        submit_thread.join(2)
        release_thread.join(2)
        assert not submit_thread.is_alive()
        assert not release_thread.is_alive()
        assert not submit_errors

        service.acquire_manual(caller_id="caller-b", session_id="session-b")
        with pytest.raises(ControlTaskRejected, match="owned by another caller"):
            service.set_manual_deadman(
                False,
                caller_id="caller-a",
                session_id="session-a",
            )
        service.release_manual(caller_id="caller-b", session_id="session-b")
    finally:
        continue_submit.set()
        service.close()


def test_sensor_identity_selects_generated_logical_resource() -> None:
    inputs = (
        SensorInput("front", "observation.images.front", "fake", {}),
        SensorInput("wrist", "observation.images.wrist", "fake", {}),
    )
    config = ControlServiceConfig.device_only(
        runtime_id="sensor-identity",
        bind="127.0.0.1",
        port=8126,
        robot_id="sensor-arm",
        robot_kind="simulated.joints",
        robot_options={"initial_joint_positions_deg": [0, 0, 0, 0, 0]},
        inputs=inputs,
        device_resources=(
            {
                "identity": "local:sensor:front-camera",
                "node": "local",
                "kind": "sensor",
                "value": "front-camera",
                "sensor_id": "front",
                "sensor_ids": ["front"],
            },
            {
                "identity": "local:sensor:wrist-camera",
                "node": "local",
                "kind": "sensor",
                "value": "wrist-camera",
                "sensor_id": "wrist",
                "sensor_ids": ["wrist"],
            },
        ),
    )
    assert _sensor_identity(config, inputs[0]).key == "local:sensor:front-camera"
    assert _sensor_identity(config, inputs[1]).key == "local:sensor:wrist-camera"


def test_sensor_identity_rejects_unmatched_alias_and_keeps_legacy_fallback() -> None:
    unmatched = SensorInput("wrist", "observation.images.wrist", "fake", {})
    generated_config = ControlServiceConfig.device_only(
        runtime_id="sensor-alias-mismatch",
        bind="127.0.0.1",
        port=8127,
        robot_id="sensor-arm",
        robot_kind="simulated.joints",
        robot_options={"initial_joint_positions_deg": [0, 0, 0, 0, 0]},
        inputs=(unmatched,),
        device_resources=(
            {
                "identity": "local:sensor:front-camera",
                "node": "local",
                "kind": "sensor",
                "value": "front-camera",
                "sensor_id": "front",
                "sensor_ids": ["front"],
            },
        ),
    )
    with pytest.raises(ControlServiceError, match="does not identify"):
        _sensor_identity(generated_config, unmatched)

    legacy = SensorInput("legacy", "observation.images.front", "fake", {})
    legacy_config = ControlServiceConfig.device_only(
        runtime_id="sensor-legacy",
        bind="127.0.0.1",
        port=8128,
        robot_id="sensor-arm",
        robot_kind="simulated.joints",
        robot_options={"initial_joint_positions_deg": [0, 0, 0, 0, 0]},
        inputs=(legacy,),
        device_resources=(
            {
                "identity": "local:sensor:legacy-camera",
                "node": "local",
                "kind": "sensor",
                "value": "legacy-camera",
            },
        ),
    )
    assert _sensor_identity(legacy_config, legacy).key == "local:sensor:legacy-camera"


def test_server_wires_persistent_application_routes_and_protects_legacy_control(
    tmp_path,
) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    config = ControlServiceConfig.device_only(
        runtime_id="http-fake",
        bind="127.0.0.1",
        port=port,
        robot_id="fake-http",
        robot_kind="simulated.joints",
        robot_options={
            "initial_joint_positions_deg": [0, 0, 0, 0, 0],
            "initial_gripper_position": 0,
            "step_limit_mode": "clip",
        },
    )
    manager = DeviceManager(
        "local",
        owner_id="http-control-api-test",
        lock_dir=tmp_path / "locks",
        state_path=tmp_path / "state.json",
    )
    service = ControlService(config, device_manager=manager)
    auth = AuthPolicy(
        {
            "token-a": {
                "role": "controller",
                "caller_id": "caller-a",
                "session_id": "session-a",
            },
            "token-b": {
                "role": "controller",
                "caller_id": "caller-b",
                "session_id": "session-b",
            },
        }
    )
    server = ControlHttpServer(
        service,
        state_dir=tmp_path / "control-state",
        auth_policy=auth,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        auth_headers: bool = True,
        *,
        token: str = "token-a",
        caller_id: str = "caller-a",
        session_id: str = "session-a",
    ):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        headers = {}
        if auth_headers:
            headers.update(
                {
                    "Authorization": f"Bearer {token}",
                    "X-EmbodiRun-Caller-ID": caller_id,
                    "X-EmbodiRun-Session-ID": session_id,
                }
            )
        encoded = None
        if body is not None:
            encoded = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload

    try:
        status, described = request("GET", "/v1/describe")
        assert status == 200
        assert described["application"]["execute"] is True
        assert described["capabilities"]["direct_execute"] is True
        assert described["binding"]["kind"]
        assert described["binding"]["maximum_chunk_steps"] is None
        assert described["binding"]["action_feature_names"] == []
        status, unauthenticated = request("GET", "/v1/control", auth_headers=False)
        assert status == 401
        assert unauthenticated["code"] == "authentication_required"
        action = {
            "timestamp_s": 1.0,
            "values": {
                "type": "joint_position",
                "joint_positions_deg": [1, 2, 3, 4, 5],
                "gripper_position": 20,
            },
        }
        status, executed = request(
            "POST",
            "/v1/execute",
            {"request_id": "http-direct", "action": action, "wait": True},
        )
        assert status == 200, executed
        assert executed["status"] == "completed"
        status, duplicate = request(
            "POST",
            "/v1/execute",
            {"request_id": "http-direct", "action": action, "wait": True},
        )
        assert status == 200
        assert duplicate["run_id"] == executed["run_id"]

        status, _ = request("POST", "/v1/control/manual/acquire")
        assert status == 200
        status, blocked = request(
            "POST",
            "/v1/control/manual/acquire",
            token="token-b",
            caller_id="caller-b",
            session_id="session-b",
        )
        assert status == 409
        assert "owned by another caller" in blocked["error"]
        status, _ = request("POST", "/v1/control/manual/release")
        assert status == 200
        status, _ = request(
            "POST",
            "/v1/control/manual/acquire",
            token="token-b",
            caller_id="caller-b",
            session_id="session-b",
        )
        assert status == 200
        status, _ = request(
            "POST",
            "/v1/control/manual/deadman",
            {"active": True},
            token="token-b",
            caller_id="caller-b",
            session_id="session-b",
        )
        assert status == 200
        status, late_release = request("POST", "/v1/control/manual/release")
        assert status == 409
        assert "owned by another caller" in late_release["error"]
        status, late_action = request(
            "POST",
            "/v1/control/manual/action",
            action,
        )
        assert status == 409
        assert "owned by another caller" in late_action["error"]
        status, snapshot = request(
            "GET",
            "/v1/control",
            token="token-b",
            caller_id="caller-b",
            session_id="session-b",
        )
        assert status == 200
        assert snapshot["authority"] == "manual"
        assert snapshot["deadman_active"] is True
        status, _ = request(
            "POST",
            "/v1/control/manual/release",
            token="token-b",
            caller_id="caller-b",
            session_id="session-b",
        )
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2.0)
        service.close()
    assert (tmp_path / "control-state" / "http-fake.jobs.sqlite").exists()


def test_explicit_empty_token_file_cannot_reenable_trusted_mode(tmp_path) -> None:
    token_file = tmp_path / "empty-tokens.json"
    token_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one bound principal") as error:
        _load_auth_policy(token_file)
    assert "{}" not in str(error.value)
