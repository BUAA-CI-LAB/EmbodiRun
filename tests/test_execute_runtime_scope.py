"""Execute must name the runtime it targets, matching observe and propose."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from embodirun.client import ControlClient
from embodirun.robots import RobotAction
from embodirun.services.control.application import (
    ApplicationInvalidRequest,
    ControlApplication,
)
from embodirun.services.control.arbitration import RobotControlArbiter
from embodirun.services.control.direct_execution import DirectExecutionError
from embodirun.services.control.http_api import ControlHTTPAPI
from embodirun.services.control.io import IOResult, IOStatus


class _Port:
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
        )

    def hold(self) -> IOResult:
        self.holds += 1
        return IOResult("stop", IOStatus.COMPLETED, stop_confirmed=True)

    def emergency_stop(self) -> IOResult:
        return self.hold()

    def close(self) -> None:
        return None


class _Service:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            runtime_id="xlerobot-base",
            runtime_profiles={"xlerobot-pi05": object()},
            robot_kind="simulated.policy_vector",
            inputs=None,
        )

    def describe(self) -> dict[str, Any]:
        return {"status": "ok", "robot_id": "fake-arm"}

    def observe(self, *, runtime_id: str | None, include_robot: bool) -> dict[str, Any]:
        return {"status": "ok", "observation_id": "obs-1", "runtime_id": runtime_id}


def _app() -> tuple[ControlApplication, RobotControlArbiter]:
    arbiter = RobotControlArbiter(_Port())
    app = ControlApplication(_Service(), arbiter_provider=lambda: arbiter)
    return app, arbiter


def test_execute_accepts_a_served_runtime_and_rejects_an_unknown_one() -> None:
    app, arbiter = _app()
    try:
        record = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="served",
            action={"timestamp_s": 0.0, "values": {"joint": 1.0}},
            runtime_id="xlerobot-pi05",
            wait=True,
        )
        assert record["status"] == "completed"
        assert record["parameters"]["runtime_id"] == "xlerobot-pi05"

        with pytest.raises(ApplicationInvalidRequest, match="not served"):
            app.execute(
                caller_id="caller",
                session_id="session",
                request_id="unknown",
                action={"timestamp_s": 0.0, "values": {"joint": 1.0}},
                runtime_id="some-other-owner",
                wait=True,
            )
    finally:
        app.close()
        arbiter.close(hold=False)


def test_execute_runtime_id_is_part_of_request_identity() -> None:
    app, arbiter = _app()
    try:
        app.execute(
            caller_id="caller",
            session_id="session",
            request_id="identity",
            action={"timestamp_s": 0.0, "values": {"joint": 1.0}},
            runtime_id="xlerobot-base",
            wait=True,
        )
        from embodirun.services.control.jobs import JobConflict

        with pytest.raises(JobConflict):
            app.execute(
                caller_id="caller",
                session_id="session",
                request_id="identity",
                action={"timestamp_s": 0.0, "values": {"joint": 1.0}},
                runtime_id="xlerobot-pi05",
                wait=True,
            )
    finally:
        app.close()
        arbiter.close(hold=False)


def test_execute_runtime_without_service_configuration_is_unsupported() -> None:
    arbiter = RobotControlArbiter(_Port())

    class _NoConfig:
        def describe(self) -> dict[str, Any]:
            return {"status": "ok"}

    app = ControlApplication(_NoConfig(), arbiter_provider=lambda: arbiter)
    try:
        from embodirun.services.control.application import ApplicationUnsupported

        with pytest.raises(ApplicationUnsupported):
            app.execute(
                caller_id="caller",
                session_id="session",
                request_id="no-config",
                action={"timestamp_s": 0.0, "values": {"joint": 1.0}},
                runtime_id="anything",
                wait=True,
            )
    finally:
        app.close()
        arbiter.close(hold=False)


def test_http_execute_body_carries_runtime_id() -> None:
    app, arbiter = _app()
    api = ControlHTTPAPI(app)
    try:
        response = api.dispatch(
            "POST",
            "/v1/execute",
            body={
                "request_id": "http-runtime",
                "action": {"timestamp_s": 0.0, "values": {"joint": 1.0}},
                "runtime_id": "xlerobot-pi05",
                "wait": True,
            },
        )
        assert response.status == 200
        assert response.payload["parameters"]["runtime_id"] == "xlerobot-pi05"
    finally:
        app.close()
        arbiter.close(hold=False)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status = 200
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def close(self) -> None:
        return None


def test_client_sends_runtime_id_and_omits_it_when_unknown() -> None:
    sent: list[dict[str, Any]] = []

    def opener(request: Any, timeout: float | None = None) -> _Response:
        sent.append(json.loads(request.data.decode()))
        return _Response({"status": "completed"})

    client = ControlClient(
        "http://127.0.0.1:1",
        caller_id="caller",
        session_id="session",
        opener=opener,
    )
    client.execute(
        {"timestamp_s": 0.0, "values": {"x.vel": 0.1, "theta.vel": 0.0}},
        request_id="scoped",
        runtime_id="xlerobot-base",
        wait=True,
    )
    assert sent[-1]["runtime_id"] == "xlerobot-base"

    client.execute(
        {"timestamp_s": 0.0, "values": {"x.vel": 0.1, "theta.vel": 0.0}},
        request_id="unscoped",
        wait=True,
    )
    assert sent[-1]["runtime_id"] is None


def test_direct_execution_still_reports_business_success_unknown() -> None:
    app, arbiter = _app()
    try:
        record = app.execute(
            caller_id="caller",
            session_id="session",
            request_id="direct-success",
            action={"timestamp_s": 0.0, "values": {"joint": 1.0}},
            runtime_id="xlerobot-base",
            wait=True,
        )
        assert record["result"]["business_success"] is None
        assert record["dispatch_status"] == "completed"
    finally:
        app.close()
        arbiter.close(hold=False)


def test_direct_execution_error_type_is_exported() -> None:
    assert issubclass(DirectExecutionError, RuntimeError)
