from __future__ import annotations

import sqlite3
import threading

import pytest

from embodirun.robots import RobotAction, RobotObservation
from embodirun.services.control.arbitration import (
    CommandStatus,
    RobotAdapterCommandPort,
    RobotControlArbiter,
)
from embodirun.services.control.contracts import TaskRequest, TaskResult
from embodirun.services.control.io import IOResult, IOStatus, IOUnknownError, RobotIOScheduler
from embodirun.services.control.job_store import SQLiteJobStore
from embodirun.services.control.jobs import (
    JobCancelledError,
    JobCapacityExceeded,
    JobConflict,
    JobOwnershipError,
    JobRegistry,
    JobStatus,
    JobTimeout,
    JobUnknownError,
    PhysicalStatus,
)


def test_concurrent_duplicate_accept_is_atomic_and_executes_once() -> None:
    registry = JobRegistry(max_events=16)
    started = threading.Event()
    release = threading.Event()
    call_count = 0
    count_lock = threading.Lock()

    def execute(cancel_event: threading.Event) -> dict[str, int]:
        nonlocal call_count
        with count_lock:
            call_count += 1
        started.set()
        release.wait(1.0)
        return {"actions": 2}

    handles: list[object] = []
    errors: list[BaseException] = []

    def submit_one(index: int) -> None:
        try:
            handles.append(
                registry.submit(
                    "caller",
                    "session",
                    "request-1",
                    {"b": [2, 3], "a": 1},
                    execute,
                    wait=False,
                )
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=submit_one, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    assert started.wait(1.0)
    release.set()
    for thread in threads:
        thread.join(1.0)
    assert not errors
    assert len(handles) == 8
    assert len({handle.run_id for handle in handles}) == 1
    assert handles[0].wait(timeout_s=1.0) == {"actions": 2}
    assert call_count == 1
    registry.close()


def test_same_request_id_different_parameters_conflicts_without_reexecution() -> None:
    registry = JobRegistry()
    handle = registry.submit("caller", "session", "request-1", {"x": 1}, lambda event: {"ok": 1})
    assert handle.wait(timeout_s=1.0) == {"ok": 1}
    with pytest.raises(JobConflict):
        registry.submit("caller", "session", "request-1", {"x": 2}, lambda event: {"ok": 2})
    registry.close()


def test_sync_timeout_keeps_original_request_inspectable() -> None:
    registry = JobRegistry()
    started = threading.Event()
    release = threading.Event()

    def execute(cancel_event: threading.Event) -> dict[str, str]:
        started.set()
        release.wait(1.0)
        return {"state": "done"}

    handle = registry.submit("caller", "session", "request-1", {}, execute)
    assert started.wait(1.0)
    with pytest.raises(JobTimeout) as timeout:
        registry.submit(
            "caller",
            "session",
            "request-1",
            {},
            lambda event: {"should": "not-run"},
            wait=True,
            timeout_s=0.01,
        )
    assert timeout.value.record.status is JobStatus.RUNNING
    assert handle.inspect().request_id == "request-1"
    release.set()
    assert handle.wait(timeout_s=1.0) == {"state": "done"}
    registry.close()


def test_full_history_rejects_new_identity_without_deleting_old_id() -> None:
    registry = JobRegistry(max_jobs=1)
    first = registry.submit("caller", "session", "request-1", {}, lambda event: {"ok": 1})
    assert first.wait(timeout_s=1.0) == {"ok": 1}
    with pytest.raises(JobCapacityExceeded):
        registry.submit("caller", "session", "request-2", {}, lambda event: {"ok": 2})
    assert registry.inspect("caller", "session", "request-1").status is JobStatus.COMPLETED
    registry.close()


def test_restart_marks_unfinished_job_unknown_and_never_replays(tmp_path) -> None:
    database = tmp_path / "jobs.sqlite"
    registry = JobRegistry(database)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def execute(cancel_event: threading.Event) -> dict[str, int]:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(1.0)
        return {"calls": calls}

    registry.submit("caller", "session", "request-1", {"x": 1}, execute)
    assert started.wait(1.0)
    assert registry.close(wait_s=0.01) is False
    release.set()
    assert registry.close(wait_s=1.0) is True
    reopened = JobRegistry(database)
    record = reopened.inspect("caller", "session", "request-1")
    assert record.status is JobStatus.UNKNOWN
    recovered_handle = reopened.submit("caller", "session", "request-1", {"x": 1}, execute)
    with pytest.raises(JobUnknownError):
        recovered_handle.wait(timeout_s=0.1)
    assert recovered_handle.run_id == record.run_id
    assert calls == 1
    reopened.close()


def test_live_database_owner_is_not_mistaken_for_a_crashed_registry(tmp_path) -> None:
    database = tmp_path / "live.sqlite"
    first = JobRegistry(database)
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            JobRegistry(database)
    finally:
        first.close()
    reopened = JobRegistry(database)
    reopened.close()


def test_job_store_releases_sidecar_lock_when_database_startup_fails(tmp_path) -> None:
    database = tmp_path / "startup-failure.sqlite"
    database.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        SQLiteJobStore(database)
    database.rmdir()
    store = SQLiteJobStore(database)
    store.close()


def test_close_keeps_database_lock_until_active_worker_drains(tmp_path) -> None:
    database = tmp_path / "close.sqlite"
    registry = JobRegistry(database)
    started = threading.Event()
    release = threading.Event()

    def execute(cancel_event: threading.Event) -> dict[str, bool]:
        started.set()
        release.wait(1.0)
        return {"done": True}

    registry.submit("caller", "session", "request-1", {}, execute)
    assert started.wait(1.0)
    assert registry.close(wait_s=0.01) is False
    with pytest.raises(RuntimeError, match="already owned"):
        JobRegistry(database)
    release.set()
    assert registry.close(wait_s=1.0) is True
    reopened = JobRegistry(database)
    try:
        assert reopened.inspect("caller", "session", "request-1").status is JobStatus.UNKNOWN
    finally:
        reopened.close()


def test_cancel_is_scoped_to_old_job_token_and_reports_physical_uncertainty() -> None:
    registry = JobRegistry()
    old_started = threading.Event()
    old_release = threading.Event()
    new_started = threading.Event()
    cancel_called = threading.Event()

    def old_execute(cancel_event: threading.Event) -> dict[str, str]:
        old_started.set()
        cancel_event.wait(1.0)
        old_release.set()
        return {"old": "finished"}

    def old_cancel(cancel_event: threading.Event) -> dict[str, bool]:
        cancel_called.set()
        cancel_event.set()
        return {"stop_confirmed": False}

    old = registry.submit(
        "caller",
        "session",
        "old-request",
        {"action": "old"},
        old_execute,
        old_cancel,
    )
    assert old_started.wait(1.0)
    old.cancel()
    assert cancel_called.wait(1.0)

    def new_execute(cancel_event: threading.Event) -> dict[str, str]:
        new_started.set()
        return {"new": "finished"}

    new = registry.submit(
        "caller",
        "session",
        "new-request",
        {"action": "new"},
        new_execute,
    )
    assert new.wait(timeout_s=1.0) == {"new": "finished"}
    assert new_started.is_set()
    old_release.set()
    with pytest.raises(JobCancelledError):
        old.wait(timeout_s=1.0)
    record = old.inspect()
    assert record.status is JobStatus.CANCELLED
    assert record.physical_status is PhysicalStatus.STOP_UNCONFIRMED
    registry.close()


def test_inspect_and_cancel_require_caller_session_ownership() -> None:
    registry = JobRegistry()
    handle = registry.submit("caller", "session", "request-1", {}, lambda event: {"ok": 1})
    assert handle.wait(timeout_s=1.0) == {"ok": 1}
    with pytest.raises(JobOwnershipError):
        registry.inspect("other", "session", "request-1")
    with pytest.raises(JobOwnershipError):
        registry.cancel("caller", "other-session", "request-1")
    registry.close()


def test_stage_history_is_bounded_and_task_result_wait_is_compatible() -> None:
    registry = JobRegistry(max_events=3)
    request = TaskRequest(
        request_id="request-1",
        runtime_id="runtime",
        prompt="move",
        chunk_steps=1,
        max_steps=1,
        control_hz=5.0,
        inference_timeout_s=1.0,
    )
    result = registry.submit_task(
        request,
        caller_id="caller",
        session_id="session",
        execute=lambda event: TaskResult("request-1", "runtime", 1),
        wait=True,
        timeout_s=1.0,
    )
    assert result == TaskResult("request-1", "runtime", 1)
    registry.record_stage("caller", "session", "request-1", "export", 0.01)
    registry.record_stage("caller", "session", "request-1", "extra", 0.02)
    record = registry.inspect("caller", "session", "request-1")
    assert len(record.events) <= 3
    assert record.events_truncated
    registry.close()


def test_bounded_scheduler_close_reports_unknown_and_drains_pending_stop() -> None:
    scheduler = RobotIOScheduler("close-bus", stop_timeout_s=0.02)
    read_started = threading.Event()
    release_read = threading.Event()
    stop_called = threading.Event()

    def slow_read() -> None:
        read_started.set()
        release_read.wait(1.0)

    reader = threading.Thread(target=lambda: scheduler.read(slow_read))
    reader.start()
    assert read_started.wait(1.0)
    assert scheduler.stop(lambda: stop_called.set(), timeout_s=0.01).status is IOStatus.UNKNOWN

    closing = scheduler.close(wait_s=0.01)
    assert closing.status is IOStatus.UNKNOWN
    assert not stop_called.is_set()
    release_read.set()
    assert stop_called.wait(1.0)
    reader.join(1.0)
    assert scheduler.close(wait_s=0.2).status is IOStatus.COMPLETED


def test_queued_execute_timeout_is_not_reported_as_executed() -> None:
    class Robot:
        robot_id = "queued-timeout"

        def __init__(self) -> None:
            self.executed: list[RobotAction] = []

        def observe(self) -> RobotObservation:
            return RobotObservation(0.0, {})

        def execute(self, action: RobotAction) -> None:
            self.executed.append(action)

        def stop(self) -> None:
            return None

    scheduler = RobotIOScheduler("queued-timeout-bus")
    read_started = threading.Event()
    release_read = threading.Event()

    def slow_read() -> None:
        read_started.set()
        release_read.wait(1.0)

    reader = threading.Thread(target=lambda: scheduler.read(slow_read))
    reader.start()
    assert read_started.wait(1.0)
    robot = Robot()
    arbiter = RobotControlArbiter(
        RobotAdapterCommandPort(
            robot,
            io_scheduler=scheduler,
            io_wait_timeout_s=0.02,
        )
    )
    try:
        ticket = arbiter.submit_model(
            RobotAction(1.0, {"joint": 1.0}),
            wait=True,
            timeout_s=1.0,
        )
        assert ticket.status is CommandStatus.CANCELLED
        assert ticket.result is not None
        assert ticket.result.status is IOStatus.CANCELLED
        assert robot.executed == []
    finally:
        release_read.set()
        reader.join(1.0)
        try:
            arbiter.close()
        finally:
            scheduler.close(wait_s=0.2)


def test_arbiter_close_propagates_pending_stop_instead_of_releasing_owner() -> None:
    class PendingStopPort:
        robot_id = "pending-stop"

        def __init__(self) -> None:
            self.closed = False

        def execute(self, action: RobotAction, cancel_event: threading.Event) -> None:
            return None

        def hold(self) -> IOResult:
            return IOResult(
                operation="stop",
                status=IOStatus.REJECTED,
                requested="hold",
                quarantined=True,
            )

        def close(self) -> None:
            self.closed = True

    port = PendingStopPort()
    arbiter = RobotControlArbiter(port)
    with pytest.raises(IOUnknownError):
        arbiter.close()
    assert port.closed


@pytest.mark.parametrize("io_status", [IOStatus.UNKNOWN, IOStatus.FAILED, IOStatus.REJECTED])
def test_generic_port_io_failure_is_not_reported_as_executed(io_status: IOStatus) -> None:
    class GenericPort:
        robot_id = "generic-result"

        def execute(self, action: RobotAction, cancel_event: threading.Event) -> IOResult:
            return IOResult(
                operation="execute",
                status=io_status,
                requested=action,
                error=RuntimeError("driver result failed") if io_status is IOStatus.FAILED else None,
            )

        def emergency_stop(self) -> None:
            return None

        def hold(self) -> None:
            return None

    arbiter = RobotControlArbiter(GenericPort())
    try:
        ticket = arbiter.submit_model(
            RobotAction(1.0, {"joint": 1.0}),
            wait=True,
            timeout_s=1.0,
        )
        assert ticket.status is CommandStatus.FAILED
        assert ticket.result is not None
        assert ticket.result.status is io_status
    finally:
        arbiter.close(hold=False)
