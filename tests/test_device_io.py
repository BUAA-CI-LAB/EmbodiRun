from __future__ import annotations

import threading
import time

from embodirun.services.control.io import IOStatus, RobotIOScheduler


def test_one_bus_serializes_read_and_action_transactions() -> None:
    scheduler = RobotIOScheduler("shared", max_pending_operations=4)
    entered = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def transaction(value: str) -> str:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        entered.set()
        release.wait(1.0)
        with state_lock:
            active -= 1
        return value

    action_result: list[object] = []
    action_thread = threading.Thread(
        target=lambda: action_result.append(scheduler.execute(lambda: transaction("action"), requested="action"))
    )
    action_thread.start()
    assert entered.wait(1.0)

    read_result: list[object] = []
    read_thread = threading.Thread(
        target=lambda: read_result.append(scheduler.read(lambda: transaction("read"), requested="observation"))
    )
    read_thread.start()
    time.sleep(0.02)
    with state_lock:
        assert active == 1
        assert maximum_active == 1

    release.set()
    action_thread.join(1.0)
    read_thread.join(1.0)
    assert action_result[0].status is IOStatus.COMPLETED
    assert read_result[0].status is IOStatus.COMPLETED
    assert maximum_active == 1
    scheduler.close()


def test_serialized_stop_reports_unknown_and_quarantines_blocked_read() -> None:
    scheduler = RobotIOScheduler("slow-read", stop_timeout_s=0.03)
    read_started = threading.Event()
    release_read = threading.Event()
    stop_called = threading.Event()

    def slow_read() -> None:
        read_started.set()
        release_read.wait(1.0)

    read_result: list[object] = []
    read_thread = threading.Thread(target=lambda: read_result.append(scheduler.read(slow_read, requested="state")))
    read_thread.start()
    assert read_started.wait(1.0)

    started = time.monotonic()
    stop_result = scheduler.stop(
        lambda: stop_called.set(),
        requested="hold",
        timeout_s=0.03,
    )
    elapsed = time.monotonic() - started
    assert stop_result.status is IOStatus.UNKNOWN
    assert stop_result.stop_confirmed is None
    assert stop_result.quarantined
    assert elapsed < 0.2
    assert not stop_called.is_set()

    # Repeated stops are coalesced/rejected while the first bounded request is
    # still pending; the queue cannot grow with telemetry or cleanup calls.
    duplicate = scheduler.stop(lambda: None, requested="duplicate", timeout_s=0.03)
    assert duplicate.status is IOStatus.REJECTED

    release_read.set()
    read_thread.join(1.0)
    assert read_result[0].status is IOStatus.COMPLETED
    assert stop_called.wait(1.0)
    snapshot = scheduler.snapshot()
    assert snapshot["quarantined"] is True
    assert snapshot["last_stop"].stop_confirmed is None
    assert not scheduler.recover(stop_confirmed=False)
    scheduler.close()


def test_failed_stop_preserves_quarantine_until_explicit_confirmation() -> None:
    scheduler = RobotIOScheduler("failed-stop", stop_timeout_s=0.1)

    def fail_stop() -> None:
        raise RuntimeError("driver stop failed")

    failed = scheduler.stop(fail_stop)
    assert failed.status is IOStatus.FAILED
    assert failed.quarantined is True
    assert scheduler.snapshot()["quarantined"] is True
    rejected = scheduler.execute(lambda: "motion")
    assert rejected.status is IOStatus.UNKNOWN

    confirmed = scheduler.stop(lambda: True, timeout_s=0.1)
    assert confirmed.status is IOStatus.COMPLETED
    assert confirmed.stop_confirmed is True
    assert scheduler.snapshot()["quarantined"] is True
    assert scheduler.recover(stop_confirmed=True)
    assert scheduler.snapshot()["quarantined"] is False
    scheduler.close()


def test_timed_out_active_execute_quarantines_queued_motion() -> None:
    scheduler = RobotIOScheduler("slow-write", stop_timeout_s=0.1)
    started = threading.Event()
    release = threading.Event()

    def slow_execute() -> None:
        started.set()
        release.wait(1.0)

    result_holder: list[object] = []
    execute_thread = threading.Thread(
        target=lambda: result_holder.append(scheduler.execute(slow_execute, requested="motion", timeout_s=0.03))
    )
    execute_thread.start()
    assert started.wait(1.0)
    execute_thread.join(1.0)
    assert result_holder[0].status is IOStatus.UNKNOWN
    assert result_holder[0].quarantined is True
    assert scheduler.execute(lambda: None).status is IOStatus.UNKNOWN

    release.set()
    confirmed = scheduler.stop(lambda: True, timeout_s=0.2)
    assert confirmed.stop_confirmed is True
    assert scheduler.snapshot()["quarantined"] is True
    assert scheduler.recover(stop_confirmed=True)
    scheduler.close()


def test_preemptive_stop_blocks_new_scheduler_work_until_callback_returns() -> None:
    scheduler = RobotIOScheduler(
        "fr3",
        stop_timeout_s=0.2,
        allow_preemptive_stop=True,
    )
    stop_started = threading.Event()
    release_stop = threading.Event()
    stop_result: list[object] = []

    def slow_stop() -> None:
        stop_started.set()
        release_stop.wait(1.0)

    stop_thread = threading.Thread(target=lambda: stop_result.append(scheduler.stop(slow_stop, requested="estop")))
    stop_thread.start()
    assert stop_started.wait(1.0)
    motion = scheduler.execute(lambda: "must-not-run", requested="motion")
    assert motion.status is IOStatus.UNKNOWN
    release_stop.set()
    stop_thread.join(1.0)
    assert stop_result[0].status is IOStatus.COMPLETED
    scheduler.close()


def test_independent_bus_schedulers_can_progress_separately() -> None:
    left = RobotIOScheduler("left")
    right = RobotIOScheduler("right")
    left_started = threading.Event()
    right_started = threading.Event()
    release = threading.Event()

    def left_call() -> None:
        left_started.set()
        release.wait(1.0)

    def right_call() -> None:
        right_started.set()
        release.wait(1.0)

    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(left.execute(left_call))),
        threading.Thread(target=lambda: results.append(right.execute(right_call))),
    ]
    for thread in threads:
        thread.start()
    assert left_started.wait(1.0)
    assert right_started.wait(1.0)
    release.set()
    for thread in threads:
        thread.join(1.0)
    assert all(result.status is IOStatus.COMPLETED for result in results)
    left.close()
    right.close()
