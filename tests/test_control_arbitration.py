from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from embodirun.robots import RobotAction, RobotObservation
from embodirun.services.control.arbitration import (
    ArbiterCommandSink,
    AuthorityState,
    CommandCancelled,
    CommandRejected,
    CommandStatus,
    EmergencyStopPolicy,
    RobotAdapterCommandPort,
    RobotControlArbiter,
)
from embodirun.services.control.io import IOUnknownError
from embodirun.services.control.runtime import ControlRuntime


def action(value: float) -> RobotAction:
    return RobotAction(timestamp_s=value, values={"value": value})


class FakeRobotPort:
    def __init__(self, robot_id: str) -> None:
        self.robot_id = robot_id
        self.executed: list[RobotAction] = []
        self.hold_calls = 0
        self.emergency_stop_calls = 0

    def execute(self, command: RobotAction, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return
        self.executed.append(command)

    def hold(self) -> None:
        self.hold_calls += 1

    def emergency_stop(self) -> None:
        self.emergency_stop_calls += 1


class BlockingRobotPort(FakeRobotPort):
    def __init__(self, robot_id: str) -> None:
        super().__init__(robot_id)
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_seen = threading.Event()

    def execute(self, command: RobotAction, cancel_event: threading.Event) -> None:
        self.executed.append(command)
        self.started.set()
        while not self.release.is_set():
            if cancel_event.wait(0.01):
                self.cancel_seen.set()
                return


class FakeRobot:
    robot_id = "arm"

    def __init__(self) -> None:
        self.executed: list[RobotAction] = []
        self.stop_calls = 0

    def connect(self) -> None:
        return None

    def observe(self) -> RobotObservation:
        return RobotObservation(0.0, {})

    def execute(self, command: RobotAction) -> None:
        self.executed.append(command)

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        return None


class BlockingStopRobot(FakeRobot):
    def __init__(self) -> None:
        super().__init__()
        self.execute_started = threading.Event()
        self.execute_release = threading.Event()
        self.execute_finished = threading.Event()
        self.stopped_during_execute = threading.Event()
        self.overlapping_stops = threading.Event()
        self._stop_lock = threading.Lock()
        self._stops_in_flight = 0

    def execute(self, command: RobotAction) -> None:
        self.execute_started.set()
        self.execute_release.wait(2.0)
        self.execute_finished.set()

    def stop(self) -> None:
        with self._stop_lock:
            self._stops_in_flight += 1
            if self._stops_in_flight > 1:
                self.overlapping_stops.set()
        self.stop_calls += 1
        if not self.execute_finished.is_set():
            self.stopped_during_execute.set()
        with self._stop_lock:
            self._stops_in_flight -= 1


class BlockingHoldRobot(FakeRobot):
    def __init__(self) -> None:
        super().__init__()
        self.hold_started = threading.Event()
        self.hold_release = threading.Event()
        self.fail_hold = False

    def stop(self) -> None:
        self.hold_started.set()
        self.hold_release.wait(1.0)
        if self.fail_hold:
            raise RuntimeError("blocked hold failed")
        self.stop_calls += 1


class Mapper:
    policy_action_space = "fake.action.v1"

    def map_observation(self, observation, **kwargs):
        return SimpleNamespace(observation=observation, **kwargs)

    def map_result(self, result):
        return result.actions


class Client:
    def __init__(self, actions) -> None:
        self.actions = tuple(actions)
        self.closed_sessions: list[str] = []

    def open_session(self, *, robot_id: str, action_space: str):
        return SimpleNamespace(session_id=f"{robot_id}:{action_space}")

    def step(self, request):
        return SimpleNamespace(actions=self.actions)

    def reset(self, session_id: str, *, request_id: str):
        return SimpleNamespace(session_id=f"{session_id}:{request_id}")

    def close(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


def test_runtime_executes_model_actions_through_observable_arbiter_sink() -> None:
    robot = FakeRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    runtime = ControlRuntime(
        robot,
        Client([action(1.0)]),
        instruction="move",
        mapper=Mapper(),
        chunk_steps=1,
        command_sink=ArbiterCommandSink(arbiter),
    )
    try:
        runtime.step(())
        assert [item.values["value"] for item in robot.executed] == [1.0]
    finally:
        runtime.close()
        arbiter.close()


def test_successful_runtime_cleanup_holds_without_cancelling_task_token() -> None:
    """A completed registry task must not become cancelled during close."""

    robot = FakeRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    token = arbiter.begin_model_task()
    runtime = ControlRuntime(
        robot,
        Client([action(1.0)]),
        instruction="move",
        mapper=Mapper(),
        chunk_steps=1,
        command_sink=ArbiterCommandSink(arbiter, token),
        cancel_event=token,
    )
    try:
        runtime.step(())
        runtime.close()

        assert not token.is_set()
        # Completion detaches the old token, so a late submission from that
        # runtime cannot acquire the next automatic owner accidentally.
        with pytest.raises(CommandCancelled, match="no longer current"):
            arbiter.submit_model(action(2.0), task_cancel=token)
        new_token = arbiter.begin_model_task()
        ticket = arbiter.submit_model(action(3.0), task_cancel=new_token)
        assert ticket.status is CommandStatus.EXECUTED
    finally:
        arbiter.close()


def test_stale_success_cleanup_cannot_hold_a_new_automatic_owner() -> None:
    robot = FakeRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    try:
        old_token = arbiter.begin_model_task()
        old_sink = ArbiterCommandSink(arbiter, old_token)
        new_token = arbiter.begin_model_task()
        stop_calls = robot.stop_calls

        old_sink.finish()

        assert not new_token.is_set()
        assert robot.stop_calls == stop_calls
        ticket = arbiter.submit_model(action(4.0), task_cancel=new_token)
        assert ticket.status is CommandStatus.EXECUTED
    finally:
        arbiter.close()


def test_success_cleanup_after_manual_takeover_does_not_hold_manual_owner() -> None:
    robot = FakeRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    try:
        old_token = arbiter.begin_model_task()
        old_sink = ArbiterCommandSink(arbiter, old_token)
        arbiter.acquire_manual()
        stop_calls = robot.stop_calls

        old_sink.finish()

        assert arbiter.snapshot()["authority"] == AuthorityState.MANUAL.value
        assert robot.stop_calls == stop_calls

        # An unscoped legacy sink is equally unable to finish a manual owner.
        ArbiterCommandSink(arbiter).finish()
        assert robot.stop_calls == stop_calls
    finally:
        arbiter.close(hold=False)


def test_manual_takeover_joins_a_success_cleanup_hold() -> None:
    robot = BlockingHoldRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot, stop_timeout_s=0.8))
    token = arbiter.begin_model_task()
    sink = ArbiterCommandSink(arbiter, token)
    cleanup_done = threading.Event()
    takeover_started = threading.Event()
    takeover_done = threading.Event()
    errors: list[BaseException] = []

    def finish() -> None:
        try:
            sink.finish()
        except BaseException as error:  # pragma: no cover - assertion below
            errors.append(error)
        finally:
            cleanup_done.set()

    def takeover() -> None:
        takeover_started.set()
        try:
            arbiter.acquire_manual()
        except BaseException as error:  # pragma: no cover - assertion below
            errors.append(error)
        finally:
            takeover_done.set()

    finish_thread = threading.Thread(target=finish)
    finish_thread.start()
    try:
        assert robot.hold_started.wait(1.0)
        takeover_thread = threading.Thread(target=takeover)
        takeover_thread.start()
        assert takeover_started.wait(1.0)
        # acquire_manual sets the old task token while holding the arbiter
        # lock, before it joins the blocked physical hold.  This proves the
        # takeover has reached the overlap window without relying on sleep.
        assert token.wait(1.0)
        assert not takeover_done.is_set()

        robot.hold_release.set()
        assert cleanup_done.wait(1.0)
        assert takeover_done.wait(1.0)
        finish_thread.join(1.0)
        takeover_thread.join(1.0)

        assert errors == []
        assert arbiter.snapshot()["authority"] == AuthorityState.MANUAL.value
        assert robot.stop_calls == 1
        assert token.is_set()  # takeover cancels the old token
    finally:
        robot.hold_release.set()
        arbiter.close(hold=False)


def test_manual_takeover_receives_existing_hold_failure() -> None:
    robot = BlockingHoldRobot()
    robot.fail_hold = True
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot, stop_timeout_s=0.8))
    token = arbiter.begin_model_task()
    sink = ArbiterCommandSink(arbiter, token)
    cleanup_done = threading.Event()
    takeover_started = threading.Event()
    takeover_done = threading.Event()
    errors: list[BaseException] = []

    def finish() -> None:
        try:
            sink.finish()
        except BaseException as error:
            errors.append(error)
        finally:
            cleanup_done.set()

    def takeover() -> None:
        takeover_started.set()
        try:
            arbiter.acquire_manual()
        except BaseException as error:
            errors.append(error)
        finally:
            takeover_done.set()

    finish_thread = threading.Thread(target=finish)
    finish_thread.start()
    try:
        assert robot.hold_started.wait(1.0)
        takeover_thread = threading.Thread(target=takeover)
        takeover_thread.start()
        assert takeover_started.wait(1.0)
        assert token.wait(1.0)
        assert not takeover_done.is_set()
        robot.hold_release.set()
        assert cleanup_done.wait(1.0)
        assert takeover_done.wait(1.0)
        finish_thread.join(1.0)
        takeover_thread.join(1.0)

        assert len(errors) == 2
        assert all("blocked hold failed" in str(error) for error in errors)
        assert arbiter.snapshot()["authority"] == AuthorityState.ESTOP_LATCHED.value
    finally:
        robot.hold_release.set()
        arbiter.close(hold=False)


def test_command_ticket_reuses_action_id_for_queued_and_terminal_events() -> None:
    robot = FakeRobot()
    events: list[dict[str, object]] = []
    arbiter = RobotControlArbiter(
        RobotAdapterCommandPort(robot),
        event_callback=events.append,
    )
    try:
        ticket = arbiter.submit_model(action(5.0), wait=True)
        assert ticket.status is CommandStatus.EXECUTED
        lifecycle = [event for event in events if event["stage"] in {"queued", "executed"}]
        assert {event["stage"] for event in lifecycle} == {"queued", "executed"}
        assert len({event["action_id"] for event in lifecycle}) == 1
    finally:
        arbiter.close(hold=False)


def test_runtime_surfaces_robot_execute_failure_from_arbiter_ticket() -> None:
    class FailingRobot(FakeRobot):
        def execute(self, command: RobotAction) -> None:
            raise RuntimeError("execute failed")

    robot = FailingRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    runtime = ControlRuntime(
        robot,
        Client([action(1.0)]),
        instruction="move",
        mapper=Mapper(),
        chunk_steps=1,
        command_sink=ArbiterCommandSink(arbiter),
    )
    try:
        with pytest.raises(RuntimeError, match="execute failed"):
            runtime.step(())
    finally:
        runtime.close()
        with contextlib.suppress(IOUnknownError):
            # A failed action can leave the scheduler's prior stop unresolved;
            # the caller must observe that uncertainty rather than claim a
            # clean close.
            arbiter.close()


def test_estop_cancels_active_clears_pending_latches_and_requires_reset() -> None:
    port = BlockingRobotPort("arm")
    arbiter = RobotControlArbiter(port, model_queue_capacity=2)
    try:
        active = arbiter.submit_model(action(1.0), wait=False)
        assert port.started.wait(1.0)
        pending = arbiter.submit_model(action(2.0), wait=False)

        arbiter.emergency_stop()

        assert port.cancel_seen.wait(1.0)
        assert port.emergency_stop_calls == 1
        assert active.wait(1.0).status is CommandStatus.CANCELLED
        assert pending.wait(1.0).status is CommandStatus.CLEARED
        assert arbiter.snapshot()["authority"] == AuthorityState.ESTOP_LATCHED.value
        with pytest.raises(CommandRejected, match="emergency stop"):
            arbiter.submit_model(action(3.0), wait=False)
    finally:
        port.release.set()
        arbiter.close()


def test_manual_latest_wins_and_deadman_timeout_holds_manual_authority() -> None:
    port = BlockingRobotPort("arm")
    arbiter = RobotControlArbiter(port, manual_deadman_timeout_s=0.03)
    try:
        arbiter.acquire_manual()
        arbiter.set_deadman(True)
        active = arbiter.submit_manual(action(1.0), wait=False)
        assert port.started.wait(1.0)

        replaced = arbiter.submit_manual(action(2.0), wait=False)
        latest = arbiter.submit_manual(action(3.0), wait=False)
        assert replaced.wait(1.0).status is CommandStatus.REPLACED

        port.release.set()
        assert active.wait(1.0).status is CommandStatus.EXECUTED
        assert latest.wait(1.0).status is CommandStatus.EXECUTED
        deadline = time.monotonic() + 1.0
        while arbiter.snapshot()["deadman_active"] and time.monotonic() < deadline:
            time.sleep(0.005)
        snapshot = arbiter.snapshot()
        assert snapshot["authority"] == AuthorityState.MANUAL.value
        assert snapshot["deadman_active"] is False
        assert [item.values["value"] for item in port.executed] == [1.0, 3.0]
    finally:
        port.release.set()
        arbiter.close()


def test_preemptive_estop_can_dispatch_while_execute_is_blocked() -> None:
    robot = BlockingStopRobot()
    port = RobotAdapterCommandPort(
        robot,
        emergency_stop_policy=EmergencyStopPolicy.PREEMPTIVE,
    )
    execute_thread = threading.Thread(
        target=port.execute,
        args=(action(1.0), threading.Event()),
    )
    execute_thread.start()
    assert robot.execute_started.wait(1.0)

    estop_thread = threading.Thread(target=port.emergency_stop)
    estop_thread.start()
    try:
        assert robot.stopped_during_execute.wait(1.0)
    finally:
        robot.execute_release.set()
    execute_thread.join(1.0)
    estop_thread.join(1.0)

    assert not execute_thread.is_alive()
    assert not estop_thread.is_alive()
    assert not robot.overlapping_stops.is_set()


def test_estop_is_scoped_to_one_robot() -> None:
    first = RobotControlArbiter(FakeRobotPort("arm-a"))
    second = RobotControlArbiter(FakeRobotPort("arm-b"))
    try:
        first.emergency_stop()
        with pytest.raises(CommandRejected, match="latched"):
            first.submit_model(action(1))
        ticket = second.submit_model(action(2), wait=True, timeout_s=1)
        assert ticket.status is CommandStatus.EXECUTED
        assert second.snapshot()["authority"] == "model"
    finally:
        first.close()
        second.close()


def test_failed_hold_clears_pending_work_and_keeps_estop_latched() -> None:
    class FailingPort(FakeRobotPort):
        def hold(self):
            raise RuntimeError("hold unavailable")

    arbiter = RobotControlArbiter(FailingPort("arm"))
    try:
        with pytest.raises(RuntimeError, match="hold unavailable"):
            arbiter.acquire_manual()
        state = arbiter.snapshot()
        assert state["authority"] == "estop_latched"
        assert state["pending_model"] == 0
        assert state["last_error"] == "hold unavailable"
        with pytest.raises(RuntimeError, match="hold unavailable"):
            arbiter.reset_emergency_stop()
        assert arbiter.snapshot()["authority"] == "estop_latched"
    finally:
        with pytest.raises(RuntimeError, match="hold unavailable"):
            arbiter.close()
