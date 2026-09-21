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
    CommandRejected,
    CommandSource,
    CommandStatus,
    RobotAdapterCommandPort,
    RobotControlArbiter,
)
from embodirun.services.control.io import IOStatus, RobotIOScheduler
from embodirun.services.control.runtime import ControlRuntime


def action(value: float) -> RobotAction:
    return RobotAction(timestamp_s=value, values={"joint": value})


class RecordingRobot:
    robot_id = "direct-arm"

    def __init__(self) -> None:
        self.executed: list[RobotAction] = []
        self.observed = 0
        self.stopped = 0

    def observe(self) -> RobotObservation:
        self.observed += 1
        return RobotObservation(float(self.observed), {"joint": self.observed})

    def execute(self, command: RobotAction) -> None:
        self.executed.append(command)

    def stop(self) -> None:
        self.stopped += 1


class Mapper:
    policy_action_space = "direct.action.v1"

    def map_observation(self, observation, **kwargs):
        return SimpleNamespace(observation=observation, **kwargs)

    def map_result(self, result):
        return result.actions


class Client:
    def open_session(self, *, robot_id: str, action_space: str):
        return SimpleNamespace(session_id=f"{robot_id}:{action_space}")

    def step(self, request):
        return SimpleNamespace(actions=(action(1.0), action(2.0)))

    def reset(self, session_id: str, *, request_id: str):
        return SimpleNamespace(session_id=session_id)

    def close(self, session_id: str) -> None:
        return None


def test_direct_port_preserves_action_values_and_routes_observation_on_same_bus() -> None:
    robot = RecordingRobot()
    scheduler = RobotIOScheduler("arm-bus")
    port = RobotAdapterCommandPort(robot, io_scheduler=scheduler)
    try:
        first = action(4.0)
        result = port.execute(first, threading.Event(), source=CommandSource.REPLAY)
        observation = port.observe()
        hold = port.hold()
        assert result.status is IOStatus.COMPLETED
        assert result.requested["source"] == CommandSource.REPLAY.value
        assert robot.executed == [first]
        assert observation.values["joint"] == 1
        assert hold.status is IOStatus.COMPLETED
        assert robot.stopped == 1
    finally:
        scheduler.close()


def test_old_automatic_sink_cleanup_cannot_cancel_new_task() -> None:
    robot = RecordingRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    try:
        old_token = arbiter.begin_model_task()
        old_sink = ArbiterCommandSink(arbiter, old_token)
        new_token = arbiter.begin_automatic_task()
        assert old_token.is_set()
        assert not new_token.is_set()

        old_sink.stop()
        assert not new_token.is_set()
        ticket = arbiter.submit_agent(action(3.0), task_cancel=new_token)
        assert ticket.status is CommandStatus.EXECUTED
        assert robot.executed[-1].values["joint"] == 3.0
    finally:
        arbiter.close()


def test_agent_and_replay_are_automatic_and_manual_keeps_priority() -> None:
    robot = RecordingRobot()
    arbiter = RobotControlArbiter(RobotAdapterCommandPort(robot))
    try:
        agent_ticket = arbiter.submit_agent(action(1.0))
        replay_ticket = arbiter.submit_replay(action(2.0))
        assert agent_ticket.envelope.source is CommandSource.AGENT
        assert replay_ticket.envelope.source is CommandSource.REPLAY
        assert agent_ticket.status is CommandStatus.EXECUTED
        assert replay_ticket.status is CommandStatus.EXECUTED

        arbiter.acquire_manual()
        arbiter.set_deadman(True)
        with pytest.raises(CommandRejected, match="manual control owns"):
            arbiter.submit_agent(action(3.0), wait=False)
        manual = arbiter.submit_manual(action(4.0), wait=True)
        assert manual.status is CommandStatus.EXECUTED
        assert arbiter.snapshot()["authority"] == AuthorityState.MANUAL.value
    finally:
        arbiter.close()


def test_runtime_can_inject_shared_observation_source_without_changing_actions() -> None:
    robot = RecordingRobot()
    observation_calls: list[int] = []

    def observation_source() -> RobotObservation:
        observation_calls.append(1)
        return RobotObservation(9.0, {"joint": 9})

    runtime = ControlRuntime(
        robot,
        Client(),
        instruction="move",
        mapper=Mapper(),
        chunk_steps=2,
        observation_source=observation_source,
    )
    try:
        runtime.step(())
        assert len(observation_calls) == 1
        assert [item.values["joint"] for item in robot.executed] == [1.0, 2.0]
    finally:
        runtime.close()


def test_runtime_carries_shared_snapshot_id_into_request_and_actions() -> None:
    robot = RecordingRobot()
    requests = []

    class CaptureClient(Client):
        def step(self, request):
            requests.append(request)
            return super().step(request)

    def observation_source() -> RobotObservation:
        return RobotObservation(
            9.0,
            {"joint": 9},
            metadata={"observation_id": "service:g0:o7"},
        )

    runtime = ControlRuntime(
        robot,
        CaptureClient(),
        instruction="move",
        mapper=Mapper(),
        chunk_steps=2,
        observation_source=observation_source,
    )
    try:
        runtime.step(())
        assert requests[0].metadata["observation_id"] == "service:g0:o7"
        assert requests[0].metadata["snapshot_id"] == "service:g0:o7"
        assert all(action.metadata["observation_id"] == "service:g0:o7" for action in robot.executed)
    finally:
        runtime.close()


def test_uncertain_serialized_stop_does_not_release_manual_or_estop_authority() -> None:
    class SlowRobot(RecordingRobot):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def execute(self, command: RobotAction) -> None:
            self.started.set()
            self.release.wait(1.0)
            super().execute(command)

    robot = SlowRobot()
    port = RobotAdapterCommandPort(
        robot,
        stop_timeout_s=0.03,
        io_wait_timeout_s=None,
    )
    arbiter = RobotControlArbiter(port)
    try:
        active = arbiter.submit_model(action(5.0), wait=False)
        assert robot.started.wait(1.0)
        with pytest.raises(RuntimeError, match="stop I/O"):
            arbiter.acquire_manual()
        assert arbiter.snapshot()["authority"] == AuthorityState.ESTOP_LATCHED.value

        # The first stop is still queued behind the blocked SDK call.  A
        # duplicate hold must not be interpreted as a successful reset.
        with pytest.raises(RuntimeError, match="stop I/O"):
            arbiter.reset_emergency_stop()
        assert arbiter.snapshot()["authority"] == AuthorityState.ESTOP_LATCHED.value
        robot.release.set()
        active.wait(1.0)
        time.sleep(0.05)
    finally:
        robot.release.set()
        with contextlib.suppress(RuntimeError):
            # The test intentionally leaves the scheduler quarantined; close
            # may surface that unresolved stop while still releasing threads.
            arbiter.close()
