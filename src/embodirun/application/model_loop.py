"""Generic request-response control loop for a robot-policy binding."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from embodirun.bindings import BindingMapper
from embodirun.model_services import InferenceClient, PolicyResult
from embodirun.robots import RobotAction, RobotAdapter, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame


class ControlRuntimeCancelled(RuntimeError):
    """A task-scoped control interruption cancelled model execution."""


class CommandSink(Protocol):
    robot_id: str

    def execute(self, action: RobotAction) -> None: ...

    def stop(self) -> None: ...


class ControlRuntime:
    """Own one policy session and execute mapped actions on one robot."""

    def __init__(
        self,
        robot: RobotAdapter,
        client: InferenceClient,
        *,
        instruction: str,
        mapper: BindingMapper,
        chunk_steps: int,
        control_hz: float = 5.0,
        command_sink: CommandSink | None = None,
        observation_source: Callable[[], RobotObservation] | None = None,
        cancel_event: threading.Event | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if isinstance(chunk_steps, bool) or not isinstance(chunk_steps, int) or chunk_steps <= 0:
            raise ValueError("chunk_steps must be a positive integer")
        if (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not math.isfinite(control_hz)
            or control_hz <= 0
        ):
            raise ValueError("control_hz must be a finite positive number")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.mapper = mapper
        self.chunk_steps = chunk_steps
        self.command_sink = command_sink
        # The optional source lets a caller route observations through the
        # same per-bus scheduler as actions.  The legacy direct robot path is
        # retained for adapters that have not opted into shared I/O yet.
        if observation_source is None and command_sink is not None:
            candidate = getattr(command_sink, "observe", None)
            owner = getattr(command_sink, "arbiter", None)
            port = getattr(owner, "port", None)
            if owner is not None and not callable(getattr(port, "observe", None)):
                candidate = None
            observation_source = candidate if callable(candidate) else None
        self.observation_source = observation_source
        if command_sink is not None and command_sink.robot_id != robot.robot_id:
            raise ValueError("command_sink robot_id must match robot")
        self.cancel_event = cancel_event
        self.action_period_s = 1.0 / control_hz
        self.monotonic = monotonic
        self.sleep = sleep
        self.session = client.open_session(
            robot_id=robot.robot_id,
            action_space=mapper.policy_action_space,
        )
        self.step_id = 0

    def step(
        self,
        frames: Sequence[CameraFrame],
        *,
        reset: bool = False,
    ) -> PolicyResult:
        if reset:
            self.reset()
        self._raise_if_cancelled()
        observation = self.robot.observe() if self.observation_source is None else self.observation_source()
        request = self.mapper.map_observation(
            observation,
            session_id=self.session.session_id,
            request_id=f"step-{self.step_id}-{uuid.uuid4().hex}",
            step_id=self.step_id,
            instruction=self.instruction,
            frames=tuple(frames),
        )
        observation_metadata = getattr(observation, "metadata", {})
        observation_id = (
            observation_metadata.get("observation_id") if isinstance(observation_metadata, Mapping) else None
        )
        if isinstance(observation_id, str) and observation_id:
            request_metadata = getattr(request, "metadata", {})
            metadata = dict(request_metadata) if isinstance(request_metadata, Mapping) else {}
            metadata.setdefault("observation_id", observation_id)
            metadata.setdefault("snapshot_id", observation_id)
            if hasattr(request, "__dataclass_fields__"):
                request = replace(request, metadata=metadata)
            else:
                # Keep lightweight test/custom mappers usable while the
                # versioned inference contracts remain frozen dataclasses.
                try:
                    request.metadata = metadata
                except AttributeError as error:
                    raise TypeError("mapper request must support metadata propagation") from error
        result = self.client.step(request)
        self._raise_if_cancelled()
        actions = tuple(self.mapper.map_result(result))
        if not actions:
            raise RuntimeError("binding returned an empty action chunk")
        if any(not isinstance(action, RobotAction) for action in actions):
            raise TypeError("binding action chunk must contain RobotAction values")
        if isinstance(observation_id, str) and observation_id:
            actions = tuple(
                replace(
                    action,
                    metadata={
                        **dict(action.metadata),
                        "observation_id": observation_id,
                        "snapshot_id": observation_id,
                    },
                )
                for action in actions
            )
        if len(actions) < self.chunk_steps:
            raise RuntimeError(
                f"binding returned {len(actions)} action(s), fewer than requested chunk_steps={self.chunk_steps}"
            )
        actions = actions[: self.chunk_steps]
        deadline_s = self.monotonic()
        for index, action in enumerate(actions):
            self._raise_if_cancelled()
            self._execute(action)
            if index + 1 == len(actions):
                continue
            deadline_s += self.action_period_s
            remaining_s = deadline_s - self.monotonic()
            if remaining_s > 0:
                if self.cancel_event is None:
                    self.sleep(remaining_s)
                else:
                    self.cancel_event.wait(remaining_s)
                    self._raise_if_cancelled()
        self.step_id += 1
        return result

    def reset(self) -> None:
        self._stop()
        self.session = self.client.reset(
            self.session.session_id,
            request_id=f"reset-{uuid.uuid4().hex}",
        )
        self.step_id = 0

    def close(self) -> None:
        try:
            # A normal task still needs a physical hold, but its registry
            # cancellation event must remain clear so the completed task is
            # not rewritten as ``cancelled`` during finalization.  The shared
            # arbiter sink exposes ``finish`` for that scoped distinction;
            # older sinks retain the historical stop behavior.
            finish = getattr(self.command_sink, "finish", None)
            if self.command_sink is not None and callable(finish):
                finish()
            else:
                self._stop()
        finally:
            self.client.close(self.session.session_id)

    def _execute(self, action: RobotAction) -> None:
        if self.command_sink is None:
            self.robot.execute(action)
            return
        self.command_sink.execute(action)

    def _stop(self) -> None:
        if self.command_sink is None:
            self.robot.stop()
            return
        self.command_sink.stop()

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ControlRuntimeCancelled("model task was cancelled by control authority")


__all__ = ["CommandSink", "ControlRuntime", "ControlRuntimeCancelled"]
