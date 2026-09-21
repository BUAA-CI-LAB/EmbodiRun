"""Bounded direct-agent/replay execution through the existing arbiter.

This module deliberately does not know a robot SDK or solve kinematics.  One
short segment is submitted to the arbiter and paced locally; the adapter keeps
the action-space, numerical, unit, and hardware-limit checks.  A missing
adapter capability therefore remains ``unsupported`` instead of being
silently converted to an EE or joint command here.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from embodirun.devices.execution.arbitration import (
    CommandSource,
    CommandStatus,
    RobotControlArbiter,
)
from embodirun.devices.execution.io import IOResult
from embodirun.robots import RobotAction

from .jobs import PhysicalStatus


class DirectExecutionError(RuntimeError):
    """A direct segment could not complete through the arbiter."""

    def __init__(self, message: str, *, ticket: Any = None) -> None:
        self.ticket = ticket
        super().__init__(message)


class DirectExecutionCancelled(DirectExecutionError):
    """The job cancellation token stopped a segment before completion."""


class DirectExecutionUnsupported(DirectExecutionError):
    """The configured adapter cannot accept this action space."""


@dataclass(frozen=True, slots=True)
class DirectActionOutcome:
    """One requested action and the facts available after dispatch."""

    requested: Mapping[str, Any]
    command_status: str
    io_status: str | None
    driver_returned: bool | None
    driver_receipt: Any = None
    mapped_or_limited: Any = None
    measured_feedback: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": dict(self.requested),
            "command_status": self.command_status,
            "io_status": self.io_status,
            "driver_returned": self.driver_returned,
            "driver_receipt": self.driver_receipt,
            "mapped_or_limited": self.mapped_or_limited,
            "measured_feedback": self.measured_feedback,
        }


@dataclass(frozen=True, slots=True)
class DirectExecutionResult:
    """Facts for one bounded segment; it is not a business-success claim."""

    source: str
    requested_steps: int
    dispatched_steps: int
    executed_steps: int
    observation_id: str | None
    control_hz: float
    elapsed_s: float
    outcomes: tuple[DirectActionOutcome, ...]
    physical_status: str = PhysicalStatus.NOT_REQUESTED.value
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "requested_steps": self.requested_steps,
            "dispatched_steps": self.dispatched_steps,
            "executed_steps": self.executed_steps,
            "observation_id": self.observation_id,
            "control_hz": self.control_hz,
            "elapsed_s": self.elapsed_s,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "physical_status": self.physical_status,
            "simulated": self.simulated,
            "business_success": None,
        }


class DirectExecutionRunner:
    """Run one bounded action sequence using the existing control arbiter."""

    def __init__(
        self,
        arbiter_provider: Callable[[], RobotControlArbiter],
        *,
        max_steps: int = 50,
        step_timeout_s: float = 1.0,
        max_duration_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        simulated: bool = False,
    ) -> None:
        if not callable(arbiter_provider):
            raise TypeError("arbiter_provider must be callable")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(step_timeout_s, bool)
            or not isinstance(step_timeout_s, (int, float))
            or not math.isfinite(step_timeout_s)
            or step_timeout_s <= 0
        ):
            raise ValueError("step_timeout_s must be finite and positive")
        if (
            isinstance(max_duration_s, bool)
            or not isinstance(max_duration_s, (int, float))
            or not math.isfinite(max_duration_s)
            or max_duration_s <= 0
        ):
            raise ValueError("max_duration_s must be finite and positive")
        self._arbiter_provider = arbiter_provider
        self.max_steps = max_steps
        self.step_timeout_s = float(step_timeout_s)
        self.max_duration_s = float(max_duration_s)
        self._clock = clock
        self._sleep = sleep
        self.simulated = simulated
        self._sessions_lock = threading.Lock()
        self._sessions: dict[threading.Event, threading.Event] = {}

    def run(
        self,
        actions: Sequence[RobotAction],
        *,
        source: CommandSource | str,
        cancel_event: threading.Event,
        control_hz: float,
        observation_id: str | None = None,
    ) -> DirectExecutionResult:
        source_value = _automatic_source(source)
        action_values = tuple(actions)
        if not action_values:
            raise ValueError("direct segment must contain at least one action")
        if len(action_values) > self.max_steps:
            raise ValueError(f"direct segment exceeds max_steps={self.max_steps}")
        if (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not math.isfinite(control_hz)
            or control_hz <= 0
        ):
            raise ValueError("control_hz must be finite and positive")
        if (len(action_values) - 1) / float(control_hz) > self.max_duration_s:
            raise DirectExecutionError("direct segment exceeds max_duration_s")
        if not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event")

        arbiter = self._arbiter_provider()
        control_token = arbiter.begin_automatic_task()
        with self._sessions_lock:
            self._sessions[cancel_event] = control_token
        started = self._clock()
        period_s = 1.0 / float(control_hz)
        next_tick = started
        outcomes: list[DirectActionOutcome] = []
        result: DirectExecutionResult | None = None
        release_status = PhysicalStatus.STOP_UNCONFIRMED.value
        release_error: BaseException | None = None
        try:
            for action in action_values:
                if cancel_event.is_set():
                    raise DirectExecutionCancelled("direct segment was cancelled")
                if self._clock() - started > self.max_duration_s:
                    raise DirectExecutionError("direct segment exceeded max_duration_s")
                if not isinstance(action, RobotAction):
                    raise TypeError("direct segment actions must be RobotAction values")
                ticket = self._submit(arbiter, action, source_value, cancel_event)
                io_result = ticket.result
                outcomes.append(_outcome(action, ticket.status, io_result))
                if ticket.status is CommandStatus.CANCELLED:
                    raise DirectExecutionCancelled("direct command was cancelled", ticket=ticket)
                if ticket.status is not CommandStatus.EXECUTED:
                    raise DirectExecutionError(
                        f"direct command ended with status {ticket.status.value}",
                        ticket=ticket,
                    )
                next_tick += period_s
                remaining = next_tick - self._clock()
                # Event.wait keeps cadence bounded while allowing a
                # cancellation request to interrupt the idle interval.
                if remaining > 0 and len(outcomes) < len(action_values) and cancel_event.wait(remaining):
                    raise DirectExecutionCancelled("direct segment was cancelled")
            result = DirectExecutionResult(
                source=source_value.value,
                requested_steps=len(action_values),
                dispatched_steps=len(outcomes),
                executed_steps=sum(outcome.command_status == CommandStatus.EXECUTED.value for outcome in outcomes),
                observation_id=observation_id,
                control_hz=float(control_hz),
                elapsed_s=max(0.0, self._clock() - started),
                outcomes=tuple(outcomes),
                simulated=self.simulated,
            )
        finally:
            # Release only this automatic token.  If a newer owner has
            # already replaced it, the arbiter returns False and leaves that
            # owner untouched.
            try:
                released = arbiter.cancel_automatic_work(control_token)
                release_status = (
                    PhysicalStatus.STOP_REQUESTED.value if released else PhysicalStatus.STOP_UNCONFIRMED.value
                )
            except BaseException as error:
                release_error = error
            with self._sessions_lock:
                if self._sessions.get(cancel_event) is control_token:
                    self._sessions.pop(cancel_event, None)
        if release_error is not None:
            if result is not None:
                raise DirectExecutionError(
                    "direct segment completed but its owned hold is unconfirmed"
                ) from release_error
            raise release_error
        if result is None:
            raise DirectExecutionError("direct segment ended without a result")
        return replace(result, physical_status=release_status)

    def cancel(self, cancel_event: threading.Event) -> Mapping[str, Any]:
        """Cancel only the arbiter task token belonging to this job."""

        with self._sessions_lock:
            control_token = self._sessions.get(cancel_event)
        if control_token is None:
            return {
                "physical_status": PhysicalStatus.STOP_UNCONFIRMED.value,
                "stop_confirmed": False,
            }
        try:
            cancel_event.set()
            accepted = self._arbiter_provider().cancel_automatic_work(control_token)
        except Exception as error:
            return {
                "physical_status": PhysicalStatus.STOP_UNCONFIRMED.value,
                "stop_confirmed": False,
                "error": str(error),
            }
        return {
            "physical_status": (
                PhysicalStatus.STOP_REQUESTED.value if accepted else PhysicalStatus.STOP_UNCONFIRMED.value
            ),
            "stop_confirmed": False,
        }

    def _submit(
        self,
        arbiter: RobotControlArbiter,
        action: RobotAction,
        source: CommandSource,
        cancel_event: threading.Event,
    ) -> Any:
        with self._sessions_lock:
            control_token = self._sessions.get(cancel_event)
        if control_token is None:
            raise DirectExecutionCancelled("direct execution token is no longer active")
        if source is CommandSource.AGENT:
            return arbiter.submit_agent(
                action,
                wait=True,
                timeout_s=self.step_timeout_s,
                task_cancel=control_token,
            )
        return arbiter.submit_replay(
            action,
            wait=True,
            timeout_s=self.step_timeout_s,
            task_cancel=control_token,
        )


def _automatic_source(value: CommandSource | str) -> CommandSource:
    try:
        source = value if isinstance(value, CommandSource) else CommandSource(value)
    except (TypeError, ValueError) as error:
        raise DirectExecutionUnsupported("direct execution source must be 'agent' or 'replay'") from error
    if source not in {CommandSource.AGENT, CommandSource.REPLAY}:
        raise DirectExecutionUnsupported("direct execution cannot claim model or manual ownership")
    return source


def action_payload(action: RobotAction) -> dict[str, Any]:
    """Detach a normalized action for idempotency and response records."""

    if not isinstance(action, RobotAction):
        raise TypeError("action must be a RobotAction")
    return {
        "timestamp_s": action.timestamp_s,
        "values": _detached(action.values),
        "metadata": _detached(action.metadata),
    }


def _outcome(
    action: RobotAction,
    command_status: CommandStatus,
    result: IOResult | None,
) -> DirectActionOutcome:
    return DirectActionOutcome(
        requested=action_payload(action),
        command_status=command_status.value,
        io_status=None if result is None else result.status.value,
        driver_returned=None if result is None else result.driver_returned,
        # ``driver_value`` is a receipt from the call, not measured feedback.
        driver_receipt=(None if result is None or not result.driver_returned else _detached(result.driver_value)),
    )


def _detached(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _detached(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detached(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("action values must contain finite numbers")
        return value
    raise TypeError(f"action value {type(value).__name__} is not JSON-compatible")


__all__ = [
    "DirectActionOutcome",
    "DirectExecutionCancelled",
    "DirectExecutionError",
    "DirectExecutionResult",
    "DirectExecutionRunner",
    "DirectExecutionUnsupported",
    "action_payload",
]
