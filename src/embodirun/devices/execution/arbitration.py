"""Per-robot command arbitration for the control service runtime."""

from __future__ import annotations

import contextlib
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from embodirun.robots import RobotAction, RobotAdapter

from .io import IOOperationError, IOResult, IOStatus, IOUnknownError, RobotIOScheduler


class CommandSource(str, Enum):
    MODEL = "model"
    AGENT = "agent"
    REPLAY = "replay"
    MANUAL = "manual"


_AUTOMATIC_SOURCES = frozenset({CommandSource.MODEL, CommandSource.AGENT, CommandSource.REPLAY})


class AuthorityState(str, Enum):
    MODEL = "model"
    MANUAL = "manual"
    ESTOP_LATCHED = "estop_latched"


class CommandStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    CLEARED = "cleared"
    REPLACED = "replaced"
    EXPIRED = "expired"
    FAILED = "failed"


_TERMINAL_STATUSES = frozenset(
    {
        CommandStatus.EXECUTED,
        CommandStatus.CANCELLED,
        CommandStatus.CLEARED,
        CommandStatus.REPLACED,
        CommandStatus.EXPIRED,
        CommandStatus.FAILED,
    }
)


class CommandRejected(RuntimeError):
    """The current authority state rejected a command."""


class CommandCancelled(RuntimeError):
    """A command did not execute because authority changed."""


class EmergencyStopPolicy(str, Enum):
    """How emergency stop shares a port with an in-flight execute call."""

    SERIALIZED = "serialized"
    PREEMPTIVE = "preemptive"


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    source: CommandSource
    action: RobotAction
    deadline_s: float | None = None
    task_cancel: threading.Event | None = None


class CommandTicket:
    """Observable completion state for one submitted command."""

    def __init__(self, envelope: CommandEnvelope) -> None:
        self.envelope = envelope
        # One ordinary identifier follows this command through queued,
        # running, and terminal recorder events.  It is intentionally created
        # once per ticket rather than regenerated for each lifecycle stage.
        self.action_id = uuid.uuid4().hex
        self.cancel_event = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._status = CommandStatus.QUEUED
        self._error: BaseException | None = None
        self._result: IOResult | None = None

    @property
    def status(self) -> CommandStatus:
        with self._lock:
            return self._status

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def result(self) -> IOResult | None:
        """Driver transaction result, when the command reached a port."""

        with self._lock:
            return self._result

    def wait(self, timeout_s: float | None = None) -> CommandTicket:
        if not self._done.wait(timeout_s):
            raise TimeoutError("command did not finish before the timeout")
        return self

    def raise_for_failure(self) -> None:
        status = self.status
        if status is CommandStatus.FAILED:
            error = self.error
            if error is not None:
                raise error
            raise RuntimeError("command failed without an error")
        if status is not CommandStatus.EXECUTED:
            raise CommandCancelled(f"command ended with status {status.value}")

    def _mark_running(self) -> bool:
        with self._lock:
            if self._status is not CommandStatus.QUEUED:
                return False
            self._status = CommandStatus.RUNNING
            return True

    def _finish(
        self,
        status: CommandStatus,
        error: BaseException | None = None,
        result: IOResult | None = None,
    ) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"{status.value} is not a terminal command status")
        with self._lock:
            if self._status in _TERMINAL_STATUSES:
                return
            self._status = status
            self._error = error
            self._result = result
            self._done.set()


class RobotCommandPort(Protocol):
    robot_id: str

    def execute(
        self,
        action: RobotAction,
        cancel_event: threading.Event,
    ) -> None: ...

    def hold(self) -> None: ...

    def emergency_stop(self) -> None: ...


class RobotAdapterCommandPort:
    """Adapt one synchronous RobotAdapter to the arbiter port.

    SERIALIZED keeps execute, hold, and emergency_stop mutually exclusive for
    buses that cannot interleave stop traffic with motion commands. PREEMPTIVE
    is an explicit composition choice for adapters whose stop operation may run
    during an in-flight execute call.
    """

    def __init__(
        self,
        robot: RobotAdapter,
        *,
        action_resolver: Callable[[RobotAdapter, RobotAction], RobotAction] | None = None,
        emergency_stop_policy: EmergencyStopPolicy = (EmergencyStopPolicy.SERIALIZED),
        io_scheduler: RobotIOScheduler | None = None,
        stop_timeout_s: float = 0.25,
        io_wait_timeout_s: float | None = 1.0,
        observation_wait_timeout_s: float | None = 1.0,
    ) -> None:
        self.robot = robot
        self.robot_id = robot.robot_id
        self._action_resolver = action_resolver
        self._emergency_stop_policy = emergency_stop_policy
        self._stop_timeout_s = stop_timeout_s
        # These are caller wait budgets, not physical safety timeouts.  An SDK
        # call that outlives its budget keeps running in the scheduler and is
        # reported as UNKNOWN/quarantined rather than being claimed complete.
        self._io_wait_timeout_s = io_wait_timeout_s
        self._observation_wait_timeout_s = observation_wait_timeout_s
        self._io_scheduler = io_scheduler or RobotIOScheduler(
            robot.robot_id,
            stop_timeout_s=stop_timeout_s,
            allow_preemptive_stop=emergency_stop_policy is EmergencyStopPolicy.PREEMPTIVE,
        )
        self._owns_io_scheduler = io_scheduler is None

    def execute(
        self,
        action: RobotAction,
        cancel_event: threading.Event,
        *,
        source: CommandSource | str = CommandSource.MODEL,
    ) -> IOResult:
        def invoke() -> Any:
            if cancel_event.is_set():
                return
            resolved_action = action
            if self._action_resolver is not None:
                # Resolver reads (for example, teleop target resolution) run
                # inside this scheduler transaction, so they cannot bypass
                # the bus owner even when the resolver calls robot.observe().
                resolved_action = self._action_resolver(self.robot, action)
                if cancel_event.is_set():
                    return
            return self.robot.execute(resolved_action)

        result = self._io_scheduler.execute(
            invoke,
            requested=action,
            cancel_event=cancel_event,
            timeout_s=self._io_wait_timeout_s,
            source=source.value if isinstance(source, CommandSource) else str(source),
        )
        self._raise_io_failure(result)
        return result

    def observe(self) -> Any:
        """Read through the same bus scheduler used by actions and stops."""

        result = self._io_scheduler.read(
            self.robot.observe,
            requested="observe",
            timeout_s=self._observation_wait_timeout_s,
        )
        self._raise_io_failure(result)
        return result.driver_value

    def prepare(self) -> IOResult:
        """Prepare the adapter through the same serialized bus transaction."""

        prepare = getattr(self.robot, "prepare", None)
        if not callable(prepare):
            raise RuntimeError("robot adapter does not expose explicit prepare")
        result = self._io_scheduler.execute(
            prepare,
            requested="prepare",
            timeout_s=self._io_wait_timeout_s,
            source="lifecycle",
        )
        self._raise_io_failure(result)
        return result

    def hold(self) -> IOResult:
        result = self._io_scheduler.stop(
            self.robot.stop,
            requested="hold",
            timeout_s=self._stop_timeout_s,
            preemptive=False,
        )
        self._raise_io_failure(result)
        return result

    def emergency_stop(self) -> IOResult:
        result = self._io_scheduler.stop(
            self.robot.stop,
            requested="emergency_stop",
            timeout_s=self._stop_timeout_s,
            preemptive=self._emergency_stop_policy is EmergencyStopPolicy.PREEMPTIVE,
        )
        self._raise_io_failure(result)
        return result

    def close(self) -> None:
        """Close only the scheduler; the owning service closes the robot."""

        if self._owns_io_scheduler:
            result = self._io_scheduler.close()
            if result.status is IOStatus.UNKNOWN:
                raise IOUnknownError(result)

    @staticmethod
    def _raise_io_failure(result: IOResult) -> None:
        if result.status is IOStatus.UNKNOWN:
            raise IOUnknownError(result)
        if result.status is IOStatus.FAILED and result.error is not None:
            # Preserve the driver's original diagnostic for the arbiter's
            # existing last-error surface; the structured IOResult remains
            # available to callers that need scheduler context.
            with contextlib.suppress(Exception):
                result.error._embodirun_io_result = result
            raise result.error
        if result.status is IOStatus.REJECTED and result.operation == "stop":
            # Another stop is pending/running.  The scheduler keeps a single
            # bounded stop request; callers must still treat this result as
            # pending/unknown rather than as stop confirmation.
            return
        if result.status in (IOStatus.REJECTED, IOStatus.EXPIRED):
            raise IOOperationError(result)

    @staticmethod
    def stop_result_is_uncertain(result: object) -> bool:
        """Whether a stop result must keep arbiter authority unresolved."""

        return isinstance(result, IOResult) and result.status in {
            IOStatus.UNKNOWN,
            IOStatus.REJECTED,
        }


class ArbiterCommandSink:
    """Runtime-facing model command sink backed by RobotControlArbiter."""

    def __init__(self, arbiter: RobotControlArbiter, cancel_event: threading.Event | None = None) -> None:
        self.arbiter = arbiter
        self.robot_id = arbiter.robot_id
        self.cancel_event = cancel_event

    def execute(self, action: RobotAction) -> None:
        ticket = self.arbiter.submit_model(action, wait=True, task_cancel=self.cancel_event)
        ticket.raise_for_failure()

    def observe(self) -> Any:
        """Route policy observations through the arbiter's bus port."""

        return self.arbiter.observe()

    def stop(self) -> None:
        self.arbiter.cancel_automatic_work(self.cancel_event)

    def finish(self) -> None:
        """Hold after a runtime finishes without cancelling its task token.

        ``ControlRuntime.close`` runs for both successful and interrupted
        tasks.  A normal, synchronous task must still leave the robot held,
        but setting the registry-owned cancellation event during that cleanup
        makes a completed task look cancelled.  Interrupted tasks retain the
        old cancellation path so an explicit cancel/takeover/estop remains
        visible to the job registry.
        """

        self.arbiter.finish_automatic_work(self.cancel_event)


class RobotControlArbiter:
    """Own one robot's model/manual authority, cancellation, and estop latch.

    ``event_callback`` is an optional non-blocking recorder hook.  It receives
    detached command facts at proposal and terminal stages; callback failures
    are retained in the arbiter snapshot and never change command status.
    """

    def __init__(
        self,
        port: RobotCommandPort,
        *,
        model_queue_capacity: int = 1,
        manual_deadman_timeout_s: float = 0.25,
        clock=time.monotonic,
        event_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if model_queue_capacity < 1:
            raise ValueError("model_queue_capacity must be at least 1")
        if manual_deadman_timeout_s <= 0:
            raise ValueError("manual_deadman_timeout_s must be positive")
        if event_callback is not None and not callable(event_callback):
            raise TypeError("event_callback must be callable or None")
        self.port = port
        self.robot_id = port.robot_id
        self._model_queue_capacity = model_queue_capacity
        self._manual_deadman_timeout_s = manual_deadman_timeout_s
        self._clock = clock
        self._event_callback = event_callback
        self._condition = threading.Condition(threading.Lock())
        self._authority = AuthorityState.MODEL
        self._deadman_active = False
        self._manual_heartbeat_s: float | None = None
        self._model_queue: deque[CommandTicket] = deque()
        self._manual_pending: CommandTicket | None = None
        self._active: CommandTicket | None = None
        self._hold_in_progress = False
        self._hold_error: BaseException | None = None
        self._estop_generation = 0
        self._last_stop_error: str | None = None
        self._last_command_error: str | None = None
        self._last_event_error: str | None = None
        self._model_task_cancel: threading.Event | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{self.robot_id}-command-worker",
        )
        self._watchdog = threading.Thread(
            target=self._watch_deadman,
            daemon=True,
            name=f"{self.robot_id}-deadman-watchdog",
        )
        self._worker.start()
        self._watchdog.start()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "robot_id": self.robot_id,
                "authority": self._authority.value,
                "deadman_active": self._deadman_active,
                "active_source": (self._active.envelope.source.value if self._active else None),
                "active_status": self._active.status.value if self._active else None,
                "pending_model": len(self._model_queue),
                "manual_pending": self._manual_pending is not None,
                "hold_in_progress": self._hold_in_progress,
                "last_stop_error": self._last_stop_error,
                "last_error": self._last_command_error or self._last_stop_error,
                "last_event_error": self._last_event_error,
                "closed": self._closed,
            }

    def observe(self) -> Any:
        """Read an observation through a port that exposes shared I/O.

        Legacy test ports may not provide ``observe``; runtime callers then
        keep their explicitly injected or direct robot observation source.
        """

        observe = getattr(self.port, "observe", None)
        if observe is None:
            raise RuntimeError("robot command port does not expose observations")
        return observe()

    def prepare(self) -> IOResult:
        """Upgrade a passive adapter through its shared command port queue."""

        prepare = getattr(self.port, "prepare", None)
        if not callable(prepare):
            raise RuntimeError("robot command port does not expose prepare")
        return prepare()

    def begin_automatic_task(
        self,
        cancel_event: threading.Event | None = None,
    ) -> threading.Event:
        """Register one automatic task and invalidate its older results.

        The returned identity is carried by each submission.  A late
        ``finally`` from an older model/agent/replay task can therefore only
        cancel its own work; it cannot clear a newer automatic owner.
        """

        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event or None")
        if cancel_event is not None and cancel_event.is_set():
            raise CommandCancelled("automatic task was cancelled")

        with self._condition:
            self._require_open()
            if self._authority is not AuthorityState.MODEL:
                raise CommandRejected("model control is not available")
            if self._model_task_cancel is not None:
                self._model_task_cancel.set()
            self._cancel_active_automatic_locked()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._model_task_cancel = cancel_event or threading.Event()
            self._condition.notify_all()
            return self._model_task_cancel

    def finish_automatic_work(
        self,
        task_cancel: threading.Event | None = None,
    ) -> bool:
        """Finish one automatic task while preserving a successful token.

        This is the paired cleanup operation for :meth:`begin_automatic_task`.
        It performs the same physical hold as cancellation, then detaches the
        completed task token without setting it.  A late cleanup from an older
        task is ignored when a newer token has already taken ownership.
        """

        with self._condition:
            self._require_open()
            current_token = self._model_task_cancel
            if task_cancel is None:
                # An unscoped legacy sink must not finish another task that
                # acquired the arbiter after that sink was created.
                if current_token is not None:
                    return False
            elif task_cancel is not current_token:
                return False
            # A takeover or estop may have changed authority while a runtime
            # was unwinding.  Its late successful cleanup must never issue a
            # hold against the new manual owner or latched stop state.
            if self._authority is not AuthorityState.MODEL:
                return False
            cancelled = task_cancel is not None and task_cancel.is_set()
            self._cancel_active_automatic_ticket_locked()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        try:
            self._finish_hold()
        except Exception:
            # A failed or unknown hold must remain visible and must not detach
            # the token as though the robot had reached a safe state.
            raise
        with self._condition:
            if not cancelled and (task_cancel is None or self._model_task_cancel is current_token):
                self._model_task_cancel = None
            self._condition.notify_all()
        return True

    def begin_model_task(
        self,
        cancel_event: threading.Event | None = None,
    ) -> threading.Event:
        """Backward-compatible model alias for :meth:`begin_automatic_task`."""

        return self.begin_automatic_task(cancel_event)

    def submit_model(
        self,
        action: RobotAction,
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        task_cancel: threading.Event | None = None,
    ) -> CommandTicket:
        return self.submit(
            action,
            source=CommandSource.MODEL,
            wait=wait,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
            task_cancel=task_cancel,
        )

    def submit_agent(
        self,
        action: RobotAction,
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        task_cancel: threading.Event | None = None,
    ) -> CommandTicket:
        return self.submit(
            action,
            source=CommandSource.AGENT,
            wait=wait,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
            task_cancel=task_cancel,
        )

    def submit_replay(
        self,
        action: RobotAction,
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        task_cancel: threading.Event | None = None,
    ) -> CommandTicket:
        return self.submit(
            action,
            source=CommandSource.REPLAY,
            wait=wait,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
            task_cancel=task_cancel,
        )

    def submit_manual(
        self,
        action: RobotAction,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
    ) -> CommandTicket:
        return self.submit(
            action,
            source=CommandSource.MANUAL,
            wait=wait,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
        )

    def submit(
        self,
        action: RobotAction,
        *,
        source: CommandSource,
        wait: bool = True,
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        task_cancel: threading.Event | None = None,
    ) -> CommandTicket:
        if deadline_s is not None and (
            isinstance(deadline_s, bool) or not isinstance(deadline_s, (int, float)) or not math.isfinite(deadline_s)
        ):
            raise ValueError("deadline_s must be finite")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and non-negative")
        if not isinstance(action, RobotAction):
            raise TypeError("action must be a RobotAction")
        if source is CommandSource.MANUAL and task_cancel is not None:
            raise ValueError("manual commands cannot carry an automatic task token")
        if source not in _AUTOMATIC_SOURCES and source is not CommandSource.MANUAL:
            raise ValueError(f"unsupported command source: {source!r}")
        envelope = CommandEnvelope(
            source=source,
            action=action,
            deadline_s=deadline_s,
            task_cancel=task_cancel,
        )
        ticket = CommandTicket(envelope)
        with self._condition:
            self._require_open()
            if task_cancel is not None and task_cancel is not self._model_task_cancel:
                raise CommandCancelled("automatic task is no longer current")
            if task_cancel is not None and task_cancel.is_set():
                raise CommandCancelled("automatic task was cancelled")
            if self._authority is AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is latched")
            if source is CommandSource.MANUAL:
                if self._authority is not AuthorityState.MANUAL:
                    raise CommandRejected("manual control is not acquired")
                if not self._deadman_active:
                    raise CommandRejected("manual deadman is not active")
                self._manual_heartbeat_s = self._clock()
                if self._manual_pending is not None:
                    self._manual_pending._finish(CommandStatus.REPLACED)
                self._manual_pending = ticket
            else:
                if self._authority is AuthorityState.MANUAL:
                    raise CommandRejected("manual control owns the robot")
                if len(self._model_queue) >= self._model_queue_capacity:
                    raise CommandRejected("model command queue is full")
                self._model_queue.append(ticket)
            self._condition.notify_all()
        # The callback is outside the arbiter lock.  A queued proposal is
        # recorded; a takeover that clears it before dispatch has no driver
        # transaction to report and remains visible through the ticket state.
        self._emit_event(ticket, CommandStatus.QUEUED, None)
        if wait:
            ticket.wait(timeout_s)
        return ticket

    def acquire_manual(self) -> None:
        join_existing_hold = False
        with self._condition:
            self._require_open()
            if self._authority is AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is latched")
            self._authority = AuthorityState.MANUAL
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            join_existing_hold = self._hold_in_progress
            if not join_existing_hold:
                self._hold_error = None
                self._hold_in_progress = True
            self._condition.notify_all()
        if join_existing_hold:
            self._wait_for_hold()
        else:
            self._finish_hold()

    def release_manual(self) -> None:
        with self._condition:
            self._require_open()
            if self._authority is AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is latched")
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active(CommandSource.MANUAL)
            self._clear_manual(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        self._finish_hold(next_authority=AuthorityState.MODEL)

    def set_deadman(self, active: bool) -> None:
        should_hold = False
        with self._condition:
            self._require_open()
            if self._authority is not AuthorityState.MANUAL:
                raise CommandRejected("manual control is not acquired")
            self._deadman_active = bool(active)
            self._manual_heartbeat_s = self._clock() if active else None
            if not active:
                self._cancel_active(CommandSource.MANUAL)
                self._clear_manual(CommandStatus.CLEARED)
                self._hold_in_progress = True
                should_hold = True
            self._condition.notify_all()
        if should_hold:
            self._finish_hold()

    def hold(self) -> None:
        with self._condition:
            self._require_open()
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        self._finish_hold()

    def cancel_model_work(self) -> None:
        """Cancel the current automatic task (legacy model alias)."""

        self.cancel_automatic_work()

    def cancel_automatic_work(self, task_cancel: threading.Event | None = None) -> bool:
        """Cancel only the automatic task identified by ``task_cancel``.

        A stale sink passes its old token here.  If a newer task is already
        registered, the stale cleanup is ignored and cannot cancel the new
        model/agent/replay owner.
        """

        with self._condition:
            self._require_open()
            current_token = self._model_task_cancel
            if task_cancel is not None and task_cancel is not current_token:
                return False
            if current_token is not None:
                current_token.set()
            active = self._active
            active_is_automatic = (
                active is not None
                and active.envelope.source in _AUTOMATIC_SOURCES
                and (task_cancel is None or active.envelope.task_cancel is task_cancel)
            )
            self._clear_model_queue(CommandStatus.CLEARED)
            if active_is_automatic:
                active.cancel_event.set()
            should_hold = self._authority is AuthorityState.MODEL
            if should_hold:
                self._hold_in_progress = True
            self._condition.notify_all()
        if should_hold:
            self._finish_hold()
        return True

    def emergency_stop(self) -> None:
        with self._condition:
            self._require_open()
            self._estop_generation += 1
            self._authority = AuthorityState.ESTOP_LATCHED
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._cancel_active()
            self._clear_model_queue(CommandStatus.CLEARED)
            self._clear_manual(CommandStatus.CLEARED)
            self._hold_in_progress = True
            self._condition.notify_all()
        try:
            stop_result = self.port.emergency_stop()
            if RobotAdapterCommandPort.stop_result_is_uncertain(stop_result):
                raise IOUnknownError(stop_result)
        except Exception as error:
            with self._condition:
                self._last_stop_error = str(error)
            raise
        finally:
            with self._condition:
                self._hold_in_progress = False
                self._condition.notify_all()

    def reset_emergency_stop(self) -> None:
        with self._condition:
            self._require_open()
            if self._authority is not AuthorityState.ESTOP_LATCHED:
                raise CommandRejected("emergency stop is not latched")
            estop_generation = self._estop_generation
        try:
            stop_result = self.port.hold()
            if RobotAdapterCommandPort.stop_result_is_uncertain(stop_result):
                raise IOUnknownError(stop_result)
        except Exception as error:
            with self._condition:
                self._last_stop_error = str(error)
            raise
        with self._condition:
            self._require_open()
            if self._estop_generation != estop_generation:
                raise CommandRejected("emergency stop was triggered during reset")
            self._authority = AuthorityState.MANUAL
            self._deadman_active = False
            self._manual_heartbeat_s = None
            self._last_stop_error = None
            self._last_command_error = None
            self._condition.notify_all()

    def close(self, *, hold: bool = True) -> None:
        """Stop command workers and optionally issue a physical hold.

        A passive observation owner has never prepared motors, so asking its
        adapter to hold during teardown would create the very write that the
        passive boundary forbids.  Prepared control owners retain the default
        hold behavior.
        """

        with self._condition:
            already_closed = self._closed
            if not already_closed:
                self._closed = True
                self._deadman_active = False
                self._manual_heartbeat_s = None
                self._cancel_active()
                self._clear_model_queue(CommandStatus.CLEARED)
                self._clear_manual(CommandStatus.CLEARED)
                self._condition.notify_all()
        close_error: Exception | None = None
        stop_in_progress = False
        if hold and not already_closed:
            deadline = time.monotonic() + 1.0
            with self._condition:
                while self._hold_in_progress:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                stop_in_progress = self._hold_in_progress
        try:
            if hold and not already_closed:
                if stop_in_progress:
                    # Do not issue a second non-thread-safe SDK stop while an
                    # emergency stop is still in flight.  The owner remains
                    # unresolved and the caller receives an explicit result.
                    close_error = IOUnknownError(
                        IOResult(
                            operation="stop",
                            status=IOStatus.UNKNOWN,
                            requested="hold",
                            quarantined=True,
                            error=RuntimeError("emergency stop is still in flight"),
                        )
                    )
                else:
                    stop_result = self.port.hold()
                    if RobotAdapterCommandPort.stop_result_is_uncertain(stop_result):
                        # A pending/rejected stop is not evidence that this
                        # owner is safe to release.  Keep the error visible to
                        # the service so it can retain the adapter and bus.
                        close_error = IOUnknownError(stop_result)
        except Exception as error:
            close_error = error
        finally:
            close_port = getattr(self.port, "close", None)
            if close_port is not None:
                try:
                    close_port()
                except Exception as error:
                    if close_error is None:
                        close_error = error
            self._worker.join(timeout=1.0)
            self._watchdog.join(timeout=1.0)
        if close_error is not None:
            raise close_error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("robot control arbiter is closed")

    def _finish_hold(
        self,
        *,
        next_authority: AuthorityState | None = None,
    ) -> None:
        try:
            stop_result = self.port.hold()
            if RobotAdapterCommandPort.stop_result_is_uncertain(stop_result):
                raise IOUnknownError(stop_result)
        except Exception as error:
            with self._condition:
                self._estop_generation += 1
                self._authority = AuthorityState.ESTOP_LATCHED
                self._deadman_active = False
                self._manual_heartbeat_s = None
                self._hold_in_progress = False
                self._hold_error = error
                self._last_stop_error = str(error)
                self._cancel_active()
                self._clear_model_queue(CommandStatus.CLEARED)
                self._clear_manual(CommandStatus.CLEARED)
                self._condition.notify_all()
            raise
        with self._condition:
            if next_authority is not None and self._authority is not AuthorityState.ESTOP_LATCHED:
                self._authority = next_authority
            self._hold_in_progress = False
            self._hold_error = None
            if self._authority is not AuthorityState.ESTOP_LATCHED:
                self._last_stop_error = None
            self._condition.notify_all()

    def _wait_for_hold(self) -> None:
        """Join a hold already owned by another authority transition."""

        with self._condition:
            while self._hold_in_progress and not self._closed:
                self._condition.wait()
            error = self._hold_error
        if error is not None:
            raise error

    def _cancel_active(self, source: CommandSource | None = None) -> None:
        if (source is None or source in _AUTOMATIC_SOURCES) and self._model_task_cancel is not None:
            self._model_task_cancel.set()
        if self._active is None:
            return
        if source is None or self._active.envelope.source is source:
            self._active.cancel_event.set()

    def _cancel_active_automatic_locked(self) -> None:
        """Cancel active automatic work while the condition is held."""

        if self._model_task_cancel is not None:
            self._model_task_cancel.set()
        if self._active is not None and self._active.envelope.source in _AUTOMATIC_SOURCES:
            self._active.cancel_event.set()

    def _cancel_active_automatic_ticket_locked(self) -> None:
        """Stop queued/in-flight automatic tickets without setting task token."""

        if self._active is not None and self._active.envelope.source in _AUTOMATIC_SOURCES:
            self._active.cancel_event.set()

    def _clear_model_queue(self, status: CommandStatus) -> bool:
        cleared_any = False
        while self._model_queue:
            self._model_queue.popleft()._finish(status)
            cleared_any = True
        return cleared_any

    def _clear_manual(self, status: CommandStatus) -> None:
        if self._manual_pending is not None:
            self._manual_pending._finish(status)
            self._manual_pending = None

    def _next_ticket(self) -> CommandTicket | None:
        if self._hold_in_progress or self._authority is AuthorityState.ESTOP_LATCHED:
            return None
        if self._authority is AuthorityState.MANUAL:
            if not self._deadman_active:
                return None
            ticket = self._manual_pending
            self._manual_pending = None
            return ticket
        if self._model_queue:
            return self._model_queue.popleft()
        return None

    def _run(self) -> None:
        while True:
            early_status: CommandStatus | None = None
            with self._condition:
                ticket = self._next_ticket()
                while ticket is None and not self._closed:
                    self._condition.wait()
                    ticket = self._next_ticket()
                if self._closed:
                    return
                if not ticket._mark_running():
                    continue
                if ticket.envelope.task_cancel is not None and ticket.envelope.task_cancel.is_set():
                    ticket._finish(CommandStatus.CANCELLED)
                    early_status = CommandStatus.CANCELLED
                else:
                    self._active = ticket
                    deadline_s = ticket.envelope.deadline_s
                if early_status is None and deadline_s is not None and self._clock() > deadline_s:
                    self._active = None
                    ticket._finish(CommandStatus.EXPIRED)
                    early_status = CommandStatus.EXPIRED
            if early_status is not None:
                self._emit_event(ticket, early_status, None)
                continue
            port_result: IOResult | None = None
            try:
                if isinstance(self.port, RobotAdapterCommandPort):
                    port_result = self.port.execute(
                        ticket.envelope.action,
                        ticket.cancel_event,
                        source=ticket.envelope.source,
                    )
                else:
                    maybe_result = self.port.execute(
                        ticket.envelope.action,
                        ticket.cancel_event,
                    )
                    if isinstance(maybe_result, IOResult):
                        port_result = maybe_result
            except Exception as error:
                if port_result is None:
                    port_result = getattr(error, "result", None)
                    if not isinstance(port_result, IOResult):
                        port_result = getattr(error, "_embodirun_io_result", None)
                was_cancelled = ticket.cancel_event.is_set()
                status = CommandStatus.CANCELLED if was_cancelled else CommandStatus.FAILED
                with self._condition:
                    self._last_command_error = str(error)
                if not was_cancelled:
                    # A wait=True caller must not observe a terminal command
                    # failure before the scoped emergency stop has completed.
                    # emergency_stop retains the stop error and latch.
                    with contextlib.suppress(Exception):
                        self.emergency_stop()
                self._emit_event(ticket, status, port_result)
                ticket._finish(status, error, result=port_result)
            else:
                status, result_error = self._map_port_result(
                    port_result,
                    cancelled=ticket.cancel_event.is_set(),
                )
                if status is CommandStatus.FAILED:
                    with self._condition:
                        self._last_command_error = str(result_error)
                    # Publish failure only after emergency_stop has finished, so
                    # wait=True cannot race the owner-scoped release.
                    # emergency_stop retains the stop error and latch.
                    with contextlib.suppress(Exception):
                        self.emergency_stop()
                self._emit_event(ticket, status, port_result)
                ticket._finish(status, result_error, result=port_result)
            finally:
                with self._condition:
                    if self._active is ticket:
                        self._active = None
                    self._condition.notify_all()

    def _emit_event(
        self,
        ticket: CommandTicket,
        status: CommandStatus,
        result: IOResult | None,
    ) -> None:
        callback = self._event_callback
        if callback is None:
            return
        payload = {
            # A ticket object's address can be reused after it is collected;
            # event consumers need an ordinary per-action identifier instead
            # of a lifetime-dependent Python object identity.
            "action_id": f"{self.robot_id}:command-{ticket.action_id}",
            "source": ticket.envelope.source.value,
            "stage": status.value,
            "executed": status is CommandStatus.EXECUTED,
            "observation_id": _observation_id(ticket.envelope.action),
            "outcome": status.value,
            "payload": {
                "requested": _event_value(ticket.envelope.action.values),
                "metadata": _event_value(ticket.envelope.action.metadata),
                "io_status": None if result is None else result.status.value,
                "driver_returned": None if result is None else result.driver_returned,
                "driver_receipt": (
                    None if result is None or not result.driver_returned else _event_value(result.driver_value)
                ),
                # The arbiter cannot infer adapter mapping or measured state;
                # leave those facts explicit for a later adapter callback.
                "mapped_or_limited": None,
                "measured_feedback": None,
            },
            "timestamp_ns": time.monotonic_ns(),
        }
        try:
            callback(payload)
        except BaseException as error:
            with self._condition:
                self._last_event_error = str(error)

    @staticmethod
    def _map_port_result(
        result: IOResult | None,
        *,
        cancelled: bool,
    ) -> tuple[CommandStatus, BaseException | None]:
        """Map every generic port outcome without promoting failure to success."""

        if result is None:
            return (
                CommandStatus.CANCELLED if cancelled else CommandStatus.EXECUTED,
                None,
            )
        if result.status is IOStatus.COMPLETED:
            return (
                CommandStatus.CANCELLED if cancelled else CommandStatus.EXECUTED,
                None,
            )
        if result.status is IOStatus.CANCELLED:
            return CommandStatus.CANCELLED, None
        if result.status is IOStatus.EXPIRED:
            return CommandStatus.EXPIRED, None
        if result.status is IOStatus.UNKNOWN:
            return CommandStatus.FAILED, IOUnknownError(result)
        if result.status is IOStatus.FAILED and result.error is not None:
            # Keep the driver's diagnostic as the ticket error while the
            # structured IOResult remains attached to the ticket.  Wrapping
            # it would hide the original failure from callers and telemetry.
            return CommandStatus.FAILED, result.error
        if result.status in {IOStatus.FAILED, IOStatus.REJECTED}:
            return CommandStatus.FAILED, IOOperationError(result)
        return CommandStatus.FAILED, IOOperationError(result)

    def _watch_deadman(self) -> None:
        while True:
            should_hold = False
            with self._condition:
                if self._closed:
                    return
                if (
                    self._authority is not AuthorityState.MANUAL
                    or not self._deadman_active
                    or self._manual_heartbeat_s is None
                ):
                    self._condition.wait()
                    continue
                remaining = self._manual_heartbeat_s + self._manual_deadman_timeout_s - self._clock()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                self._deadman_active = False
                self._manual_heartbeat_s = None
                self._cancel_active(CommandSource.MANUAL)
                self._clear_manual(CommandStatus.CLEARED)
                self._hold_in_progress = True
                self._condition.notify_all()
                should_hold = True
            if should_hold:
                try:
                    self._finish_hold()
                except Exception:
                    continue


def _observation_id(action: RobotAction) -> str | None:
    value = action.metadata.get("observation_id")
    return value if isinstance(value, str) and value.strip() else None


def _event_value(value: Any) -> Any:
    """Detach common action/driver values without retaining SDK objects."""

    if isinstance(value, Mapping):
        return {str(key): _event_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ArbiterCommandSink",
    "AuthorityState",
    "CommandCancelled",
    "CommandEnvelope",
    "CommandRejected",
    "CommandSource",
    "CommandStatus",
    "CommandTicket",
    "EmergencyStopPolicy",
    "IOOperationError",
    "IOResult",
    "IOStatus",
    "IOUnknownError",
    "RobotAdapterCommandPort",
    "RobotCommandPort",
    "RobotControlArbiter",
    "RobotIOScheduler",
]
