"""Bounded, single-owner I/O scheduling for a physical robot bus.

The scheduler is deliberately small.  It owns ordering and admission for one
physical bus; it does not decide authority, transform actions, or infer robot
limits.  Callers that share a non-thread-safe device pass the same scheduler
instance to every adapter using that bus.  Independent buses use independent
instances and may make progress independently.

An SDK call cannot be interrupted from Python.  If a serialized stop waits for
an active call past its deadline, the scheduler reports ``UNKNOWN`` and marks
the bus quarantined.  It never turns a driver's ``None`` return value into
physical stop feedback, and it never invokes a serialized stop concurrently
with an active call.  A caller may explicitly opt into the existing
preemptive path for drivers that document that operation as safe (FR3 uses
that path through :class:`RobotAdapterCommandPort`).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class IOStatus(str, Enum):
    """Outcome of one scheduled driver transaction."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IOResult:
    """Result with dispatch and feedback facts kept separate.

    ``driver_returned`` means only that the Python driver call returned.  The
    return value is kept in ``driver_value`` and mirrored in ``feedback`` for
    callers that use observation or stop feedback.  A ``None`` stop return
    therefore leaves ``stop_confirmed`` as ``None`` instead of claiming that
    the robot is stationary.
    """

    operation: str
    status: IOStatus
    requested: Any = None
    driver_returned: bool = False
    driver_value: Any = None
    feedback: Any = None
    stop_confirmed: bool | None = None
    error: BaseException | None = None
    elapsed_s: float = 0.0
    quarantined: bool = False

    @property
    def stop_unconfirmed(self) -> bool:
        """Whether this stop lacks explicit physical confirmation."""

        return self.operation == "stop" and self.stop_confirmed is not True


class IOOperationError(RuntimeError):
    """A scheduled driver transaction failed or was rejected."""

    def __init__(self, result: IOResult):
        self.result = result
        detail = str(result.error) if result.error is not None else result.status.value
        super().__init__(f"{result.operation} I/O {result.status.value}: {detail}")


class IOUnknownError(IOOperationError):
    """The scheduler could not establish whether a transaction completed."""


@dataclass(slots=True)
class _Request:
    operation: str
    callback: Callable[[], Any]
    requested: Any
    cancel_event: threading.Event | None
    deadline_s: float | None
    done: threading.Event
    started: bool = False
    cancelled: bool = False
    detached: bool = False
    result: IOResult | None = None


class RobotIOScheduler:
    """Serialize transactions for one physical bus with bounded admission.

    ``execute`` and ``read`` are queued behind a stop request.  Only a small
    bounded queue is admitted, and at most ``max_pending_reads`` reads may be
    waiting at once, so a telemetry producer cannot create an unbounded
    backlog.  A stop request is always inserted at the front of the queue.

    The scheduler does not retry calls.  When a serialized call is already in
    an SDK function, a stop caller waits only ``stop_timeout_s`` and then gets
    ``UNKNOWN`` while the bus remains quarantined until the old call and its
    queued stop settle.
    """

    def __init__(
        self,
        bus_id: str,
        *,
        stop_timeout_s: float = 0.25,
        max_pending_operations: int = 32,
        max_pending_reads: int = 1,
        allow_preemptive_stop: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(bus_id, str) or not bus_id:
            raise ValueError("bus_id must be a non-empty string")
        self._validate_timeout(stop_timeout_s, "stop_timeout_s")
        if max_pending_operations < 1:
            raise ValueError("max_pending_operations must be positive")
        if max_pending_reads < 0:
            raise ValueError("max_pending_reads must be non-negative")

        self.bus_id = bus_id
        self.stop_timeout_s = stop_timeout_s
        self.max_pending_operations = max_pending_operations
        self.max_pending_reads = max_pending_reads
        self.allow_preemptive_stop = allow_preemptive_stop
        self._monotonic = monotonic
        self._condition = threading.Condition()
        self._queue: deque[_Request] = deque()
        self._queued_reads = 0
        self._active: _Request | None = None
        self._queued_stop: _Request | None = None
        self._closed = False
        self._quarantined = False
        self._last_stop_result: IOResult | None = None
        self._stop_lock = threading.Lock()
        self._preemptive_active = False
        self._worker = threading.Thread(
            target=self._run,
            name=f"rlinf-device-io-{bus_id}",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _validate_timeout(value: float | None, name: str) -> None:
        if value is None:
            return
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    def snapshot(self) -> dict[str, Any]:
        """Return bounded scheduler state for diagnostics and tests."""

        with self._condition:
            active = self._active
            return {
                "bus_id": self.bus_id,
                "active_operation": active.operation if active is not None else None,
                "queued_operations": len(self._queue),
                "queued_reads": self._queued_reads,
                "quarantined": self._quarantined,
                "closed": self._closed,
                "last_stop": self._last_stop_result,
            }

    def recover(self, *, stop_confirmed: bool) -> bool:
        """Clear quarantine only after an explicit stop confirmation.

        A late SDK return or a ``None`` driver value is not confirmation.  The
        caller must supply an observed/driver-confirmed ``True`` and the bus
        must have no active transaction.
        """

        with self._condition:
            if not stop_confirmed or self._active is not None or self._preemptive_active:
                return False
            if not self._quarantined:
                return True
            last = self._last_stop_result
            if last is None or last.stop_confirmed is not True:
                return False
            self._quarantined = False
            self._condition.notify_all()
            return True

    def execute(
        self,
        callback: Callable[[], Any],
        *,
        requested: Any = None,
        cancel_event: threading.Event | None = None,
        deadline_s: float | None = None,
        timeout_s: float | None = None,
        source: str = "automatic",
    ) -> IOResult:
        """Schedule one action transaction.

        ``source`` is diagnostic provenance only; authority remains in the
        control arbiter.  It is included in the requested record without
        changing the driver's action representation.
        """

        return self._submit(
            "execute",
            callback,
            requested={"source": source, "action": requested},
            cancel_event=cancel_event,
            deadline_s=deadline_s,
            timeout_s=timeout_s,
            priority=False,
        )

    def read(
        self,
        callback: Callable[[], Any],
        *,
        requested: Any = None,
        deadline_s: float | None = None,
        timeout_s: float | None = None,
        source: str = "observation",
    ) -> IOResult:
        """Schedule one observation/telemetry transaction."""

        return self._submit(
            "read",
            callback,
            requested={"source": source, "request": requested},
            cancel_event=None,
            deadline_s=deadline_s,
            timeout_s=timeout_s,
            priority=False,
        )

    def stop(
        self,
        callback: Callable[[], Any],
        *,
        requested: Any = None,
        timeout_s: float | None = None,
        preemptive: bool | None = None,
    ) -> IOResult:
        """Request a hold/stop, ahead of queued reads and actions."""

        use_preemptive = self.allow_preemptive_stop if preemptive is None else preemptive
        if use_preemptive:
            return self._preemptive_stop(callback, requested=requested, timeout_s=timeout_s)
        return self._submit(
            "stop",
            callback,
            requested=requested,
            cancel_event=None,
            deadline_s=None,
            timeout_s=self.stop_timeout_s if timeout_s is None else timeout_s,
            priority=True,
        )

    def _submit(
        self,
        operation: str,
        callback: Callable[[], Any],
        *,
        requested: Any,
        cancel_event: threading.Event | None,
        deadline_s: float | None,
        timeout_s: float | None,
        priority: bool,
    ) -> IOResult:
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._validate_timeout(timeout_s, "timeout_s")
        if deadline_s is not None and (not isinstance(deadline_s, (int, float)) or not math.isfinite(deadline_s)):
            raise ValueError("deadline_s must be finite")

        now = self._monotonic()
        if deadline_s is not None and isinstance(deadline_s, bool):
            raise ValueError("deadline_s must be finite")
        if deadline_s is not None and now >= deadline_s:
            return self._immediate(operation, IOStatus.EXPIRED, requested=requested)

        request = _Request(
            operation=operation,
            callback=callback,
            requested=requested,
            cancel_event=cancel_event,
            deadline_s=deadline_s,
            done=threading.Event(),
        )
        with self._condition:
            if self._closed:
                return self._immediate(operation, IOStatus.REJECTED, requested=requested)
            if operation != "stop" and self._preemptive_active:
                return self._immediate(
                    operation,
                    IOStatus.UNKNOWN,
                    requested=requested,
                    quarantined=self._quarantined,
                )
            if self._quarantined and operation != "stop":
                return self._immediate(
                    operation,
                    IOStatus.UNKNOWN,
                    requested=requested,
                    quarantined=True,
                )
            if operation == "read" and self._queued_reads >= self.max_pending_reads:
                return self._immediate(operation, IOStatus.REJECTED, requested=requested)
            if operation == "stop" and (
                self._queued_stop is not None or (self._active is not None and self._active.operation == "stop")
            ):
                # One pending stop is enough to preserve priority.  Repeated
                # callers receive a bounded answer instead of growing a stop
                # queue while the old SDK call is blocked.
                return self._immediate(operation, IOStatus.REJECTED, requested=requested)
            if operation != "stop" and len(self._queue) >= self.max_pending_operations:
                return self._immediate(operation, IOStatus.REJECTED, requested=requested)
            if operation == "read":
                self._queued_reads += 1
            if operation == "stop":
                self._queued_stop = request
            if priority:
                self._queue.appendleft(request)
            else:
                self._queue.append(request)
            self._condition.notify()

        wait_timeout = timeout_s
        if wait_timeout is None and deadline_s is not None:
            wait_timeout = max(0.0, deadline_s - self._monotonic())
        if request.done.wait(wait_timeout):
            assert request.result is not None
            return request.result

        # The callback may already be inside an SDK function.  Python cannot
        # kill that call, so preserve the queue ordering and report whether the
        # caller received a bounded answer instead of inventing completion.
        with self._condition:
            if not request.started and operation != "stop":
                request.cancelled = True
                if request in self._queue:
                    self._queue.remove(request)
                    if operation == "read":
                        self._queued_reads -= 1
                status = IOStatus.CANCELLED if cancel_event is not None else IOStatus.UNKNOWN
                return self._immediate(operation, status, requested=requested)
            if request.started and operation != "stop":
                # An in-flight motion or read may have reached the driver,
                # and Python cannot determine whether the SDK completed it.
                # Quarantine the bus and discard queued motion until an
                # explicit confirmed stop/recovery is supplied.
                self._quarantined = True
                self._drop_queued_motion_locked()
                request.detached = True
                return self._immediate(
                    operation,
                    IOStatus.UNKNOWN,
                    requested=requested,
                    quarantined=True,
                )
            if operation == "stop":
                self._quarantined = True
                self._drop_queued_motion_locked()
                request.detached = True
            return self._immediate(
                operation,
                IOStatus.UNKNOWN,
                requested=requested,
                quarantined=self._quarantined,
            )

    def _preemptive_stop(
        self,
        callback: Callable[[], Any],
        *,
        requested: Any,
        timeout_s: float | None,
    ) -> IOResult:
        self._validate_timeout(timeout_s, "timeout_s")
        wait_timeout = self.stop_timeout_s if timeout_s is None else timeout_s
        started = self._monotonic()
        with self._stop_lock:
            with self._condition:
                if self._closed:
                    return self._immediate("stop", IOStatus.REJECTED, requested=requested)
                if self._preemptive_active:
                    return self._immediate("stop", IOStatus.REJECTED, requested=requested)
                if self._queued_stop is not None:
                    return self._immediate("stop", IOStatus.REJECTED, requested=requested)
                self._preemptive_active = True

            done = threading.Event()
            holder: dict[str, Any] = {}

            def invoke() -> None:
                try:
                    holder["value"] = callback()
                except BaseException as exc:  # preserve driver exception for caller
                    holder["error"] = exc
                finally:
                    with self._condition:
                        self._preemptive_active = False
                        self._condition.notify_all()
                    done.set()

            threading.Thread(
                target=invoke,
                name=f"rlinf-device-preemptive-stop-{self.bus_id}",
                daemon=True,
            ).start()
            if not done.wait(wait_timeout):
                with self._condition:
                    self._quarantined = True
                    self._drop_queued_motion_locked()
                result = IOResult(
                    operation="stop",
                    status=IOStatus.UNKNOWN,
                    requested=requested,
                    elapsed_s=self._monotonic() - started,
                    stop_confirmed=None,
                    quarantined=True,
                )
                self._last_stop_result = result
                return result

            error = holder.get("error")
            value = holder.get("value")
            result = IOResult(
                operation="stop",
                status=IOStatus.FAILED if error is not None else IOStatus.COMPLETED,
                requested=requested,
                driver_returned=error is None,
                driver_value=value,
                feedback=value,
                stop_confirmed=self._stop_confirmation(value),
                error=error,
                elapsed_s=self._monotonic() - started,
            )
            with self._condition:
                if error is not None or result.stop_confirmed is False:
                    self._quarantined = True
                    self._drop_queued_motion_locked()
                result = replace(result, quarantined=self._quarantined)
                self._last_stop_result = result
            return result

    def _run(self) -> None:
        while True:
            with self._condition:
                while (not self._queue or self._preemptive_active) and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                request = self._queue.popleft()
                if request.operation == "read":
                    self._queued_reads -= 1
                if request is self._queued_stop:
                    self._queued_stop = None
                if request.cancelled:
                    request.result = self._immediate(
                        request.operation,
                        IOStatus.CANCELLED,
                        requested=request.requested,
                    )
                    request.done.set()
                    continue
                self._active = request
                request.started = True

            started = self._monotonic()
            result: IOResult
            if request.deadline_s is not None and started >= request.deadline_s:
                result = IOResult(
                    operation=request.operation,
                    status=IOStatus.EXPIRED,
                    requested=request.requested,
                    elapsed_s=self._monotonic() - started,
                )
            elif request.cancel_event is not None and request.cancel_event.is_set():
                result = IOResult(
                    operation=request.operation,
                    status=IOStatus.CANCELLED,
                    requested=request.requested,
                    elapsed_s=self._monotonic() - started,
                )
            else:
                try:
                    value = request.callback()
                except BaseException as exc:  # driver failure belongs to result
                    result = IOResult(
                        operation=request.operation,
                        status=IOStatus.FAILED,
                        requested=request.requested,
                        error=exc,
                        elapsed_s=self._monotonic() - started,
                    )
                else:
                    result = IOResult(
                        operation=request.operation,
                        status=IOStatus.COMPLETED,
                        requested=request.requested,
                        driver_returned=True,
                        driver_value=value,
                        feedback=value,
                        stop_confirmed=(self._stop_confirmation(value) if request.operation == "stop" else None),
                        elapsed_s=self._monotonic() - started,
                    )

            with self._condition:
                self._active = None
                if request.operation == "stop":
                    with_quarantine = self._quarantined
                    if result.status is IOStatus.FAILED or result.stop_confirmed is False:
                        self._quarantined = True
                    result = replace(
                        result,
                        quarantined=self._quarantined or with_quarantine,
                    )
                    self._last_stop_result = result
                elif result.status == IOStatus.FAILED:
                    # A driver failure leaves the bus usable for an explicit
                    # stop request, but does not silently retry the action.
                    pass
                request.result = result
                self._condition.notify_all()
                if not request.detached:
                    request.done.set()

    @staticmethod
    def _stop_confirmation(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            confirmed = value.get("stop_confirmed")
            return confirmed if isinstance(confirmed, bool) else None
        return None

    @staticmethod
    def _immediate(
        operation: str,
        status: IOStatus,
        *,
        requested: Any = None,
        quarantined: bool = False,
    ) -> IOResult:
        return IOResult(
            operation=operation,
            status=status,
            requested=requested,
            stop_confirmed=None if operation == "stop" else None,
            quarantined=quarantined,
        )

    def _drop_queued_motion_locked(self) -> None:
        """Drop queued reads/actions after a stop wait becomes uncertain."""

        retained: deque[_Request] = deque()
        while self._queue:
            request = self._queue.popleft()
            if request.operation == "stop":
                retained.append(request)
                continue
            if request.operation == "read":
                self._queued_reads -= 1
            request.cancelled = True
            request.result = self._immediate(
                request.operation,
                IOStatus.UNKNOWN,
                requested=request.requested,
                quarantined=True,
            )
            request.done.set()
        self._queue = retained

    def close(self, *, wait_s: float = 0.1) -> IOResult:
        """Bound shutdown without claiming that an in-flight call stopped.

        Queued motion/read work is discarded, but the single pending stop is
        retained and allowed to drain.  If the active SDK call or that stop is
        still in flight after ``wait_s``, the result is UNKNOWN and the owner
        must retain device ownership/quarantine.
        """

        self._validate_timeout(wait_s, "wait_s")
        with self._condition:
            if self._closed:
                active = self._active
                drained = active is None and not self._queue and not self._preemptive_active
                return IOResult(
                    operation="close",
                    status=IOStatus.COMPLETED if drained else IOStatus.UNKNOWN,
                    requested="close",
                    quarantined=not drained or self._quarantined,
                )
            self._closed = True
            retained: deque[_Request] = deque()
            while self._queue:
                request = self._queue.popleft()
                if request.operation == "stop":
                    retained.append(request)
                    continue
                if request.operation == "read":
                    self._queued_reads -= 1
                request.cancelled = True
                request.result = self._immediate(
                    request.operation,
                    IOStatus.CANCELLED,
                    requested=request.requested,
                )
                request.done.set()
            self._queue = retained
            self._condition.notify_all()
        self._worker.join(wait_s)
        with self._condition:
            active = self._active
            drained = active is None and not self._queue and not self._preemptive_active
            return IOResult(
                operation="close",
                status=IOStatus.COMPLETED if drained else IOStatus.UNKNOWN,
                requested="close",
                quarantined=not drained or self._quarantined,
            )


__all__ = [
    "IOOperationError",
    "IOResult",
    "IOStatus",
    "IOUnknownError",
    "RobotIOScheduler",
]
