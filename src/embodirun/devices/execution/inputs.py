"""Non-blocking input events for control-service arbitration."""

from __future__ import annotations

import enum
import os
import select
import struct
import sys
import termios
import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol


class InputEventKind(str, enum.Enum):
    ACQUIRE = "acquire"
    RELEASE = "release"
    DEADMAN = "deadman"
    AXIS = "axis"
    ESTOP = "estop"
    RESET_ESTOP = "reset_estop"
    QUIT = "quit"


@dataclass(frozen=True, slots=True)
class InputEvent:
    kind: InputEventKind
    control: str | None = None
    value: object = None
    source: str = "input"


@dataclass(frozen=True, slots=True)
class InputPollResult:
    events: tuple[InputEvent, ...] = ()
    error: Exception | str | None = None
    eof: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.eof


class InputSource(Protocol):
    def poll(self, timeout_s: float = 0.0) -> InputPollResult: ...

    def close(self) -> None: ...


class InputMonitorError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        source_error: Exception | str | None = None,
        hold_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.source_error = source_error
        self.hold_error = hold_error
        if isinstance(source_error, Exception):
            self.__cause__ = source_error
        elif hold_error is not None:
            self.__cause__ = hold_error


def normalize_axis(value: float, deadzone: float = 0.1) -> float:
    if value > 1.0:
        value = 1.0
    elif value < -1.0:
        value = -1.0
    return 0.0 if abs(value) < deadzone else value


class KeyboardInput:
    def __init__(
        self,
        fd: int | None = None,
        *,
        configure_terminal: bool = True,
        select_fn: Callable[..., tuple[list[int], list[int], list[int]]] = select.select,
        read_fn: Callable[[int, int], bytes] = os.read,
    ) -> None:
        self.fd = sys.stdin.fileno() if fd is None else fd
        self._select = select_fn
        self._read = read_fn
        self._saved_attrs: list[int | bytes] | None = None
        self._closed = False
        if configure_terminal and os.isatty(self.fd):
            self._saved_attrs = termios.tcgetattr(self.fd)
            attrs = termios.tcgetattr(self.fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, attrs)

    @staticmethod
    def events_from_bytes(data: bytes) -> tuple[InputEvent, ...]:
        events: list[InputEvent] = []
        for byte in data:
            char = chr(byte)
            if char == " ":
                events.append(InputEvent(InputEventKind.ESTOP, source="keyboard"))
            elif char in {"r", "R"}:
                events.append(InputEvent(InputEventKind.RESET_ESTOP, source="keyboard"))
            elif char in {"q", "Q"}:
                events.append(InputEvent(InputEventKind.QUIT, source="keyboard"))
        return tuple(events)

    def poll(self, timeout_s: float = 0.0) -> InputPollResult:
        try:
            ready, _, _ = self._select([self.fd], [], [], timeout_s)
            if not ready:
                return InputPollResult()
            data = self._read(self.fd, 32)
        except Exception as error:
            return InputPollResult(error=error)
        if data == b"":
            return InputPollResult(eof=True)
        return InputPollResult(events=self.events_from_bytes(data))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._saved_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved_attrs)


class JoystickInput:
    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80

    _BUTTONS = {
        0: InputEventKind.ACQUIRE,
        1: InputEventKind.ESTOP,
        3: InputEventKind.RELEASE,
    }
    _AXES = {
        0: "left_x",
        1: "left_y",
        3: "right_x",
        4: "right_y",
        2: "lt",
        5: "rt",
        6: "dpad_x",
        7: "dpad_y",
    }

    def __init__(
        self,
        path: str | None = None,
        *,
        fd: int | None = None,
        deadzone: float = 0.1,
        select_fn: Callable[..., tuple[list[int], list[int], list[int]]] = select.select,
        read_fn: Callable[[int, int], bytes] = os.read,
    ) -> None:
        if fd is None and path is None:
            path = "/dev/input/js0"
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK) if fd is None else fd
        self._owns_fd = fd is None
        self._deadzone = deadzone
        self._select = select_fn
        self._read = read_fn
        self._buffer = bytearray()
        self._deadman = False
        self._has_deadman = False
        self._closed = False

    def poll(self, timeout_s: float = 0.0) -> InputPollResult:
        try:
            ready, _, _ = self._select([self.fd], [], [], timeout_s)
            if ready:
                data = self._read(self.fd, 256)
                if data == b"":
                    return InputPollResult(eof=True)
                self._buffer.extend(data)
        except BlockingIOError:
            pass
        except Exception as error:
            return InputPollResult(error=error)

        events: list[InputEvent] = []
        while len(self._buffer) >= 8:
            _time_ms, value, event_type, number = struct.unpack("<IhBB", self._buffer[:8])
            del self._buffer[:8]
            base_type = event_type & ~self.JS_EVENT_INIT
            if event_type & self.JS_EVENT_INIT and base_type == self.JS_EVENT_BUTTON:
                # Device-open snapshots are not deliberate button presses.
                # Keep an already-pressed emergency button effective, but never
                # acquire control or arm the deadman from an initial snapshot.
                if value and self._BUTTONS.get(number) is InputEventKind.ESTOP:
                    events.append(InputEvent(InputEventKind.ESTOP, source="joystick"))
                continue
            if base_type == self.JS_EVENT_BUTTON and value:
                kind = self._BUTTONS.get(number)
                if kind is not None:
                    events.append(InputEvent(kind, source="joystick"))
            if base_type == self.JS_EVENT_BUTTON and number == 4:
                self._deadman = bool(value)
                self._has_deadman = True
                events.append(self._event(InputEventKind.DEADMAN, self._deadman))
            elif base_type == self.JS_EVENT_AXIS:
                control = self._AXES.get(number)
                if control is not None:
                    axis = normalize_axis(value / 32767.0, self._deadzone)
                    events.append(InputEvent(InputEventKind.AXIS, control, axis, "joystick"))
        if self._has_deadman and not any(event.kind is InputEventKind.DEADMAN for event in events):
            events.append(self._event(InputEventKind.DEADMAN, self._deadman))
        return InputPollResult(events=tuple(events))

    @staticmethod
    def _event(kind: InputEventKind, value: object) -> InputEvent:
        return InputEvent(kind, value=value, source="joystick")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_fd:
            os.close(self.fd)


class ControlInputBridge:
    """Poll input sources and submit manual intent without observing the robot."""

    def __init__(
        self,
        arbiter,
        *,
        action_factory: Callable[[dict[str, float]], object],
        keyboard: InputSource | None = None,
        joystick: InputSource | None = None,
    ) -> None:
        self.arbiter = arbiter
        self._action_factory = action_factory
        self._sources = tuple(source for source in (keyboard, joystick) if source)
        self._axes: dict[str, float] = {}
        self._running = True
        self._closed = False
        self._lock = threading.Lock()
        self._failure: InputMonitorError | None = None

    def poll_once(self, timeout_s: float = 0.0) -> InputPollResult:
        remote_error = self._remote_error()
        if remote_error is not None:
            return self._fail(remote_error, False, hold=False)
        events: list[InputEvent] = []
        for index, source in enumerate(self._sources):
            result = source.poll(timeout_s if index == 0 else 0.0)
            if not result.ok:
                return self._fail(result.error or "input EOF", result.eof)
            events.extend(result.events)
        self._apply(events)
        remote_error = self._remote_error()
        if remote_error is not None:
            return self._fail(remote_error, False, hold=False)
        return InputPollResult(events=tuple(events))

    def raise_for_failure(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "running": self._running,
                "last_error": str(self._failure.source_error) if self._failure else None,
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary_error: BaseException | None = None
        try:
            self.arbiter.hold()
        except BaseException as error:
            primary_error = error
        try:
            with ExitStack() as stack:
                for source in self._sources:
                    stack.callback(source.close)
        except BaseException as error:
            if primary_error is None:
                primary_error = error
        if primary_error is not None:
            raise primary_error

    def close_sources(self) -> None:
        with ExitStack() as stack:
            for source in self._sources:
                stack.callback(source.close)

    def _fail(
        self,
        source_error: Exception | str,
        eof: bool,
        *,
        hold: bool = True,
    ) -> InputPollResult:
        hold_error: Exception | None = None
        if hold:
            try:
                self.arbiter.hold()
            except Exception as error:
                hold_error = error
        failure = InputMonitorError(
            f"input {'EOF' if eof else 'failure'}: {source_error}",
            source_error=source_error,
            hold_error=hold_error,
        )
        with self._lock:
            self._failure = failure
            self._running = False
        return InputPollResult(error=failure, eof=eof)

    def _apply(self, events: list[InputEvent]) -> None:
        if not events:
            return
        if any(event.kind is InputEventKind.ESTOP for event in events):
            self.arbiter.emergency_stop()
            self._axes.clear()
            return
        if any(event.kind is InputEventKind.QUIT for event in events):
            self.arbiter.hold()
            self._axes.clear()
            with self._lock:
                self._running = False
            return
        released_deadman = False
        for event in events:
            if event.kind is InputEventKind.ACQUIRE:
                self.arbiter.acquire_manual()
                self._axes.clear()
            elif event.kind is InputEventKind.RELEASE:
                if self._authority() == "manual":
                    self.arbiter.release_manual()
                self._axes.clear()
            elif event.kind is InputEventKind.RESET_ESTOP:
                self._axes.clear()
                if self._authority() == "estop_latched":
                    self.arbiter.reset_emergency_stop()
            elif event.kind is InputEventKind.DEADMAN:
                if self._authority() != "manual":
                    continue
                active = bool(event.value)
                self.arbiter.set_deadman(active)
                if not active:
                    self._axes.clear()
                    released_deadman = True
            elif event.kind is InputEventKind.AXIS and not released_deadman:
                if event.control is not None and self._authority() == "manual":
                    self._axes[event.control] = float(event.value)
        if not released_deadman and self._axes and self._ready_for_manual():
            action = self._action_factory(dict(self._axes))
            self.arbiter.submit_manual(action, wait=False)

    def _authority(self) -> object:
        return self.arbiter.snapshot().get("authority")

    def _ready_for_manual(self) -> bool:
        snapshot = self.arbiter.snapshot()
        return snapshot.get("authority") == "manual" and snapshot.get("deadman_active") is True

    def _remote_error(self) -> str | None:
        try:
            snapshot = self.arbiter.snapshot()
        except AttributeError:
            return None
        error = snapshot.get("last_error")
        return str(error) if error else None


__all__ = [
    "ControlInputBridge",
    "InputEvent",
    "InputEventKind",
    "InputMonitorError",
    "InputPollResult",
    "InputSource",
    "JoystickInput",
    "KeyboardInput",
    "normalize_axis",
]
