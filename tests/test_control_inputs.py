from __future__ import annotations

import os
import struct

import pytest

from embodirun.robots import RobotObservation
from embodirun.robots.arx.x5 import ARX5_ACTION_SPACE
from embodirun.robots.lerobot.so101 import SO101_ACTION_SPACE
from embodirun.services.control import teleop as teleop_module
from embodirun.services.control.inputs import (
    ControlInputBridge,
    InputEvent,
    InputEventKind,
    InputMonitorError,
    InputPollResult,
    JoystickInput,
    KeyboardInput,
    normalize_axis,
)
from embodirun.services.control.teleop import (
    TELEOP_AXES_ACTION_SPACE,
    ControlHttpTeleopClient,
    axes_action,
    intent_action_factory,
    resolve_teleop_action,
)


class FakeArbiter:
    def __init__(self, authority: str = "model") -> None:
        self.authority = authority
        self.deadman_active = False
        self.last_error = None
        self.calls: list[tuple[str, object]] = []

    def snapshot(self) -> dict[str, object]:
        return {
            "authority": self.authority,
            "deadman_active": self.deadman_active,
            "last_error": self.last_error,
        }

    def acquire_manual(self) -> None:
        self.calls.append(("acquire_manual", None))
        self.authority = "manual"

    def release_manual(self) -> None:
        self.calls.append(("release_manual", None))
        self.authority = "model"
        self.deadman_active = False

    def set_deadman(self, active: bool) -> None:
        self.calls.append(("set_deadman", active))
        self.deadman_active = active

    def submit_manual(self, action, *, wait=False):
        self.calls.append(("submit_manual", action))
        assert wait is False

    def emergency_stop(self) -> None:
        self.calls.append(("emergency_stop", None))
        self.authority = "estop_latched"
        self.deadman_active = False

    def reset_emergency_stop(self) -> None:
        self.calls.append(("reset_emergency_stop", None))
        self.authority = "manual"
        self.deadman_active = False

    def hold(self) -> None:
        self.calls.append(("hold", None))


class HoldFailingArbiter(FakeArbiter):
    def hold(self) -> None:
        self.calls.append(("hold", None))
        raise RuntimeError("hold failed")


class FakeSource:
    def __init__(self, *results: InputPollResult) -> None:
        self.results = list(results)
        self.closed = 0

    def poll(self, _timeout_s: float = 0.0) -> InputPollResult:
        return self.results.pop(0) if self.results else InputPollResult()

    def close(self) -> None:
        self.closed += 1


class FakeRobot:
    def __init__(
        self,
        values: dict[str, object],
        action_space: str,
        *,
        robot_kind: str | None = None,
    ) -> None:
        self.values = values
        self.action_space = action_space
        self.config = type("Config", (), {"robot_kind": robot_kind})()

    def observe(self) -> RobotObservation:
        return RobotObservation(
            timestamp_s=1.0,
            values=self.values,
            metadata={"action_space": self.action_space},
        )


def _event(kind: InputEventKind, *, control=None, value=None) -> InputEvent:
    return InputEvent(kind=kind, control=control, value=value)


def _js_packet(event_type: int, number: int, value: int) -> bytes:
    return struct.pack("<IhBB", 123, value, event_type, number)


def test_joystick_initial_buttons_cannot_acquire_or_enable_motion() -> None:
    packets = b"".join(_js_packet(0x81, button, 1) for button in (0, 1, 3, 4))
    source = JoystickInput(
        fd=42,
        select_fn=lambda *args: ([42], [], []),
        read_fn=lambda *args: packets,
    )
    assert [event.kind for event in source.poll().events] == [InputEventKind.ESTOP]


def test_keyboard_parser_exposes_estop_reset_and_quit_only() -> None:
    events = KeyboardInput.events_from_bytes(b" rRqQx")

    assert [event.kind for event in events] == [
        InputEventKind.ESTOP,
        InputEventKind.RESET_ESTOP,
        InputEventKind.RESET_ESTOP,
        InputEventKind.QUIT,
        InputEventKind.QUIT,
    ]
    assert all(event.source == "keyboard" for event in events)


def test_keyboard_poll_is_safe_for_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    keyboard = KeyboardInput(
        fd=0,
        select_fn=lambda readers, _w, _e, _timeout: (list(readers), [], []),
        read_fn=lambda _fd, _size: b" ",
    )

    result = keyboard.poll()

    assert result.ok
    assert [event.kind for event in result.events] == [InputEventKind.ESTOP]


def test_joystick_supports_split_packets_and_deadman_heartbeat() -> None:
    chunks = iter(
        (
            _js_packet(JoystickInput.JS_EVENT_BUTTON, 0, 1)[:3],
            _js_packet(JoystickInput.JS_EVENT_BUTTON, 0, 1)[3:] + _js_packet(JoystickInput.JS_EVENT_BUTTON, 4, 1),
        )
    )
    joystick = JoystickInput(
        fd=0,
        read_fn=lambda _fd, _size: next(chunks),
        select_fn=lambda readers, _w, _e, _timeout: (list(readers), [], []),
    )

    assert joystick.poll().events == ()
    second = joystick.poll()
    joystick._select = lambda _readers, _w, _e, _timeout: ([], [], [])
    third = joystick.poll()

    assert [event.kind for event in second.events] == [
        InputEventKind.ACQUIRE,
        InputEventKind.DEADMAN,
    ]
    assert third.events == (InputEvent(InputEventKind.DEADMAN, value=True, source="joystick"),)
    assert normalize_axis(0.05) == 0.0


def test_bridge_prioritizes_estop_and_submits_latest_cpu_intent() -> None:
    arbiter = FakeArbiter()
    source = FakeSource(
        InputPollResult(
            events=(
                _event(InputEventKind.ACQUIRE),
                _event(InputEventKind.DEADMAN, value=True),
                _event(InputEventKind.AXIS, control="left_x", value=0.2),
                _event(InputEventKind.AXIS, control="right_x", value=-0.4),
            )
        ),
        InputPollResult(
            events=(
                _event(InputEventKind.ESTOP),
                _event(InputEventKind.AXIS, control="left_x", value=1.0),
            )
        ),
    )
    bridge = ControlInputBridge(
        arbiter,
        action_factory=intent_action_factory("arx5"),
        joystick=source,
    )

    bridge.poll_once()
    bridge.poll_once()

    assert [name for name, _value in arbiter.calls] == [
        "acquire_manual",
        "set_deadman",
        "submit_manual",
        "emergency_stop",
    ]
    action = arbiter.calls[2][1]
    assert action.metadata == {
        "action_space": TELEOP_AXES_ACTION_SPACE,
        "robot_kind": "arx.x5",
    }
    assert action.values["axes"] == {"left_x": 0.2, "right_x": -0.4}


def test_bridge_failure_holds_and_preserves_hold_error_context() -> None:
    bridge = ControlInputBridge(
        HoldFailingArbiter(authority="manual"),
        action_factory=axes_action,
        joystick=FakeSource(InputPollResult(error="device read failed")),
    )

    result = bridge.poll_once()

    assert not result.ok
    with pytest.raises(InputMonitorError) as captured:
        bridge.raise_for_failure()
    assert str(captured.value.source_error) == "device read failed"
    assert str(captured.value.hold_error) == "hold failed"


def test_bridge_polls_last_error_and_surfaces_execution_failure() -> None:
    arbiter = FakeArbiter(authority="manual")
    arbiter.last_error = "execute failed"
    bridge = ControlInputBridge(
        arbiter,
        action_factory=axes_action,
        joystick=FakeSource(InputPollResult()),
    )

    result = bridge.poll_once()

    assert not result.ok
    with pytest.raises(InputMonitorError, match="execute failed"):
        bridge.raise_for_failure()
    assert [name for name, _value in arbiter.calls] == []


def test_http_teleop_client_matches_control_service_routes() -> None:
    calls: list[tuple[str, str, object]] = []
    state = {"authority": "manual", "deadman_active": True, "last_error": None}

    def transport(method: str, path: str, payload):
        calls.append((method, path, payload))
        return dict(state)

    client = ControlHttpTeleopClient("http://127.0.0.1:8100", transport=transport)
    client.acquire_manual()
    client.set_deadman(True)
    client.submit_manual(intent_action_factory("so101")({"left_x": 0.5}))
    client.release_manual()
    client.reset_emergency_stop()
    client.emergency_stop()

    assert [call[:2] for call in calls] == [
        ("POST", "/v1/control/manual/acquire"),
        ("POST", "/v1/control/manual/deadman"),
        ("POST", "/v1/control/manual/action"),
        ("POST", "/v1/control/manual/release"),
        ("POST", "/v1/control/reset"),
        ("POST", "/v1/control/emergency-stop"),
    ]
    assert calls[2][2]["metadata"]["robot_kind"] == "lerobot.so101"


def test_resolve_teleop_action_maps_arx5_axes_from_robot_namespace() -> None:
    robot = FakeRobot(
        {"eef_xyzrpy_gripper": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.04]},
        ARX5_ACTION_SPACE,
    )

    resolved = resolve_teleop_action(
        robot,
        axes_action(
            {"left_x": 2.0, "left_y": -2.0, "rt": 2.0, "right_x": 0.5},
            robot_kind="arx5",
            timestamp_s=2.0,
        ),
    )

    assert resolved.metadata["action_space"] == ARX5_ACTION_SPACE
    assert resolved.values == {
        "type": "eef_xyzrpy_gripper",
        "eef_xyzrpy_gripper": [1.01, 1.99, 3.01, 0.1, 0.2, 0.32, 0.04],
    }


def test_resolve_teleop_action_maps_so101_axes_from_robot_namespace() -> None:
    robot = FakeRobot(
        {"joint_positions_deg": [0, 10, 20, 30, 40], "gripper_position": 50},
        SO101_ACTION_SPACE,
    )

    resolved = resolve_teleop_action(
        robot,
        axes_action(
            {"left_x": 1.0, "right_y": -0.5, "dpad_y": 1.0},
            robot_kind="so101",
            timestamp_s=2.0,
        ),
    )

    assert resolved.metadata["action_space"] == SO101_ACTION_SPACE
    assert resolved.values == {
        "type": "joint_position",
        "joint_positions_deg": [2.0, 10.0, 20.0, 29.0, 40.0],
        "gripper_position": 52.0,
    }


def test_resolve_teleop_action_rejects_metadata_namespace_mismatch() -> None:
    robot = FakeRobot(
        {"joint_positions_deg": [0, 10, 20, 30, 40], "gripper_position": 50},
        SO101_ACTION_SPACE,
    )

    with pytest.raises(ValueError, match="arx.x5 teleop cannot target"):
        resolve_teleop_action(robot, axes_action({}, robot_kind="arx5"))


def test_so101_teleop_clamps_gripper_to_adapter_range() -> None:
    robot = FakeRobot(
        {"joint_positions_deg": [0, 10, 20, 30, 40], "gripper_position": 99},
        SO101_ACTION_SPACE,
    )

    resolved = resolve_teleop_action(
        robot,
        axes_action({"dpad_y": 3.0}, robot_kind="so101", timestamp_s=2.0),
    )

    assert resolved.values["gripper_position"] == 100.0


def test_cli_keyboard_only_does_not_require_robot_kind(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class Keyboard:
        def close(self) -> None:
            calls.append("keyboard.close")

    class Bridge:
        def __init__(self, client, *, action_factory, keyboard, joystick):
            calls.append("bridge.init")
            assert keyboard is not None
            assert joystick is None
            self._first = True

        def snapshot(self):
            if self._first:
                self._first = False
                return {"running": False, "last_error": None}
            return {"running": False, "last_error": None}

        def poll_once(self, _timeout_s):
            calls.append("poll_once")
            return InputPollResult()

        def raise_for_failure(self):
            return None

        def close(self):
            calls.append("bridge.close")

    monkeypatch.setattr("embodirun.services.control.inputs.KeyboardInput", Keyboard)
    monkeypatch.setattr("embodirun.services.control.inputs.ControlInputBridge", Bridge)

    assert teleop_module.main(["--endpoint", "http://127.0.0.1:8100"]) == 0

    assert calls == ["bridge.init", "bridge.close", "keyboard.close"]
    assert "space=emergency-stop" in capsys.readouterr().err


def test_cli_closes_keyboard_if_joystick_initialization_fails(monkeypatch) -> None:
    calls: list[str] = []

    class Keyboard:
        def close(self) -> None:
            calls.append("keyboard.close")

    class Joystick:
        def __init__(self, _path):
            raise RuntimeError("joystick missing")

    monkeypatch.setattr("embodirun.services.control.inputs.KeyboardInput", Keyboard)
    monkeypatch.setattr("embodirun.services.control.inputs.JoystickInput", Joystick)

    with pytest.raises(RuntimeError, match="joystick missing"):
        teleop_module.main(
            [
                "--endpoint",
                "http://127.0.0.1:8100",
                "--robot-kind",
                "arx5",
                "--joystick",
                "/dev/input/js0",
            ]
        )

    assert calls == ["keyboard.close"]


def test_cli_joystick_requires_explicit_robot_kind(monkeypatch, capsys) -> None:
    class Keyboard:
        def __init__(self):
            raise AssertionError("inputs must not initialize after parser error")

    monkeypatch.setattr("embodirun.services.control.inputs.KeyboardInput", Keyboard)

    with pytest.raises(SystemExit) as captured:
        teleop_module.main(["--endpoint", "http://127.0.0.1:8100", "--joystick", "/dev/input/js0"])

    assert captured.value.code == 2
    assert "--joystick requires --robot-kind" in capsys.readouterr().err
