"""Service-native manual teleoperation HTTP client and intent resolver."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from embodirun.robots import RobotAction, RobotAdapter
from embodirun.robots.arx.x5.teleop import (
    resolve_axes_action as resolve_arx5_axes_action,
)
from embodirun.robots.lerobot.so101.teleop import (
    resolve_axes_action as resolve_so101_axes_action,
)

TELEOP_AXES_ACTION_SPACE = "rlinf.teleop.axes.v1"
_ROBOT_KIND_ALIASES = {
    "arx5": "arx.x5",
    "arx.x5": "arx.x5",
    "so101": "lerobot.so101",
    "lerobot.so101": "lerobot.so101",
}
_RESOLVERS = {
    "arx.x5": resolve_arx5_axes_action,
    "lerobot.so101": resolve_so101_axes_action,
}


def axes_action(
    axes: Mapping[str, float],
    *,
    robot_kind: str | None = None,
    timestamp_s: float | None = None,
) -> RobotAction:
    metadata = {"action_space": TELEOP_AXES_ACTION_SPACE}
    if robot_kind is not None:
        metadata["robot_kind"] = normalize_robot_kind(robot_kind)
    return RobotAction(
        timestamp_s=time.time() if timestamp_s is None else timestamp_s,
        values={"type": "teleop_axes", "axes": dict(axes)},
        metadata=metadata,
    )


def intent_action_factory(robot_kind: str) -> Callable[[dict[str, float]], RobotAction]:
    normalized = normalize_robot_kind(robot_kind)
    return lambda axes: axes_action(axes, robot_kind=normalized)


def resolve_teleop_action(robot: RobotAdapter, action: RobotAction) -> RobotAction:
    """Resolve an explicit teleop intent through the registered robot namespace."""

    if action.metadata.get("action_space") != TELEOP_AXES_ACTION_SPACE:
        return action
    if not isinstance(action.values, Mapping):
        raise ValueError("teleop action values must be an object")
    if action.values.get("type") != "teleop_axes":
        raise ValueError("teleop action type must be 'teleop_axes'")
    axes = action.values.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError("teleop action axes must be an object")
    robot_kind = _robot_kind_from_action_or_config(robot, action)
    if robot_kind is None:
        raise ValueError("teleop action metadata must include robot_kind")
    resolver = _RESOLVERS.get(robot_kind)
    if resolver is None:
        raise ValueError(f"unsupported teleop robot kind {robot_kind!r}")
    return resolver(robot.observe(), axes, timestamp_s=action.timestamp_s)


class ControlHttpTeleopClient:
    """Arbiter-shaped HTTP client for ControlInputBridge."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 2.0,
        transport: Callable[[str, str, Mapping[str, Any] | None], Mapping[str, Any]] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport

    def snapshot(self) -> Mapping[str, Any]:
        return self._request("GET", "/v1/control")

    def acquire_manual(self) -> None:
        self._request("POST", "/v1/control/manual/acquire")

    def release_manual(self) -> None:
        self._request("POST", "/v1/control/manual/release")

    def set_deadman(self, active: bool) -> None:
        self._request("POST", "/v1/control/manual/deadman", {"active": active})

    def submit_manual(self, action: RobotAction, *, wait: bool = False) -> None:
        del wait
        self._request("POST", "/v1/control/manual/action", _action_payload(action))

    def emergency_stop(self) -> None:
        self._request("POST", "/v1/control/emergency-stop")

    def reset_emergency_stop(self) -> None:
        self._request("POST", "/v1/control/reset")

    def hold(self) -> None:
        snapshot = self.snapshot()
        if snapshot.get("authority") == "manual":
            self.set_deadman(False)
        else:
            self.emergency_stop()

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if self._transport is not None:
            return self._transport(method, path, payload)
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                return _json_response(response.read())
        except HTTPError as error:
            detail = _json_response(error.read()).get("error", error.reason)
            raise RuntimeError(f"control service rejected {path}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"control service is unreachable: {error.reason}") from error


def normalize_robot_kind(value: str) -> str:
    try:
        return _ROBOT_KIND_ALIASES[value]
    except KeyError:
        raise ValueError(f"unsupported teleop robot kind {value!r}") from None


def main(argv: list[str] | None = None) -> int:
    from .inputs import ControlInputBridge, JoystickInput, KeyboardInput

    parser = argparse.ArgumentParser(
        prog="rlinf-control-teleop",
        description=(
            "Poll keyboard and optional joystick input and forward manual control "
            "requests to the loopback control service. Keyboard commands: space "
            "emergency-stops, r resets emergency stop, q releases/holds and exits. "
            "Joystick axes create CPU-only teleop intent actions."
        ),
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--robot-kind", choices=("arx5", "so101"))
    parser.add_argument("--joystick")
    parser.add_argument("--poll-timeout", type=float, default=0.02)
    parser.add_argument("--http-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.joystick and args.robot_kind is None:
        parser.error("--joystick requires --robot-kind")

    client = ControlHttpTeleopClient(args.endpoint, timeout_s=args.http_timeout)
    with ExitStack() as stack:
        keyboard = KeyboardInput()
        stack.callback(keyboard.close)
        joystick = None
        if args.joystick:
            joystick = JoystickInput(args.joystick)
            stack.callback(joystick.close)
        bridge = ControlInputBridge(
            client,
            action_factory=(
                intent_action_factory(args.robot_kind) if args.robot_kind is not None else _no_motion_action
            ),
            keyboard=keyboard,
            joystick=joystick,
        )
        stack.callback(bridge.close)
        print(
            "Keyboard: space=emergency-stop, r=reset, q=quit. Joystick axes submit manual motion intents.",
            file=sys.stderr,
        )
        while bridge.snapshot()["running"]:
            bridge.poll_once(args.poll_timeout)
            bridge.raise_for_failure()
    return 0


def _no_motion_action(_axes: dict[str, float]) -> RobotAction:
    raise RuntimeError("--robot-kind is required before submitting motion intent")


def _robot_kind_from_action_or_config(
    robot: RobotAdapter,
    action: RobotAction,
) -> str | None:
    value = action.metadata.get("robot_kind")
    if isinstance(value, str):
        return normalize_robot_kind(value)
    config = getattr(robot, "config", None)
    config_value = getattr(config, "robot_kind", None)
    if isinstance(config_value, str):
        return normalize_robot_kind(config_value)
    return None


def _action_payload(action: RobotAction) -> dict[str, Any]:
    if not isinstance(action.values, Mapping):
        raise ValueError("manual action values must be an object")
    return {
        "timestamp_s": action.timestamp_s,
        "values": dict(action.values),
        "metadata": dict(action.metadata),
    }


def _json_response(data: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("control service returned invalid JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError("control service returned a non-object JSON response")
    return value


__all__ = [
    "TELEOP_AXES_ACTION_SPACE",
    "ControlHttpTeleopClient",
    "axes_action",
    "intent_action_factory",
    "main",
    "normalize_robot_kind",
    "resolve_teleop_action",
]


if __name__ == "__main__":
    raise SystemExit(main())
