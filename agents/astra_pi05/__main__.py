"""Run the Astra-pi0.5 validator or a complete software-only HTTP demo.

Usage::

    PYTHONPATH=.:src python -m agents.astra_pi05 --validate-only
    PYTHONPATH=.:src python -m agents.astra_pi05 --demo

The demo starts a local fake action service and a real Control HTTP server. It
uses a demo-only twelve-joint simulated adapter and binding, so it never
imports a hardware SDK. The public flow is observe -> propose -> review ->
execute a three-row prefix -> observe again. The fake model and robot are
fixtures for API composition, not a real Astra model or physical device.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

from agents.astra_pi05.cooperative import CooperativeLoop
from agents.astra_pi05.decision import (
    ACTION_DIM,
    HORIZON,
    validate_decision,
    validate_proposal,
)
from embodirun.application.contracts import ControlServiceConfig
from embodirun.bindings import BindingDefinition, binding_definition
from embodirun.client import ControlClient
from embodirun.devices import DeviceManager
from embodirun.model_services import ImagePayload, PolicyObservation, PolicyResult
from embodirun.robots import (
    RobotAction,
    RobotAdapter,
    RobotDefinition,
    RobotObservation,
    robot_definition,
)
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.server import ControlHttpServer, ControlService


def _validate_only() -> dict[str, Any]:
    metadata = {
        "action_semantics": "biso101_so101_v1",
        "action_layout": "left6_right6",
        "action_encoding": "absolute",
        "joint_position_unit": "degrees",
        "action_dim": ACTION_DIM,
        "horizon": HORIZON,
    }
    proposal_id = "example-proposal"
    observation_id = "example-observation"
    actions = [[float(row)] * ACTION_DIM for row in range(HORIZON)]
    validate_proposal(actions, metadata)
    decision = validate_decision(
        {
            "proposal_id": proposal_id,
            "observation_id": observation_id,
            "decision": "execute_prefix",
            "execute_steps": 3,
            "corrections": [],
            "reason": "hardware-free example selects a short prefix",
        },
        proposal_id=proposal_id,
        observation_id=observation_id,
    )
    return {
        "status": "validated",
        "proposal_steps": len(actions),
        "selected_steps": decision["execute_steps"],
        "discarded_steps": len(actions) - decision["execute_steps"],
        "hardware_access": False,
    }


class _DemoActionHandler(BaseHTTPRequestHandler):
    """Fake SGLang-compatible model endpoint used only by ``--demo``."""

    def log_message(self, *_args: object) -> None:
        return None

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        self._send(200, {"status": "ok"} if self.path == "/health" else {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/actions/generations":
            self._send(404, {"error": "not found"})
            return
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        server = self.server
        server.requests.append(payload)
        dimension = len(payload["input"]["observation"]["state"])
        rows = [[0.25] * dimension for _ in range(server.horizon)]
        self._send(
            200,
            {
                "id": payload["request_id"],
                "model": "demo-sglang",
                "data": [{"action": {"values": rows}}],
            },
        )


class _DemoActionServer(ThreadingHTTPServer):
    """Own the demo model request log and configured proposal horizon."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], horizon: int):
        super().__init__(address, _DemoActionHandler)
        self.horizon = horizon
        self.requests: list[dict[str, Any]] = []


class _DemoCamera:
    """Return one deterministic camera frame for the software fixture."""

    def capture(self) -> tuple[CameraFrame, ...]:
        now = time.monotonic_ns()
        return (
            CameraFrame(
                "front",
                "image/png",
                b"demo-frame",
                captured_timestamp_ns=now,
                received_timestamp_ns=now,
                clock_domain="host_monotonic_ns",
            ),
        )

    def close(self) -> None:
        return None


class _DemoRobot(RobotAdapter):
    """Twelve-joint in-memory robot used to show before/after state."""

    def __init__(self, config: tuple[str, int]):
        self.robot_id, self.joint_count = config
        self._connected = False
        self._prepared = False
        self._state = [0.0] * self.joint_count

    def connect(self, *, prepare: bool = True) -> None:
        self._connected = True
        self._prepared = bool(prepare)

    def prepare(self) -> None:
        if not self._connected:
            raise RuntimeError("demo robot is not connected")
        self._prepared = True

    def observe(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("demo robot is not connected")
        return RobotObservation(
            timestamp_s=time.time(),
            values={"joint_positions": list(self._state)},
            metadata={
                "simulated": True,
                "hardware_access": False,
                "units": "demo_native",
                "action_space": "demo.twelve_joint.v1",
                "robot_id": self.robot_id,
                "captured_timestamp_ns": time.monotonic_ns(),
                "clock_domain": "host_monotonic_ns",
            },
        )

    def execute(self, action: RobotAction) -> None:
        if not self._connected or not self._prepared:
            raise RuntimeError("demo robot is not prepared")
        values = action.values
        if not isinstance(values, Mapping):
            raise ValueError("demo action must be a joint_position object")
        if values.get("type") == "joint_position":
            positions = values.get("joint_positions")
        else:
            expected = {f"joint_{index}" for index in range(self.joint_count)}
            if set(values) != expected:
                raise ValueError("demo action must declare all joint values")
            positions = [values[f"joint_{index}"] for index in range(self.joint_count)]
        if not isinstance(positions, Sequence) or len(positions) != self.joint_count:
            raise ValueError("demo action has an invalid joint count")
        numbers = [float(value) for value in positions]
        if any(not math.isfinite(value) for value in numbers):
            raise ValueError("demo action contains a non-finite joint")
        self._state = numbers

    def stop(self) -> None:
        return None

    def close(self) -> None:
        self._connected = False
        self._prepared = False


class _DemoMapper:
    """Map the demo policy schema to the demo robot's explicit joint actions."""

    def __init__(self, joint_count: int, horizon: int):
        self.joint_count = joint_count
        self.horizon = horizon
        self.policy_action_space = "demo.twelve_joint.v1"
        self.feature_names = tuple(f"joint_{index}" for index in range(joint_count))

    def map_observation(
        self,
        observation: RobotObservation,
        *,
        session_id: str,
        request_id: str,
        step_id: int,
        instruction: str,
        frames: Sequence[CameraFrame],
    ) -> PolicyObservation:
        state = observation.values.get("joint_positions") if isinstance(observation.values, Mapping) else None
        if not isinstance(state, Sequence) or len(state) != self.joint_count or not frames:
            raise ValueError("demo binding requires a complete joint state and camera frame")
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state={"joint_positions": list(state)},
            images=(ImagePayload(frames[0].name, frames[0].mime_type, frames[0].data),),
            metadata={"simulated": True, "hardware_access": False},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != self.policy_action_space or len(result.actions) != 1:
            raise ValueError("demo policy result action space is invalid")
        rows = result.actions[0].values.get("data")
        if not isinstance(rows, Sequence) or len(rows) != self.horizon:
            raise ValueError("demo policy result has an invalid horizon")
        actions = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Sequence) or len(row) != self.joint_count:
                raise ValueError("demo policy result has an invalid action dimension")
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values={
                        "type": "joint_position",
                        "joint_positions": [float(value) for value in row],
                    },
                    metadata={
                        "action_space": self.policy_action_space,
                        "simulated": True,
                        "hardware_access": False,
                        "chunk_index": row_index,
                    },
                )
            )
        return tuple(actions)


def _register_demo_types(joint_count: int, horizon: int) -> tuple[str, str]:
    """Register transient demo-only binding and robot definitions.

    The registrations live in process memory for this CLI invocation and do
    not alter the repository's real SO101 binding or robot registry.
    """

    robot_kind = f"simulated.demo{joint_count}"
    binding_kind = f"demo.twelve_joint{joint_count}"
    robot_module_name = f"embodirun.robots.{robot_kind}"
    robot_module = ModuleType(robot_module_name)
    robot_module.ROBOT_DEFINITION = RobotDefinition(
        kind=robot_kind,
        config_factory=lambda robot_id, _options: (robot_id, joint_count),
        adapter_type=_DemoRobot,
        environment_group="host",
    )
    sys.modules[robot_module_name] = robot_module
    binding_parent = ModuleType("embodirun.bindings.demo")
    binding_parent.__path__ = []
    sys.modules["embodirun.bindings.demo"] = binding_parent
    binding_module_name = f"embodirun.bindings.{binding_kind}"
    binding_module = ModuleType(binding_module_name)
    binding_module.BINDING_DEFINITION = BindingDefinition(
        kind=binding_kind,
        robot_kind=robot_kind,
        model_kind="demo",
        mapper_factory=lambda: _DemoMapper(joint_count, horizon),
        maximum_chunk_steps=horizon,
    )
    sys.modules[binding_module_name] = binding_module
    binding_definition.cache_clear()
    robot_definition.cache_clear()
    importlib.import_module(binding_module_name)
    importlib.import_module(robot_module_name)
    return robot_kind, binding_kind


def _run_demo(joint_count: int) -> dict[str, Any]:
    """Run the public HTTP flow against local software fixtures.

    This function contains the algorithm-side calls (``ControlClient`` and
    ``CooperativeLoop``) while the ``_Demo*`` classes above provide the fake
    model, camera, and robot facilities. Keeping those roles visible makes it
    clear that the example exercises API calls rather than a private driver.
    """

    if joint_count != ACTION_DIM:
        raise ValueError(f"--joint-count must equal the Astra action dimension {ACTION_DIM} for this decision contract")
    robot_kind, binding_kind = _register_demo_types(joint_count, HORIZON)
    model_server = _DemoActionServer(("127.0.0.1", 0), HORIZON)
    model_thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    model_thread.start()
    input_value = SensorInput("demo-camera", "front", "fake", {})
    with tempfile.TemporaryDirectory(prefix="astra-pi05-demo-") as state_dir:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            control_port = probe.getsockname()[1]
        config = ControlServiceConfig(
            runtime_id="astra-demo",
            binding_kind=binding_kind,
            bind="127.0.0.1",
            port=control_port,
            inference_transport="http",
            inference_endpoint=f"http://127.0.0.1:{model_server.server_port}",
            inference_options={
                "action_feature_names": [f"joint_{index}" for index in range(joint_count)],
                "output_action_dim": joint_count,
                "state_fields": ["joint_positions"],
            },
            inference_backend="sglang",
            robot_id="demo-robot",
            robot_kind=robot_kind,
            robot_options={"joint_count": joint_count},
            inputs=(input_value,),
            runtime_options={},
        )
        service = ControlService(
            config,
            camera_factory=lambda _items: _DemoCamera(),
            device_manager=DeviceManager("demo", lock_dir=Path(state_dir) / "locks"),
        )
        server = ControlHttpServer(service, state_dir=state_dir)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = ControlClient(
            f"http://127.0.0.1:{server.server_port}",
            caller_id="astra-demo",
            session_id="demo-session",
        )
        try:
            before = client.observe(include_robot=True)
            names = tuple(f"joint_{index}" for index in range(joint_count))

            def proposal_provider(observation):
                payload = client.propose(
                    request_id="astra-demo-proposal",
                    observation_id=observation.observation_id,
                    instruction="move demo joints",
                    max_age_ns=2_000_000_000,
                )
                policy_action = payload["proposal"]["actions"][0]
                return {
                    "proposal_id": "astra-demo-proposal",
                    "observation_id": observation.observation_id,
                    "actions": policy_action["values"]["data"],
                    "metadata": {
                        "action_semantics": "biso101_so101_v1",
                        "action_layout": "left6_right6",
                        "action_encoding": "absolute",
                        "joint_position_unit": "degrees",
                        "action_dim": ACTION_DIM,
                        "horizon": HORIZON,
                        "feature_names": list(names),
                    },
                }

            def reviewer(packet):
                return {
                    "proposal_id": packet["proposal"]["proposal_id"],
                    "observation_id": packet["observation"]["observation_id"],
                    "decision": "execute_prefix",
                    "execute_steps": 3,
                    "corrections": [],
                    "reason": "software demo reviewer selects three rows",
                }

            result = CooperativeLoop(
                client,
                proposal_provider=proposal_provider,
                reviewer=reviewer,
                feature_names=names,
                post_observation_timeout_s=2.0,
            ).run_round(prompt="move demo joints")
            if result.post_observation is None:
                raise RuntimeError(result.error or "demo did not receive a new observation")
            return {
                "status": result.status,
                "proposal_steps": HORIZON,
                "executed_steps": result.executed_steps,
                "discarded_steps": result.discarded_steps,
                "before_state": before.robot,
                "after_state": result.post_observation.robot,
                "model_requests": len(model_server.requests),
                "hardware_access": False,
            }
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    model_server.shutdown()
    model_server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the local HTTP cooperative demo")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the pure decision contract",
    )
    parser.add_argument("--joint-count", type=int, default=ACTION_DIM, help="demo joint count")
    args = parser.parse_args(argv)
    if args.demo and args.validate_only:
        parser.error("--demo and --validate-only are mutually exclusive")
    try:
        payload = _run_demo(args.joint_count) if args.demo else _validate_only()
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
