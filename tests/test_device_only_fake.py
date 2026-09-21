"""Host and adapter coverage for the dependency-free device-only example."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.robots import RobotAction, robot_definition
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras.fake import create_source
from embodirun.services.control.contracts import ControlServiceConfig
from embodirun.services.control.devices import (
    DeviceManager,
    DeviceResource,
    ResourceIdentity,
)
from embodirun.services.control.server import ControlService
from embodirun.services.host.cli.command.run import RunError, run
from embodirun.services.host.config import ConfigError, load_config
from embodirun.services.host.environment import environment_profiles
from embodirun.services.host.plan import build_plan

ROOT = Path(__file__).parents[1]
FAKE_CONFIG = ROOT / "examples" / "shared-device-fake.yaml"


def test_host_builds_real_device_only_service_from_yaml() -> None:
    config = load_config(FAKE_CONFIG)
    plan = build_plan(config)

    assert config.models == {}
    assert len(plan.services) == 1
    service = plan.services[0]
    assert service.kind == "control"
    assert service.command.argv == ("embodirun-control-serve",)
    runtime = plan.runtimes[0]
    assert runtime.model is None
    assert runtime.binding is None
    assert runtime.model_endpoint is None
    generated = ControlServiceConfig.from_json(service.control_config_json or "")
    assert generated.inference_enabled is False
    assert generated.inference_transport == "disabled"
    assert generated.inference_endpoint == ""
    assert generated.robot_kind == "simulated.joints"
    assert generated.inputs[0].kind == "fake"
    assert [profile.group for profile in environment_profiles(config)] == ["host"]


def test_no_model_is_a_robot_only_runtime_mode(tmp_path: Path) -> None:
    source = """
metadata:
  name: invalid-device-simulator
  deploy-commit: main
nodes:
  local:
    type: workstation
    connection:
      type: local
simulators:
  fake-sim:
    type: libero
    node: local
    suite: libero_spatial
    task_id: 0
runtimes:
  runtime:
    simulator: fake-sim
    server:
      bind: 127.0.0.1
      port: 8100
"""
    path = tmp_path / "invalid.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="simulator runtimes require model"):
        load_config(path)


def test_model_and_binding_are_optional_only_as_a_pair(tmp_path: Path) -> None:
    source = FAKE_CONFIG.read_text(encoding="utf-8")
    source = source.replace("    inputs:\n", "    model: fake\n    inputs:\n")
    path = tmp_path / "invalid.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="model and .*binding.*provided together"):
        load_config(path)


def test_host_run_explains_that_device_only_has_no_inference() -> None:
    plan = build_plan(load_config(FAKE_CONFIG))
    with pytest.raises(RunError, match="device-only.*no inference"):
        run(
            SimpleNamespace(runtime="fake-device", prompt=None),
            SimpleNamespace(deployment=plan),
        )


def test_fake_joints_adapter_has_passive_boundary_and_finite_state() -> None:
    definition = robot_definition("simulated.joints")
    config = definition.config_factory(
        "demo",
        {"initial_joint_positions_deg": [1, 2, 3, 4, 5], "initial_gripper_position": 20},
    )
    adapter = definition.adapter_type(config)
    adapter.connect(prepare=False)
    passive = adapter.observe()
    assert passive.metadata["simulated"] is True
    assert passive.metadata["hardware_access"] is False
    assert all(math.isfinite(value) for value in passive.values["joint_positions_deg"])
    assert math.isfinite(passive.values["gripper_position"])
    with pytest.raises(RuntimeError, match="not prepared"):
        adapter.execute(
            RobotAction(
                timestamp_s=0.0,
                values={
                    "type": "joint_position",
                    "joint_positions_deg": [1, 2, 3, 4, 5],
                    "gripper_position": 20,
                },
                metadata={},
            )
        )
    adapter.prepare()
    adapter.execute(
        RobotAction(
            timestamp_s=0.0,
            values={
                "type": "joint_position",
                "joint_positions_deg": [2, 3, 4, 5, 6],
                "gripper_position": 25,
            },
            metadata={},
        )
    )
    assert adapter.observe().values["gripper_position"] == 25.0
    adapter.close()


def test_fake_camera_emits_static_png_with_host_timestamp() -> None:
    source = create_source(
        (
            SensorInput(
                sensor_id="front",
                name="observation.images.front",
                kind="fake",
                options={"width": 8, "height": 6, "color": [1, 2, 3]},
            ),
        )
    )
    frame = source.capture()[0]
    assert frame.mime_type == "image/png"
    assert frame.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert frame.captured_timestamp_ns is not None
    assert frame.received_timestamp_ns is not None
    assert frame.received_timestamp_ns >= frame.captured_timestamp_ns
    assert frame.clock_domain == "host_monotonic_ns"
    assert frame.profile["simulated"] is True
    source.close()


def test_device_only_control_service_observes_fake_robot_and_camera(
    tmp_path: Path,
) -> None:
    plan = build_plan(load_config(FAKE_CONFIG))
    service_spec = plan.services[0]
    generated = ControlServiceConfig.from_json(service_spec.control_config_json or "")
    manager = DeviceManager(
        "local",
        owner_id="fake-service",
        lock_dir=tmp_path / "locks",
        state_path=tmp_path / "state.json",
    )
    service = ControlService(generated, device_manager=manager)
    assert service.health()["inference"] == "disabled"
    observed = service.observe(include_robot=True)
    assert observed["frames"][0]["mime_type"] == "image/png"
    assert observed["robot"]["metadata"]["simulated"] is True
    service.close()


def test_device_manager_serializes_close_callback_and_lease_release(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = []

    def closer(_value: object) -> None:
        closed.append("close")
        entered.set()
        assert release.wait(2.0)

    identity = ResourceIdentity("local", "fake", "serial-close-race")
    manager = DeviceManager(
        "local",
        owner_id="close-race",
        lock_dir=tmp_path / "locks",
        state_path=tmp_path / "state.json",
    )
    lease = manager.acquire(DeviceResource(identity, opener=lambda: object(), closer=closer))
    lease_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    release_thread = threading.Thread(target=lambda: _capture(lease.close, lease_errors), daemon=True)
    release_thread.start()
    assert entered.wait(2.0)
    manager_thread = threading.Thread(target=lambda: _capture(manager.close, close_errors), daemon=True)
    manager_thread.start()
    release.set()
    release_thread.join(2.0)
    manager_thread.join(2.0)
    assert not release_thread.is_alive()
    assert not manager_thread.is_alive()
    assert lease_errors == []
    assert close_errors == []
    assert closed == ["close"]


def _capture(callback, errors: list[BaseException]) -> None:
    try:
        callback()
    except BaseException as error:  # pragma: no cover - assertion helper
        errors.append(error)
