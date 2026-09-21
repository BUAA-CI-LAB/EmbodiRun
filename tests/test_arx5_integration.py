"""Small regression set for the installed ARX5 deployment boundary (no hardware)."""

import sys
from pathlib import Path

import pytest

from embodirun.bindings import binding_definition
from embodirun.robots import robot_definition
from embodirun.robots.arx.x5 import ARX5Adapter, ARX5Config
from embodirun.services.inference import PolicyAction, PolicyResult


def test_standard_host_plan_selects_arx5_and_dm05(tmp_path):
    import yaml

    from embodirun.services.host.config import load_config
    from embodirun.services.host.plan import build_plan

    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/http-wireless-inference/http.yaml").read_text())
    node = config["robots"]["so101-1"]["node"]
    config["robots"] = {"arm": {"type": "arx.x5", "node": node}}
    model = next(iter(config["models"].values()))
    model.update(type="dm05", environment=".venv-vvla-dm05")
    model.pop("server_args")
    adapter_config = tmp_path / "dm05.json"
    adapter_config.write_text(
        '{"policy_kwargs": {"norm_stats": "/models/dm05/norm_stats.json", '
        '"is_history": true, "robot_type": "ARX5", "output_action_dim": 7}}'
    )
    model["adapter_config"] = str(adapter_config)
    config["models"] = {"policy": model}
    runtime = next(iter(config["runtimes"].values()))
    runtime.update(robot="arm", model="policy", binding="arx.x5.dm05")
    runtime["inputs"] = {"cam_global": "front-camera", "cam_arm": "wrist-camera"}
    config["runtimes"] = {"arm-runtime": runtime}
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(config))
    plan = build_plan(load_config(path))
    assert {environment.group for environment in plan.environments} == {
        "robot-arx5",
        "dm05",
    }
    model_service = next(service for service in plan.services if service.kind == "model")
    assert model_service.command.argv[:3] == ("vvla-http-serve", "--policy", "dm05")
    args = model_service.command.argv
    assert args[args.index("--adapter-config") + 1] == str(adapter_config)


def test_arx5_environment_has_both_camera_backends():
    toml = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    project = toml.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    group = robot_definition("arx.x5").environment_group
    dependencies = project["dependency-groups"][group]
    for package in ("Pillow", "pyrealsense2", "opencv-python-headless"):
        assert any(item.startswith(package + ">=") for item in dependencies)


def test_vendor_sdk_path_does_not_require_host_pythonpath(tmp_path):
    # A stand-in for the separately built vendor extension; never touches CAN.
    (tmp_path / "fixture_arx_sdk.py").write_text(
        "class SingleArm:\n    def __init__(self, config): self.config = config\n"
    )
    config = ARX5Config.from_mapping(
        "arm",
        {
            "operator_confirmed": True,
            "sdk_module": "fixture_arx_sdk",
            "sdk_path": str(tmp_path),
        },
    )
    before = list(sys.path)
    adapter = ARX5Adapter(config)
    try:
        adapter.connect()
        assert adapter.arm.config == {"can_port": "can1", "type": 0}
        assert sys.path == before
    finally:
        sys.modules.pop("fixture_arx_sdk", None)


def test_binding_preserves_last_action_and_normalized_gripper():
    from embodirun.bindings.arx.x5.dm05 import (
        ACTION_FEATURE_NAMES,
        ACTION_REPRESENTATION,
    )

    mapper = binding_definition("arx.x5.dm05").mapper_factory()
    result = PolicyResult(
        session_id="session",
        request_id="step",
        step_id=0,
        session_revision=1,
        action_space="dm05.action_chunk.v1",
        actions=(
            PolicyAction(
                "action_chunk",
                {
                    "data": [[index / 1000, 0, 0, 0, 0, 0, 0.04] for index in range(50)],
                    "feature_names": ACTION_FEATURE_NAMES,
                    "representation": ACTION_REPRESENTATION,
                    "output_transform_applied": True,
                },
            ),
        ),
    )
    actions = mapper.map_result(result)
    assert [a.metadata["source_action_index"] for a in actions] == list(range(1, 50, 2))
    assert actions[-1].values["eef_xyzrpy_gripper"][0] == 0.049
    assert all(a.values["eef_xyzrpy_gripper"][6] == 0.04 for a in actions)
