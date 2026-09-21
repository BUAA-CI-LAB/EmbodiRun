import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from embodirun.deployment.config import ConfigError, load_config
from embodirun.deployment.operations.init import InitError, _probe_resources
from embodirun.deployment.plan import ServiceError, _binding_adapter_config, build_plan
from embodirun.robots.lerobot.bi_so101 import BI_SO101_POSITION_FEATURES

EXAMPLE = Path(__file__).parents[1] / "configs/pi05/bi-so101-vvla.yaml"


def write_config(tmp_path, document):
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_config(path)


def test_dual_so101_plan_generates_mixed_precision_vvla_config():
    config = load_config(EXAMPLE)
    plan = build_plan(config)
    service = next(item for item in plan.services if item.kind == "model")
    adapter = json.loads(service.adapter_config_json)
    assert adapter["state_fields"] == list(BI_SO101_POSITION_FEATURES)
    assert adapter["action_feature_names"] == list(BI_SO101_POSITION_FEATURES)
    assert adapter["image_fields"] == [
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert adapter["return_steps"] == 50
    assert adapter["policy_kwargs"] == {
        "attention": "eager",
        "vision_attention": "sdpa",
        "native_embeddings": True,
        "low_cpu_mem_usage": True,
    }
    argv = service.command.argv
    assert argv[argv.index("--dtype") + 1] == "auto"
    assert argv[argv.index("--num-steps") + 1] == "10"
    assert "--capture-full-loop" in argv
    control = json.loads(next(item for item in plan.services if item.kind == "control").control_config_json)
    assert control["binding"] == "lerobot.bi_so101.pi05"
    assert control["robot"]["options"]["right_port"].endswith("REPLACE_RIGHT_ARM")
    assert {profile.group for profile in plan.environments} == {"pi05", "robot-so101"}


@pytest.mark.parametrize("kwargs", [[], "eager", {"": True}, {"attention": float("nan")}])
def test_invalid_policy_options_fail_validation(tmp_path, kwargs):
    document = yaml.safe_load(EXAMPLE.read_text())
    document["models"]["pi05"]["policy_kwargs"] = kwargs
    with pytest.raises(ServiceError):
        build_plan(write_config(tmp_path, document))


def test_model_policy_options_do_not_replace_binding_state_fields(tmp_path):
    document = yaml.safe_load(EXAMPLE.read_text())
    document["models"]["pi05"]["policy_kwargs"]["state_fields"] = ["invalid"]
    plan = build_plan(write_config(tmp_path, document))
    adapter = json.loads(plan.services[0].adapter_config_json)
    assert adapter["state_fields"] == list(BI_SO101_POSITION_FEATURES)
    assert adapter["policy_kwargs"]["state_fields"] == ["invalid"]


def test_policy_options_cannot_be_silently_ignored_by_other_backends():
    config = load_config(EXAMPLE)
    model = replace(config.models["pi05"], backend="sglang")
    with pytest.raises(ServiceError, match="requires VVLA"):
        _binding_adapter_config(config, model)


def test_policy_options_require_generated_config_when_binding_has_none():
    config = load_config(EXAMPLE)
    config.runtimes.clear()
    with pytest.raises(ServiceError, match="requires a binding-generated"):
        _binding_adapter_config(config, config.models["pi05"])


def test_generated_adapter_rejects_external_file_conflict(tmp_path):
    document = yaml.safe_load(EXAMPLE.read_text())
    document["models"]["pi05"]["adapter_config"] = "/models/custom.json"
    with pytest.raises(ServiceError, match="conflicts"):
        build_plan(write_config(tmp_path, document))


@pytest.mark.parametrize("side", ["left", "right"])
def test_both_arm_ports_participate_in_existing_conflict_check(tmp_path, side):
    document = yaml.safe_load(EXAMPLE.read_text())
    document["robots"]["single"] = {
        "type": "lerobot.so101",
        "node": "robot-compute",
        "port": document["robots"]["bi-so101"][f"{side}_port"],
    }
    with pytest.raises(ConfigError, match="share port"):
        write_config(tmp_path, document)


def test_init_probes_both_ports_before_using_hardware():
    config = load_config(EXAMPLE)
    commands = []

    class Executor:
        def run(self, command, **kwargs):
            commands.append(command.argv)
            return SimpleNamespace(exit_code=int(command.argv[-1].endswith("REPLACE_RIGHT_ARM")))

    with pytest.raises(InitError, match="REPLACE_RIGHT_ARM"):
        _probe_resources(
            SimpleNamespace(config=config),
            Executor(),
            SimpleNamespace(node_id="robot-compute", deploy_project="/deploy", inference_project="/inference"),
        )
    assert commands == [
        ("test", "-e", "/dev/serial/by-id/REPLACE_LEFT_ARM"),
        ("test", "-e", "/dev/serial/by-id/REPLACE_RIGHT_ARM"),
    ]


def test_camera_mapping_reaches_vvla_with_policy_options(tmp_path):
    document = yaml.safe_load(EXAMPLE.read_text())
    document["models"]["pi05"]["image_keys"] = {
        "observation.images.front": "observation.images.base_0_rgb",
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
    }
    plan = build_plan(write_config(tmp_path, document))
    adapter = json.loads(plan.services[0].adapter_config_json)
    assert adapter["image_keys"] == document["models"]["pi05"]["image_keys"]
    assert adapter["policy_kwargs"]["native_embeddings"] is True
