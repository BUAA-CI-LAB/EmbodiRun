"""Host plan metadata needed for unambiguous shared sensor ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from embodirun.services.host.config import load_config
from embodirun.services.host.plan import ServiceError, build_plan

ROOT = Path(__file__).parents[1]
FAKE_CONFIG = ROOT / "examples" / "shared-device-fake.yaml"


def test_generated_resources_match_each_sensor_input_by_sensor_id(tmp_path: Path) -> None:
    document = yaml.safe_load(FAKE_CONFIG.read_text(encoding="utf-8"))
    document["sensors"]["fake-wrist"] = {
        "type": "fake",
        "node": "local",
        "resource": "fake-wrist-camera",
        "width": 32,
        "height": 24,
    }
    document["runtimes"]["fake-device"]["inputs"]["observation.images.wrist"] = "fake-wrist"
    path = tmp_path / "two-cameras.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    plan = build_plan(load_config(path))
    control = next(item for item in plan.services if item.kind == "control")
    config = json.loads(control.control_config_json or "")
    sensors = {item["sensor_id"]: item for item in config["device_resources"] if item["kind"] == "sensor"}

    assert set(sensors) == {"fake-front", "fake-wrist"}
    assert sensors["fake-front"]["identity"] == "local:sensor:fake-front"
    assert sensors["fake-wrist"]["identity"] == "local:sensor:fake-wrist-camera"
    assert sensors["fake-front"]["sensor_ids"] == ["fake-front"]
    assert sensors["fake-wrist"]["sensor_ids"] == ["fake-wrist"]


def test_generated_resource_keeps_logical_aliases_for_one_physical_sensor(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(FAKE_CONFIG.read_text(encoding="utf-8"))
    document["sensors"]["fake-alias"] = {
        "type": "fake",
        "node": "local",
        "resource": "fake-front",
        "width": 64,
        "height": 48,
    }
    document["runtimes"]["fake-device"]["inputs"]["observation.images.alias"] = "fake-alias"
    path = tmp_path / "camera-alias.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    plan = build_plan(load_config(path))
    control = next(item for item in plan.services if item.kind == "control")
    config = json.loads(control.control_config_json or "")
    sensor = next(item for item in config["device_resources"] if item["identity"] == "local:sensor:fake-front")

    assert sensor["sensor_id"] == "fake-front"
    assert sensor["sensor_ids"] == ["fake-front", "fake-alias"]


@pytest.mark.parametrize("scope", ["arms", "base"])
def test_xlerobot_external_owner_builds_control_plan(scope: str, tmp_path: Path) -> None:
    document = yaml.safe_load(FAKE_CONFIG.read_text(encoding="utf-8"))
    document["robots"]["fake-arm"] = {
        "type": "lerobot.xlerobot",
        "node": "local",
        "resource": "xlerobot-owner",
        "owner": "teleop-owner",
        "url": "http://127.0.0.1:9999",
        "token": "test-token",
        "scope": scope,
    }
    path = tmp_path / f"xlerobot-{scope}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    plan = build_plan(load_config(path))

    environment = next(item for item in plan.environments if item.group == "robot-xlerobot-external-owner")
    assert environment.project == "deploy"
    control = next(item for item in plan.services if item.kind == "control")
    control_config = json.loads(control.control_config_json or "")
    robot_resources = [item for item in control_config["device_resources"] if item["kind"] == "robot"]
    assert len(robot_resources) == 1
    assert robot_resources[0]["external_owner"] is True
    assert robot_resources[0]["owner"] == "teleop-owner"
    assert control.command.argv == ("embodirun-control-serve",)


def test_unsupported_external_owner_still_fails_plan(tmp_path: Path) -> None:
    document = yaml.safe_load(FAKE_CONFIG.read_text(encoding="utf-8"))
    document["robots"]["fake-arm"]["owner"] = "legacy-owner"
    path = tmp_path / "unsupported-owner.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ServiceError, match="unsupported external owner"):
        build_plan(load_config(path))
