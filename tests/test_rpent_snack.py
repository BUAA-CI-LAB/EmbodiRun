from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from agents.rpent.snack_agent import AstraSnackAgent
from examples.xlerobot_snack_delivery.services import prepare

from embodirun.application.contracts import ControlServiceConfig
from embodirun.bindings import binding_definition
from embodirun.model_services import PolicyAction, PolicyResult
from embodirun.robots.lerobot.xlerobot.units import ARM_UNITS, validate_action


def test_real_rpent_entrypoint_invokes_isolated_astra_and_checks_identity():
    calls = []

    async def command(argv, cwd):
        calls.append(argv)
        Path(argv[argv.index("--output-last-message") + 1]).write_text(
            json.dumps(
                {
                    "observation_id": "obs-1",
                    "decision": "proceed",
                    "instruction": "grasp chips",
                    "max_steps": 3,
                    "reason": "target visible",
                }
            )
        )
        return 0, "", ""

    agent = AstraSnackAgent(command_runner=command)
    packet = {
        "observation_id": "obs-1",
        "max_steps": 8,
        "instruction": "grasp chips",
        "images": [{"mime_type": "image/png", "data": base64.b64encode(b"fixture-image").decode()}],
    }
    assert agent.before_grasp(packet)["max_steps"] == 3
    assert "--ignore-user-config" in calls[0]
    assert calls[0][calls[0].index("--sandbox") + 1] == "read-only"
    assert "gpt-6-astra" in calls[0]
    with pytest.raises(ValueError, match="different observation"):
        agent.before_grasp({**packet, "observation_id": "obs-2"})
    with pytest.raises(ValueError, match="requires snapshot images"):
        agent.before_grasp({**packet, "images": []})


def test_xlerobot_binding_preserves_names_units_and_rejects_bare_vectors():
    mapper = binding_definition("lerobot.xlerobot.pi05").mapper_factory()
    names = list(reversed(ARM_UNITS))

    def result(features):
        return PolicyResult(
            request_id="r",
            session_id="s",
            step_id=0,
            session_revision=1,
            action_space="pi05.action_chunk.v1",
            actions=(
                PolicyAction("action_chunk", {"data": [[float(i) for i in range(12)]], "feature_names": features}),
            ),
        )

    (action,) = mapper.map_result(result(names))
    assert action.values[names[0]] == 0
    validate_action(action.values, action.metadata, scope="arms")
    with pytest.raises(Exception, match="features"):
        mapper.map_result(result([]))


def test_deployment_yaml_materializes_two_scopes_with_private_generated_auth(tmp_path):
    hardware = tmp_path / "hardware.json"
    hardware.write_text("{}")
    config = {
        "owner": {"port": 8766, "hardware_config": str(hardware)},
        "control": {"base_port": 8100, "manipulation_port": 8101},
        "model": {"endpoint": "http://localhost:8000"},
        "cameras": {"observation.images.front": "front"},
    }
    commands = prepare(config, tmp_path / "private", base_dir=tmp_path)
    assert len(commands) == 3
    base = ControlServiceConfig.from_json((tmp_path / "private/base-control.json").read_text())
    arms = ControlServiceConfig.from_json((tmp_path / "private/arms-control.json").read_text())
    assert base.robot_options["scope"] == "base" and not base.inference_enabled
    assert arms.robot_options["scope"] == "arms" and arms.inference_enabled
    assert base.robot_options["token"] == arms.robot_options["token"]
    assert (tmp_path / "private/arms-control.json").stat().st_mode & 0o777 == 0o600
    assert arms.inputs[0].kind == "xlerobot"
    assert arms.binding_kind == "lerobot.xlerobot.pi05"


def test_camera_without_owner_capture_age_never_becomes_fresh(monkeypatch):
    from embodirun.robots.lerobot.xlerobot.adapter import XLeRobotAdapter
    from embodirun.robots.sensors import SensorInput
    from embodirun.robots.sensors.cameras.xlerobot import XLeRobotCameraSource

    source = XLeRobotCameraSource(
        [
            SensorInput(
                "front",
                "front",
                "xlerobot",
                {"url": "http://127.0.0.1:1", "token": "fixture", "scope": "arms", "camera": "front"},
            )
        ]
    )
    reply = {
        "observation": {"camera_timestamps_ns": {"front": 123}},
        "images": {"front": base64.b64encode(b"jpeg").decode()},
    }
    monkeypatch.setattr(XLeRobotAdapter, "_request", lambda *args: reply)
    with pytest.raises(ValueError, match="capture age"):
        source.capture()
    reply["observation"]["camera_ages_ns"] = {"front": 2_000_000_000}
    (frame,) = source.capture()
    assert frame.received_timestamp_ns - frame.captured_timestamp_ns >= 2_000_000_000
    assert frame.profile["remote_capture_timestamp_ns"] == 123
    source.close()


def test_diagram_rejects_unknown_segment_and_never_generates_motor_values():
    agent = AstraSnackAgent()
    agent._decide = lambda *args: {"decision": "proceed", "outbound": ["invented"], "return": ["back"]}
    with pytest.raises(ValueError, match="invalid outbound"):
        agent.plan_routes({"segments": {"there": "to table", "back": "return"}, "images": [{}]})
