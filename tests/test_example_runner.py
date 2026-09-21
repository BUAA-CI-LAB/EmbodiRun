"""Exercise example dispatch and safety gates without hardware or optional SDKs."""

import json
import os
import sys
import time

import pytest
import yaml
from examples import runner

CONFIGS = [
    "multi_robot_serving/example.yaml",
    "engine_comparison/embodiinfer-http.yaml",
    "engine_comparison/embodiinfer-wireless.yaml",
    "engine_comparison/sglang-http.yaml",
    "microduck_vln/example.yaml",
    "xlerobot_snack_delivery/example.yaml",
]


@pytest.mark.parametrize("config", CONFIGS)
def test_shipped_manifests_validate_and_plan_without_resources(config, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("offline commands must not start a process")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    path = runner.ROOT / "examples" / config
    assert runner.main([str(path), "validate"]) == 0
    capsys.readouterr()
    assert runner.main([str(path), "plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["commands"]) == (3 if config.startswith("multi_robot") else 1)
    assert all(isinstance(command, list) for command in plan["commands"])


@pytest.mark.parametrize("command", ["up", "run"])
@pytest.mark.parametrize("config", [CONFIGS[0], CONFIGS[-1]])
def test_hardware_requires_explicit_gate(config, command, monkeypatch, capsys):
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("hardware executed"))
    assert runner.main([str(runner.ROOT / "examples" / config), command]) == 2
    assert "--allow-hardware" in capsys.readouterr().err


def test_placeholders_cannot_start_setup_or_hardware(capsys):
    path = runner.ROOT / "examples" / CONFIGS[0]
    for command in ("setup", "up", "run"):
        assert runner.main([str(path), command, "--allow-hardware"]) == 2
        assert "REPLACE_" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunks", True),
        ("control_hz", float("nan")),
        ("chunk_steps", 51),
        ("runtimes", ["missing"]),
        ("runtimes", ["arm-1", "arm-1"]),
        ("chunk_step", 5),
    ],
)
def test_invalid_rollout_parameters_fail_before_launch(tmp_path, field, value):
    source = runner.ROOT / "examples" / CONFIGS[0]
    data = yaml.safe_load(source.read_text())
    data["parameters"]["deployment"] = str(source.parent / "deployment.yaml")
    data["parameters"][field] = value
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        runner.load_example(path)


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "example.yaml"
    path.write_text("version: 1\nversion: 1\n")
    with pytest.raises(ValueError, match="duplicate"):
        runner.load_example(path)


def test_microduck_argv_paths_and_legacy_env_are_explicit(tmp_path):
    example = runner.load_example(runner.ROOT / "examples" / CONFIGS[-2])
    commands, env = runner.commands(example, "check", tmp_path)
    command = commands[0]
    assert "--check-only" in command
    assert command[command.index("--output") + 1] == str(tmp_path / "result")
    assert command[command.index("--manifest") + 1] == str(example.path.parent / "assets.md5.json")
    assert command[0].endswith(".venv-microduck/bin/python")
    assert env["MUJOCO_GL"] == "egl"
    assert "integrations/microduck_vln/src" in env["PYTHONPATH"]


def test_failed_rollout_stops_deployment_and_records_failure(tmp_path, monkeypatch):
    example = runner.load_example(runner.ROOT / "examples" / CONFIGS[0])
    deployment = tmp_path / "deployment.yaml"
    deployment.write_text("# Replace every REPLACE_* value before use.\nconfigured: true\n")
    example.parameters["deployment"] = str(deployment)
    example.data["output_dir"] = str(tmp_path / "outputs")
    monkeypatch.setattr(runner, "load_example", lambda path: example)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "test-revision\n")
    stopped = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: stopped.append(command) or type("Result", (), {"returncode": 0})(),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("a robot client failed")

    monkeypatch.setattr(runner, "execute", fail)
    assert runner.main(["ignored.yaml", "run", "--allow-hardware"]) == 2
    assert stopped == [example.host("down")]
    report = json.loads(next((tmp_path / "outputs").glob("*/run.json")).read_text())
    assert report["status"] == "failed"
    assert report["cleanup_returncode"] == 0
    output = next((tmp_path / "outputs").iterdir())
    assert (output / report["input_snapshots"][str(deployment)]).read_text() == deployment.read_text()


def test_parallel_failure_reaps_other_local_children(tmp_path):
    pid_path = tmp_path / "child.pid"
    waiting = [
        sys.executable,
        "-c",
        "import os,time,pathlib; pathlib.Path("
        + repr(str(pid_path))
        + ").write_text(str(os.getpid())); time.sleep(30)",
    ]
    failing = [sys.executable, "-c", "import time; time.sleep(.3); raise SystemExit(1)"]
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        runner.execute([waiting, failing], dict(os.environ), tmp_path, parallel=True, interactive=False)
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_slurm_allocation_uses_yaml_and_is_not_nested(tmp_path, monkeypatch):
    example = runner.load_example(runner.ROOT / "examples" / CONFIGS[-2])
    example.parameters["slurm"] = "auto"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    command = runner.commands(example, "run", tmp_path)[0][0]
    assert command[0] == "srun"
    assert "--cpus-per-task=4" in command
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert runner.commands(example, "run", tmp_path)[0][0][0] == example.python
    assert runner.commands(example, "provision", tmp_path)[0][0][0] == example.python
