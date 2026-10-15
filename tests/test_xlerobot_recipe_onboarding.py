"""Regress installation/dispatch drift and explainable read-only recipe checks."""

import json
import sys
from pathlib import Path

import pytest
import yaml
from examples import configuration, runner
from examples.configuration import initialize
from examples.xlerobot_snack_delivery import run as recipe


def _select_python(path, python):
    data = yaml.safe_load(path.read_text())
    data["python"] = str(python)
    path.write_text(yaml.safe_dump(data))


def _installed_metadata(python, packages):
    return {"version": [3, 12, 11], "packages": dict.fromkeys(packages, "installed"), "sdk_src": None}


@pytest.mark.parametrize("selection", ["manifest", "environment"])
def test_setup_installs_into_the_same_venv_used_by_scene(selection, tmp_path, monkeypatch):
    path = initialize("xlerobot", tmp_path / "recipe")
    environment = tmp_path / "custom venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = fixture\n")
    python = environment / "bin/python"
    python.symlink_to(sys.executable)
    if selection == "manifest":
        _select_python(path, python)
        monkeypatch.setenv("EMBODIRUN_SCENE_PYTHON", "/ignored/bin/python")
    else:
        monkeypatch.setenv("EMBODIRUN_SCENE_PYTHON", str(python))
    example = runner.load_example(path)
    setup = runner.commands(example, "setup", tmp_path, mode="software")[0][0]
    scene = runner.commands(example, "dry-run", tmp_path)[0][0]
    assert setup[setup.index("--environment") + 1] == str(environment)
    assert scene[0] == str(python)


def test_system_python_cannot_be_an_install_target(tmp_path, monkeypatch):
    path = initialize("xlerobot", tmp_path / "recipe")
    monkeypatch.setenv("EMBODIRUN_SCENE_PYTHON", "/usr/bin/python3")
    with pytest.raises(ValueError, match="not a virtual environment"):
        runner.commands(runner.load_example(path), "setup", tmp_path)


def test_deployment_ports_drive_check_and_control_clients_without_moving_routes(tmp_path, monkeypatch):
    path = initialize("xlerobot", tmp_path / "recipe")
    _select_python(path, sys.executable)
    task = path.parent / "config.local.json"
    assert "endpoints" not in json.loads(task.read_text())["control"]
    deployment = path.parent / "deployment.local.yaml"
    values = yaml.safe_load(deployment.read_text())
    values["control"] = {"base_port": 8210, "manipulation_port": 8211}
    deployment.write_text(yaml.safe_dump(values))
    config = recipe.load_recipe_config(task, deployment)
    connected = []
    monkeypatch.setattr("embodirun.client.ControlClient", lambda endpoint, **kw: connected.append(endpoint))
    recipe.ControlRuntimes(config)
    monkeypatch.setattr(configuration, "_environment_details", _installed_metadata)
    report = configuration.check_report(runner.load_example(path), mode="software")
    assert connected == ["http://127.0.0.1:8210", "http://127.0.0.1:8211"]
    assert [item["endpoint"] for item in report["control_endpoints"].values()] == connected
    assert Path(config["_config_dir"]) == task.parent
    assert all((task.parent / ref).is_file() for ref in config["navigation"]["routes"].values())


def test_standalone_config_rehearsal_keeps_legacy_endpoint_support(tmp_path, monkeypatch):
    monkeypatch.setattr("embodirun.client.ControlClient", lambda *a, **kw: pytest.fail("connected to Control"))
    task = runner.ROOT / "examples/xlerobot_snack_delivery/config.example.json"
    assert recipe.main(["--config", str(task), "--output", str(tmp_path), "--mode", "dry-run", "--auto-confirm"]) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["physical_success"] is None
    assert status["task_success"] == "unverified"


def test_software_check_accepts_fixtures_hardware_check_explains_missing_inputs(tmp_path, monkeypatch, capsys):
    path = initialize("xlerobot", tmp_path / "recipe")
    _select_python(path, sys.executable)
    monkeypatch.setattr(configuration, "_environment_details", _installed_metadata)
    monkeypatch.setattr(runner, "execute", lambda *a, **kw: pytest.fail("started a child service"))
    assert runner.main([str(path), "check", "--mode", "software", "--json"]) == 0
    software = json.loads(capsys.readouterr().out)
    assert software["issues"] == []
    assert software["unverified"]
    assert runner.main([str(path), "check", "--json"]) == 2
    hardware = json.loads(capsys.readouterr().out)
    assert {item["code"] for item in hardware["issues"]} >= {"fixture_route", "sdk_missing", "calibration_missing"}
    assert all(item["location"] and item["next_action"] for item in hardware["issues"])
    assert not (path.parent / "runs").exists()


def test_json_check_reports_an_unreadable_manifest_as_one_actionable_object(tmp_path, capsys):
    path = tmp_path / "missing.yaml"
    assert runner.main([str(path), "check", "--mode", "software", "--json"]) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "needs_attention"
    assert report["issues"][0]["next_action"]
    assert captured.err == ""


def test_json_check_reports_malformed_deployment_ports_without_a_traceback(tmp_path, capsys):
    path = initialize("xlerobot", tmp_path / "recipe")
    deployment = path.parent / "deployment.local.yaml"
    values = yaml.safe_load(deployment.read_text())
    values["control"] = []
    deployment.write_text(yaml.safe_dump(values))
    assert runner.main([str(path), "check", "--mode", "software", "--json"]) == 2
    captured = capsys.readouterr()
    assert "deployment.control" in json.loads(captured.out)["issues"][0]["message"]
    assert captured.err == ""


def test_microduck_missing_python_action_uses_its_supported_setup_command(tmp_path):
    path = initialize("microduck", tmp_path / "recipe")
    _select_python(path, tmp_path / "missing-venv/bin/python")
    report = configuration.check_report(runner.load_example(path))
    action = next(item["next_action"] for item in report["issues"] if item["code"] == "python_missing")
    assert action.endswith(" setup")
    assert "--mode" not in action


def test_existing_init_directory_prints_a_reuse_action_without_changing_files(tmp_path, capsys):
    directory = tmp_path / "recipe"
    path = initialize("xlerobot", directory)
    original = path.read_bytes()
    assert runner.main(["init", "xlerobot", "--output", str(directory)]) == 2
    assert path.read_bytes() == original
    assert str(path) in capsys.readouterr().err


def test_refused_launch_records_failure_and_prints_the_retry_without_starting_services(tmp_path, monkeypatch, capsys):
    path = initialize("xlerobot", tmp_path / "recipe")

    def refused(*args):
        raise RuntimeError("this recipe already has a running launcher")

    monkeypatch.setattr(runner, "Supervisor", refused)
    monkeypatch.setattr(runner, "execute", lambda *a, **kw: pytest.fail("started a child service"))
    assert runner.main([str(path), "setup", "--mode", "software"]) == 2
    report_path = next((path.parent / "runs").glob("*/run.json"))
    assert json.loads(report_path.read_text())["status"] == "failed"
    captured = capsys.readouterr()
    assert str(report_path.parent) in captured.err
    assert "Next:" in captured.err and "setup --mode software" in captured.err
