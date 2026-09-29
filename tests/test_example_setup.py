"""Exercise recipe preparation, CLI compatibility and owned-process cancellation."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
from examples import lifecycle, runner
from examples.configuration import TEMPLATES, check_requirements, initialize
from examples.lifecycle import Supervisor, stop_owned
from examples.setup_environment import install_commands

from embodirun.services.host.cli.cli import main as host_main


@pytest.mark.parametrize("template", TEMPLATES)
def test_generated_recipe_resolves_references_outside_checkout(template, tmp_path, monkeypatch):
    destination = tmp_path / "a folder with spaces" / template
    path = initialize(template, destination)
    monkeypatch.chdir(tmp_path)
    example = runner.load_example(path)
    assert example.output == destination / "runs"
    assert runner.main([str(path), "validate"]) == 0
    assert runner.main([str(path), "plan"]) == 0
    if template == "xlerobot":
        hardware = json.loads((destination / "hardware.local.json").read_text())
        assert hardware["allow_motion"] is False
        assert hardware["camera_roles_confirmed"] is False
        task = json.loads(example.resolve(example.parameters["config"]).read_text())
        assert all((destination / ref).is_file() for ref in task["navigation"]["routes"].values())


def test_init_never_overwrites_existing_user_configuration(tmp_path):
    target = tmp_path / "robot"
    target.mkdir()
    calibration = target / "hardware.local.json"
    calibration.write_text("operator calibration")
    with pytest.raises(FileExistsError):
        initialize("xlerobot", target)
    assert calibration.read_text() == "operator calibration"
    assert list(target.iterdir()) == [calibration]


def test_microduck_asset_root_derives_paths_and_retains_inventory(tmp_path):
    assets = tmp_path / "external assets"
    path = initialize("microduck", tmp_path / "recipe", assets=assets)
    example = runner.load_example(path)
    assert example.parameters["checkpoint"] == str(assets / "lightnav_sft_v3/merged")
    assert example.parameters["episodes"] == str(assets / "data/demo_microduck_vln.jsonl")
    assert (
        example.resolve(example.parameters["manifest"]).read_bytes()
        == (runner.ROOT / "examples/microduck_vln/assets.md5.json").read_bytes()
    )


def test_check_lists_hardware_preconditions_without_opening_devices(tmp_path, monkeypatch, capsys):
    path = initialize("xlerobot", tmp_path / "robot")
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("check started a child"))
    assert runner.main([str(path), "check"]) == 2
    output = capsys.readouterr().err
    assert "calibration" in output
    assert "camera_roles_confirmed" in output
    assert "software fixture" in output
    assert not (path.parent / "runs").exists()


def test_check_accepts_inline_calibration_and_camera_indices_or_streams(tmp_path):
    path = initialize("xlerobot", tmp_path / "robot")
    hardware_path = path.parent / "hardware.local.json"
    hardware = json.loads(hardware_path.read_text())
    hardware["calibration"] = {"motors": {}}
    hardware["cameras"] = {
        "front": 0,
        "left_wrist": "rtsp://camera.local/live",
        "right_wrist": None,
    }
    hardware_path.write_text(json.dumps(hardware))
    problems = check_requirements(runner.load_example(path))
    assert any("camera role right_wrist has no source" in item for item in problems)
    assert not any("calibration does not exist" in item for item in problems)
    assert not any("camera role front does not exist" in item for item in problems)
    assert not any("camera role left_wrist does not exist" in item for item in problems)


def test_rollout_check_reports_exact_placeholder_location():
    example = runner.load_example(runner.ROOT / "examples/multi_robot_serving/example.yaml")
    problems = check_requirements(example)
    assert any("nodes.gpu.connection.host" in item for item in problems)
    assert any("robots.arm-1.calibration_dir" in item for item in problems)


def test_cli_and_shell_launcher_have_the_same_validation_output(tmp_path):
    path = initialize("xlerobot", tmp_path / "recipe")
    cli = subprocess.run(
        [
            sys.executable,
            "-c",
            "from embodirun.services.host.cli.cli import main; raise SystemExit(main())",
            "example",
            str(path),
            "validate",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    shell = subprocess.run(
        ["bash", str(runner.ROOT / "examples/run.sh"), str(path), "validate"],
        env={**os.environ, "EMBODIRUN_EXAMPLE_PYTHON": sys.executable},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert cli.stdout == shell.stdout


def test_deployment_cli_still_requires_config():
    with pytest.raises(SystemExit) as error:
        host_main(["validate"])
    assert error.value.code == 2


def test_profile_setup_does_not_need_assets_and_uses_one_dependency_source(tmp_path):
    example = runner.load_example(runner.ROOT / "examples/microduck_vln/example.yaml")
    command = runner.commands(example, "setup", tmp_path)[0][0]
    assert command[-1] == "microduck"
    for profile in ("snack", "microduck"):
        commands = install_commands(profile, tmp_path / profile)
        assert "--frozen" in commands[0]
        assert commands[0][commands[0].index("--python") + 1] == "3.12"
        assert str(runner.ROOT / "integrations") in commands[1][-1]
        assert ("--torch-backend" in commands[1]) is (profile == "snack")
        if profile == "snack":
            assert commands[1][commands[1].index("--torch-backend") + 1] == "cpu"


def test_duplicate_launch_does_not_remove_first_owners_socket(tmp_path):
    with Supervisor(tmp_path, "run") as owner:
        with pytest.raises(RuntimeError, match="already running"), Supervisor(tmp_path, "run"):
            pass
        assert owner.pointer.exists()


def test_stale_stop_socket_does_not_signal_any_pid(tmp_path, monkeypatch):
    (tmp_path / ".launcher-run.json").write_text(json.dumps({"socket": str(tmp_path / "missing.sock")}))
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("must not signal a stored PID"))
    with pytest.raises(RuntimeError, match="Cannot confirm"):
        stop_owned(tmp_path)


def test_stale_run_does_not_skip_stopping_a_live_up_launcher(tmp_path, monkeypatch, capsys):
    (tmp_path / ".launcher-run.json").write_text("{bad json")
    (tmp_path / ".launcher-up.json").write_text(json.dumps({"socket": "/private/up.sock"}))
    contacted = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def connect(self, endpoint):
            contacted.append(endpoint)

        def sendall(self, value):
            assert value == b"stop\n"

        def recv(self, count):
            return b"exited\n"

    monkeypatch.setattr(lifecycle.socket, "socket", lambda *args: FakeClient())
    with pytest.raises(RuntimeError, match="Cannot confirm run cleanup"):
        stop_owned(tmp_path)
    assert contacted == ["/private/up.sock"]
    assert "up: launcher finished" in capsys.readouterr().out


@pytest.mark.parametrize("stop_method", ["down", "sigterm"])
def test_stop_waits_for_cleanup_and_leaves_no_owned_children(tmp_path, stop_method):
    script = r"""
import os, pathlib, signal, subprocess, sys, time
from examples.lifecycle import Supervisor
root = pathlib.Path(sys.argv[1])
with Supervisor(root, 'run') as supervisor:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    (root / 'child').write_text(str(child.pid))
    (root / 'ready').touch()
    try:
        while not supervisor.cancelled.wait(.02):
            pass
    finally:
        child.terminate()
        child.wait(timeout=5)
        time.sleep(.1)
        (root / 'cleaned').touch()
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], cwd=runner.ROOT)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail("launcher did not start")
            time.sleep(0.02)
        if stop_method == "down":
            stop_owned(tmp_path)
            assert (tmp_path / "cleaned").exists()
        else:
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "child").read_text()), 0)
        assert not (tmp_path / ".launcher-run.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_down_works_even_if_referenced_configuration_was_removed(tmp_path):
    path = initialize("xlerobot", tmp_path / "recipe")
    (path.parent / "config.local.json").unlink()
    assert runner.main([str(path), "down"]) == 0


def test_generated_xlerobot_dry_run_uses_fixtures_without_git_metadata(tmp_path, monkeypatch):
    path = initialize("xlerobot", tmp_path / "recipe")

    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(runner.subprocess, "check_output", no_git)
    assert runner.main([str(path), "dry-run"]) == 0
    report = json.loads(next((path.parent / "runs").glob("*/run.json")).read_text())
    assert report["status"] == "complete"
    assert report["revision"] is None
