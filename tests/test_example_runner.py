"""Exercise example dispatch and safety gates without hardware or optional SDKs."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest
import yaml
from examples import configuration, runner

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
    assert command[0] == example.python
    assert env["MUJOCO_GL"] == "egl"
    assert "integrations/microduck_vln/src" in env["PYTHONPATH"]


@pytest.mark.parametrize("as_json", [False, True])
def test_microduck_software_check_needs_no_assets_and_never_starts_a_launcher(tmp_path, monkeypatch, capsys, as_json):
    path = configuration.initialize("microduck", tmp_path / "recipe with spaces")
    data = yaml.safe_load(path.read_text())
    data["python"] = sys.executable
    path.write_text(yaml.safe_dump(data))

    def metadata(python, packages):
        assert "embodirun-microduck" in packages
        assert not {"torch", "mujoco", "numpy", "Pillow"}.intersection(packages)
        return {"version": [3, 12, 3], "packages": dict.fromkeys(packages, "installed"), "sdk_src": None}

    monkeypatch.setattr(configuration, "_environment_details", metadata)
    monkeypatch.setattr(runner, "Supervisor", lambda *a, **k: pytest.fail("software check opened a launcher"))
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("software check started a scene"))
    arguments = [str(path), "check", "--mode", "software", *(["--json"] if as_json else [])]
    assert runner.main(arguments) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        report = json.loads(output.out)
        assert report["mode"] == "software" and report["issues"] == []
        assert any("CUDA/EGL" in item for item in report["unverified"])
        assert any("External scene" in item for item in report["unverified"])
    else:
        assert "scene/CUDA/model was not tested" in output.out
    assert not (path.parent / "runs").exists()


@pytest.mark.parametrize(
    "template,mode,message",
    [
        ("microduck", "hardware", "MicroDuck setup/check mode must be software or simulation"),
        ("microduck", "unknown", "MicroDuck setup/check mode must be software or simulation"),
        ("xlerobot", "simulation", "XLeRobot setup/check mode must be software or hardware"),
    ],
)
def test_recipe_mode_errors_are_actionable_json_before_a_process_starts(
    tmp_path, monkeypatch, capsys, template, mode, message
):
    path = configuration.initialize(template, tmp_path / "recipe")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("mode error started a process"))
    assert runner.main([str(path), "check", "--mode", mode, "--json"]) == 2
    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report["status"] == "needs_attention"
    assert message in report["issues"][0]["message"]


def test_microduck_software_missing_environment_explains_the_matching_setup(tmp_path, capsys):
    path = configuration.initialize("microduck", tmp_path / "recipe")
    data = yaml.safe_load(path.read_text())
    data["python"] = str(tmp_path / "missing environment/bin/python")
    path.write_text(yaml.safe_dump(data))
    assert runner.main([str(path), "check", "--mode", "software", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in report["issues"]] == ["python_missing"]
    assert "setup --mode software" in report["issues"][0]["next_action"]


def test_microduck_simulation_json_rejects_an_empty_inference_directory(tmp_path):
    path = configuration.initialize("microduck", tmp_path / "recipe")
    example = runner.load_example(path)
    inference = tmp_path / "empty inference directory"
    inference.mkdir()
    example.parameters["inference_root"] = str(inference)
    report = configuration.check_report(example, mode="simulation")
    issue = next(item for item in report["issues"] if item["code"] == "inference_source_missing")
    assert "embodiinfer/__init__.py" in issue["message"]
    assert "submodule update --init third_party/embodiinfer" in issue["next_action"]


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
    output = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
    assert (output / report["input_snapshots"][str(deployment)]).read_text() == deployment.read_text()


def test_rollout_down_still_stops_host_when_launcher_pointer_is_stale(tmp_path, monkeypatch, capsys):
    source = runner.ROOT / "examples" / CONFIGS[0]
    data = yaml.safe_load(source.read_text())
    data["output_dir"] = str(tmp_path / "outputs")
    data["parameters"]["deployment"] = str(source.parent / "deployment.yaml")
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(data))
    example = runner.Example(path, data)

    def stale_stop(_output):
        raise RuntimeError("stale pointer")

    monkeypatch.setattr(runner, "stop_owned", stale_stop)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "test-revision\n")

    class FakeSupervisor:
        def __init__(self, *args):
            self.cancelled = type("Cancelled", (), {"is_set": lambda self: False})()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(runner, "Supervisor", FakeSupervisor)
    executed = []
    monkeypatch.setattr(runner, "execute", lambda commands, *a, **k: executed.extend(commands))

    assert runner.main([str(path), "down"]) == 2
    assert executed == [example.host("down")]
    assert "Local launcher stop unconfirmed: stale pointer" in capsys.readouterr().err
    report = json.loads(next((tmp_path / "outputs").glob("*/run.json")).read_text())
    assert report["status"] == "failed"


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


@pytest.mark.parametrize("outcome", ["complete", "failure", "down"])
def test_command_cleanup_releases_a_stubborn_descendant_listener(tmp_path, outcome):
    from examples.lifecycle import Supervisor, stop_owned

    ready = tmp_path / "descendant.json"
    descendant = r"""
import json, os, pathlib, signal, socket, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen()
pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'port': listener.getsockname()[1]}))
while True:
    connection, _ = listener.accept()
    connection.close()
"""
    wait_ready = (
        "import pathlib, time; ready=pathlib.Path(" + repr(str(ready)) + "); deadline=time.monotonic()+5\n"
        "while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "assert ready.exists(), 'descendant did not start'\n"
    )
    leader = (
        "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', "
        + repr(descendant)
        + ", "
        + repr(str(ready))
        + "])\n"
        + wait_ready
        + ("time.sleep(60)" if outcome != "complete" else "")
    )
    commands = [[sys.executable, "-c", leader]]
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stopper = None
    stop_errors = []
    try:
        if outcome == "failure":
            commands.append([sys.executable, "-c", wait_ready + "raise SystemExit(1)"])
            with pytest.raises(RuntimeError, match="rollout failed"):
                runner.execute(commands, dict(os.environ), tmp_path, parallel=True, interactive=False)
        elif outcome == "down":

            def request_stop():
                try:
                    deadline = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    assert ready.exists(), "descendant did not start"
                    stop_owned(tmp_path)
                except BaseException as error:
                    stop_errors.append(error)

            with Supervisor(tmp_path, "run") as supervisor:
                stopper = threading.Thread(target=request_stop, daemon=True)
                stopper.start()
                with pytest.raises(KeyboardInterrupt):
                    runner.execute(
                        commands, dict(os.environ), tmp_path, parallel=False, interactive=False, supervisor=supervisor
                    )
            stopper.join(timeout=5)
            assert not stopper.is_alive()
            assert not stop_errors
        else:
            runner.execute(commands, dict(os.environ), tmp_path, parallel=False, interactive=False)

        assert other.poll() is None, "cleanup stopped a process outside its owned session"
        port = json.loads(ready.read_text())["port"]
        deadline = time.monotonic() + 2
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    pass
            except OSError:
                break
            if time.monotonic() >= deadline:
                pytest.fail("launcher cleanup left its descendant listener running")
            time.sleep(0.01)
    finally:
        other.terminate()
        other.wait(timeout=5)
        if ready.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(json.loads(ready.read_text())["pid"], signal.SIGKILL)
        if stopper is not None:
            stopper.join(timeout=5)


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
