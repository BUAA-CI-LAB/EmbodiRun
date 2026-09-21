"""Exercise the installed EmbodiRun commands and their compatibility aliases."""

import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

from embodirun.services.host.config import config_digest, load_config
from embodirun.services.host.plan import build_plan


@pytest.mark.parametrize("first,second", [("embodirun", "rlinf_deploy"), ("rlinf_deploy", "embodirun")])
def test_import_aliases_share_modules_in_both_orders(first, second) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import pickle
import sys
for suffix in (
    '', '.robots', '.bindings', '.services.host.config',
    '.services.host.plan', '.services.control.contracts',
    '.services.inference.backends.vvla',
):
    left = importlib.import_module({first!r} + suffix)
    right = importlib.import_module({second!r} + suffix)
    assert left is right, suffix
    assert left.__spec__.name.startswith('embodirun'), suffix
from embodirun.services.host.config import ConfigError
assert pickle.loads(b'crlinf_deploy.services.host.config\\nConfigError\\n.') is ConfigError
assert not {{'torch', 'sglang', 'vvla'}} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "services.control.server",
        "services.simulation.server",
        "services.control.teleop",
    ],
)
def test_new_and_legacy_module_entry_points(module) -> None:
    outputs = []
    for package in ("embodirun", "rlinf_deploy"):
        result = subprocess.run(
            [sys.executable, "-m", f"{package}.{module}", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("current", "legacy"),
    [
        ("embodirun", "rlinf-deploy"),
        ("embodirun-control-serve", "rlinf-control-serve"),
        ("embodirun-simulation-serve", "rlinf-simulation-serve"),
        ("embodirun-sglang-serve", "rlinf-sglang-serve"),
        ("embodirun-go2-streamvln", "rlinf-go2-streamvln"),
    ],
)
def test_installed_cli_aliases(current: str, legacy: str) -> None:
    entries = {
        entry.name: entry for entry in distribution("embodirun").entry_points if entry.group == "console_scripts"
    }
    assert entries[current].value == entries[legacy].value
    if current == "embodirun-sglang-serve":
        pytest.importorskip("sglang.multimodal_gen")
    assert entries[current].load() is entries[legacy].load()
    for name in (current, legacy):
        result = subprocess.run(
            [str(Path(sys.executable).parent / name), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_new_and_legacy_cli_validate_existing_config() -> None:
    config = Path(__file__).parents[1] / "configs/pi05/bi-so101-vvla.yaml"
    outputs = []
    for name in ("embodirun", "rlinf-deploy"):
        result = subprocess.run(
            [
                str(Path(sys.executable).parent / name),
                "--config",
                str(config),
                "validate",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_cli_aliases_read_legacy_state_without_replaying_or_migrating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both installed entry points honor a stopped service in existing v1 state."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_path = Path(__file__).parents[1] / "configs/pi05/bi-so101-vvla.yaml"
    config = load_config(config_path)
    plan = build_plan(config)
    runtime = plan.runtimes[0]
    legacy_root = tmp_path / ".local/state/rlinf-deploy"
    legacy_root.mkdir(parents=True)
    state_file = legacy_root / f"{plan.name}.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "name": plan.name,
                "config_digest": config_digest(config),
                "deploy_commit": plan.deploy_commit,
                "inference_commit": "a" * 40,
                "environments": {
                    runtime.environment_id: {
                        "environment_id": runtime.environment_id,
                        "node": runtime.node,
                        "project": "deploy",
                        "group": "robot-so101",
                        "path": ".venv-robot-so101",
                        "status": "ready",
                    }
                },
                "services": {
                    runtime.service_id: {
                        "service_id": runtime.service_id,
                        "node": runtime.node,
                        "status": "stopped",
                    }
                },
            }
        )
    )
    for name in ("calibration.json", "tasks.sqlite", "recording.bin"):
        (legacy_root / name).write_bytes(b"existing data; do not migrate")

    def snapshot() -> dict[str, tuple[bytes, int]]:
        return {
            str(path.relative_to(tmp_path)): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    original = snapshot()

    def reject_execution(_node: object) -> None:
        pytest.fail("reading stopped legacy state must not execute any node commands")

    entries = {entry.name: entry for entry in distribution("embodirun").entry_points}
    for name in ("embodirun", "rlinf-deploy"):
        result = entries[name].load()(
            [
                "--config",
                str(config_path),
                "run",
                "--runtime",
                runtime.runtime_id,
                "--prompt",
                "must not execute",
            ],
            executor_factory=reject_execution,
        )
        assert result == 1
        assert "is not running" in capsys.readouterr().err
        assert snapshot() == original
        assert not (tmp_path / ".local/state/embodirun").exists()
