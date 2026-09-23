"""Exercise the installed EmbodiRun commands."""

import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

from embodirun.services.host.config import config_digest, load_config
from embodirun.services.host.plan import build_plan


def test_installed_cli_commands_exist() -> None:
    entries = {
        entry.name: entry for entry in distribution("embodirun").entry_points if entry.group == "console_scripts"
    }
    for name in (
        "embodirun",
        "embodirun-control-serve",
        "embodirun-simulation-serve",
        "embodirun-sglang-serve",
        "embodirun-go2-streamvln",
    ):
        assert name in entries, name
        if name == "embodirun-sglang-serve":
            pytest.importorskip("sglang.multimodal_gen")
        assert entries[name].load() is not None
        result = subprocess.run(
            [str(Path(sys.executable).parent / name), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_cli_validates_existing_config() -> None:
    config = Path(__file__).parents[1] / "configs/pi05/bi-so101-embodiinfer.yaml"
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "embodirun"),
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


def test_cli_honors_a_stopped_service_without_replaying_or_migrating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The installed entry point honors a stopped service in existing v1 state."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_path = Path(__file__).parents[1] / "configs/pi05/bi-so101-embodiinfer.yaml"
    config = load_config(config_path)
    plan = build_plan(config)
    runtime = plan.runtimes[0]
    state_root = tmp_path / ".local/state/embodirun"
    state_root.mkdir(parents=True)
    state_file = state_root / f"{plan.name}.json"
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
        (state_root / name).write_bytes(b"existing data; do not migrate")

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
        pytest.fail("reading stopped state must not execute any node commands")

    entries = {entry.name: entry for entry in distribution("embodirun").entry_points}
    result = entries["embodirun"].load()(
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
    # the stopped-service state is honored in place; no legacy directory exists
    # and nothing was replayed or migrated
    assert not (tmp_path / ".local/state/rlinf-deploy").exists()
