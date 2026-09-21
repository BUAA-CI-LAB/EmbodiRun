"""Regression checks for the canonical deployment module ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"

LEAF_PAIRS = (
    ("environment", "environment"),
    ("network", "network"),
    ("plan", "plan"),
    ("probe", "probe"),
    ("simulation", "simulation"),
    ("source", "source"),
    ("state", "state"),
    ("supervisor", "supervisor"),
    ("control", "control"),
    ("config.loader", "config.loader"),
    ("config.model", "config.model"),
    ("config.validation", "config.validation"),
    ("executor.command", "executor.command"),
    ("executor.executor", "executor.executor"),
)


def _import_script(old_first: bool) -> str:
    first = "embodirun.services.host" if old_first else "embodirun.deployment"
    second = "embodirun.deployment" if old_first else "embodirun.services.host"
    return f"""
import importlib
import sys
old = importlib.import_module({first!r})
new = importlib.import_module({second!r})
assert old.__all__ == new.__all__
for name in old.__all__:
    assert getattr(old, name) is getattr(new, name), name
for suffix, canonical_suffix in {LEAF_PAIRS!r}:
    legacy = importlib.import_module('embodirun.services.host.' + suffix)
    canonical = importlib.import_module('embodirun.deployment.' + canonical_suffix)
    assert legacy is canonical, (suffix, legacy, canonical)
# Package facades retain their historical package objects and paths.
assert importlib.import_module('embodirun.services.host.config') is not importlib.import_module('embodirun.deployment.config')
assert importlib.import_module('embodirun.services.host.executor') is not importlib.import_module('embodirun.deployment.executor')
print('ok')
"""


def _run_python(script: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd or ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_legacy_and_canonical_imports_share_leaf_identity_in_both_orders() -> None:
    for old_first in (True, False):
        result = _run_python(_import_script(old_first))
        assert result.returncode == 0, result.stderr or result.stdout


def test_deployment_domain_does_not_import_host_cli_or_legacy_host_modules() -> None:
    deployment = ROOT / "src" / "embodirun" / "deployment"
    forbidden = ("embodirun.services.host", "embodirun.services.host.cli")
    for path in deployment.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_supervisor_help_works_as_bare_script_from_unrelated_cwd(
    tmp_path: Path,
) -> None:
    for relative in (
        "src/embodirun/deployment/supervisor.py",
        "src/embodirun/services/host/supervisor.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            cwd=tmp_path,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "{start,status,stop}" in result.stdout
