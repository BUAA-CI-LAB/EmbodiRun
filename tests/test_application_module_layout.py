"""Regression checks for canonical application and simulation ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"

PAIRS = (
    ("embodirun.services.control.application", "embodirun.application.api"),
    ("embodirun.services.control.auth", "embodirun.application.auth"),
    ("embodirun.services.control.contracts", "embodirun.application.contracts"),
    (
        "embodirun.services.control.direct_execution",
        "embodirun.application.direct_execution",
    ),
    ("embodirun.services.control.jobs", "embodirun.application.jobs"),
    ("embodirun.services.control.job_store", "embodirun.application.job_store"),
    ("embodirun.services.control.runtime", "embodirun.application.model_loop"),
    ("embodirun.services.control.proposals", "embodirun.application.proposals"),
    (
        "embodirun.services.simulation.contracts",
        "embodirun.application.simulation.contracts",
    ),
    (
        "embodirun.services.simulation.runtime",
        "embodirun.application.simulation.runtime",
    ),
)


def _run_import_order(old_first: bool) -> subprocess.CompletedProcess[str]:
    first = PAIRS if old_first else tuple((new, old) for old, new in PAIRS)
    lines = ["import importlib"]
    for old, new in first:
        lines.extend(
            (
                f"old = importlib.import_module({old!r})",
                f"new = importlib.import_module({new!r})",
                "assert old is new",
            )
        )
    lines.extend(
        (
            "from embodirun.services.control.server import ControlService as old_service",
            "from embodirun.application.control_service import ControlService as new_service",
            "assert old_service is new_service",
            "from embodirun.services.simulation.server import SimulationService as old_simulation",
            "from embodirun.application.simulation.service import SimulationService as new_simulation",
            "assert old_simulation is new_simulation",
            "from embodirun.application import ControlService",
            "assert ControlService is new_service",
            "print('ok')",
        )
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", "\n".join(lines)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_legacy_and_canonical_application_imports_share_identity_in_both_orders() -> None:
    for old_first in (True, False):
        result = _run_import_order(old_first)
        assert result.returncode == 0, result.stderr or result.stdout


def test_application_has_no_dependency_on_legacy_service_entrypoints() -> None:
    for path in (ROOT / "src" / "embodirun" / "application").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "embodirun.services.control" not in text, path
        assert "embodirun.services.simulation" not in text, path


def test_devices_and_model_services_do_not_import_application() -> None:
    for package in ("devices", "model_services"):
        for path in (ROOT / "src" / "embodirun" / package).rglob("*.py"):
            assert "embodirun.application" not in path.read_text(encoding="utf-8"), path
