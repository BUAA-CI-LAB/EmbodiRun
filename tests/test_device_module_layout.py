"""Structural regression tests for the canonical device-domain ownership."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_LAYOUT = (
    (
        "embodirun.services.control.devices",
        "embodirun.devices.lifecycle",
        "DeviceManager",
    ),
    (
        "embodirun.services.control.io",
        "embodirun.devices.execution.io",
        "RobotIOScheduler",
    ),
    (
        "embodirun.services.control.arbitration",
        "embodirun.devices.execution.arbitration",
        "RobotControlArbiter",
    ),
    (
        "embodirun.services.control.teleop",
        "embodirun.devices.execution.teleop",
        "ControlHttpTeleopClient",
    ),
    (
        "embodirun.services.control.inputs",
        "embodirun.devices.execution.inputs",
        "ControlInputBridge",
    ),
    (
        "embodirun.services.control.observation_values",
        "embodirun.devices.observations.values",
        "ObservationSnapshot",
    ),
    (
        "embodirun.services.control.observation_store",
        "embodirun.devices.observations.store",
        "ObservationStore",
    ),
    (
        "embodirun.services.control.observation_producer",
        "embodirun.devices.observations.producer",
        "ObservationProducer",
    ),
    (
        "embodirun.services.control.observation_views",
        "embodirun.devices.observations.views",
        "snapshot_payload",
    ),
    (
        "embodirun.services.control.shared_sensors",
        "embodirun.devices.observations.hub",
        "SharedSensorHub",
    ),
    (
        "embodirun.services.control.recordings",
        "embodirun.devices.recording",
        "ObservationRecorder",
    ),
    (
        "embodirun.services.control.observations",
        "embodirun.devices.observations",
        "ObservationSnapshot",
    ),
)


@pytest.mark.parametrize("legacy_name, canonical_name, symbol", _LAYOUT)
def test_legacy_imports_resolve_to_one_canonical_module(legacy_name: str, canonical_name: str, symbol: str) -> None:
    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)
    assert legacy is canonical
    assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_device_domain_has_no_upward_service_dependencies() -> None:
    device_root = _ROOT / "src/embodirun/devices"
    forbidden = {
        "embodirun.application",
        "embodirun.model_services",
        "embodirun.deployment",
        "embodirun.client",
        "embodirun.agents",
        "agents",
    }
    for path in device_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(
            name in forbidden or name.startswith(f"{prefix}.") for name in imported for prefix in forbidden
        ), path


def test_legacy_and_canonical_import_orders_work_in_fresh_processes() -> None:
    code = """
import importlib
orders = (
    ("embodirun.services.control.devices", "embodirun.devices.lifecycle"),
    ("embodirun.devices.lifecycle", "embodirun.services.control.devices"),
    ("embodirun.services.control.observations", "embodirun.devices.observations"),
    ("embodirun.devices.observations", "embodirun.services.control.observations"),
)
for first, second in orders:
    left = importlib.import_module(first)
    right = importlib.import_module(second)
    assert left is right, (first, second, left.__name__, right.__name__)
"""
    environment = os.environ.copy()
    source_root = str(_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(item for item in (source_root, environment.get("PYTHONPATH")) if item)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
