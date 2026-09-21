#!/usr/bin/env python3
"""Bounded, read-only inventory for the two Feetech buses used by XLeRobot.

This module deliberately does not import LeRobot at module import time.  The
CLI takes the recovered SDK source explicitly so an installed newer LeRobot
cannot silently change the protocol tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_ID = 10
FIELDS = (
    "Present_Position",
    "Present_Velocity",
    "Moving",
    "Torque_Enable",
    "Operating_Mode",
    "Homing_Offset",
    "Min_Position_Limit",
    "Max_Position_Limit",
)
CONTROL_PROFILE_FIELDS = (
    "Firmware_Major_Version",
    "Firmware_Minor_Version",
    "CW_Dead_Zone",
    "CCW_Dead_Zone",
    "Minimum_Startup_Force",
    "Goal_Time",
    "Goal_Velocity",
    "Acceleration",
    "Moving_Velocity_Threshold",
    "Velocity_Unit_factor",
    "DTs",
    "Hts",
    "Maximum_Velocity_Limit",
    "Maximum_Acceleration",
    "P_Coefficient",
    "I_Coefficient",
    "D_Coefficient",
    "Present_Load",
    "Present_Current",
    "Present_Voltage",
    "Present_Temperature",
)
ROLE_PATTERNS = {
    "left_arm_head_candidate": frozenset(range(1, 9)),
    "right_arm_wheels_candidate": frozenset((1, 2, 3, 4, 5, 6, 9, 10)),
}


def import_sdk(sdk_src: str):
    """Import only the recovered SDK after prepending its source directory."""
    source_path = Path(sdk_src).expanduser().resolve()
    source = str(source_path)
    if not source_path.is_dir():
        raise FileNotFoundError(f"SDK source directory does not exist: {source}")
    if source not in sys.path:
        sys.path.insert(0, source)
    loaded = sys.modules.get("lerobot")
    if (
        loaded is not None
        and getattr(loaded, "__file__", None)
        and source_path not in Path(loaded.__file__).resolve().parents
    ):
        for name in list(sys.modules):
            if name == "lerobot" or name.startswith("lerobot."):
                del sys.modules[name]
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    module_path = Path(sys.modules["lerobot"].__file__).resolve()
    if source_path not in module_path.parents:
        raise ImportError(f"lerobot resolved outside --sdk-src: {module_path}")
    return FeetechMotorsBus, Motor, MotorNormMode


def _motors(Motor, MotorNormMode) -> dict[str, Any]:
    return {
        f"motor_{motor_id}": Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
        for motor_id in range(1, MAX_ID + 1)
    }


def _read(bus: Any, field: str, name: str) -> Any:
    try:
        return {"value": bus.read(field, name, normalize=False)}
    except Exception as exc:  # noqa: BLE001  # retain per-register failures in the artifact
        return {"error": f"{type(exc).__name__}: {exc}"}


def inspect_bus(
    bus: Any, saved_calibration: dict[str, Any] | None = None, include_control_profile: bool = False
) -> dict[str, Any]:
    """Inventory one already-connected bus; no method here writes to a motor."""
    records: list[dict[str, Any]] = []
    responding: list[int] = []
    for motor_id in range(1, MAX_ID + 1):
        name = f"motor_{motor_id}"
        item: dict[str, Any] = {"id": motor_id}
        try:
            model_number = bus.ping(motor_id, num_retry=0, raise_on_error=False)
            item["model_number"] = model_number
            if model_number is not None:
                responding.append(motor_id)
                fields = {field: _read(bus, field, name) for field in FIELDS}
                if include_control_profile:
                    fields["control_profile"] = {field: _read(bus, field, name) for field in CONTROL_PROFILE_FIELDS}
                item["fields"] = fields
        except Exception as exc:  # noqa: BLE001  # retain per-motor ping failures
            item["ping_error"] = f"{type(exc).__name__}: {exc}"
        records.append(item)

    ids = frozenset(responding)
    candidates = [role for role, pattern in ROLE_PATTERNS.items() if ids == pattern]
    if not candidates:
        candidates = ["unknown_or_partial"]
    role = candidates[0] if len(candidates) == 1 else "unknown_or_partial"
    calibration_comparison = compare_calibration(records, saved_calibration or {}, role)
    return {
        "responding_ids": responding,
        "device_status": "no_devices" if not responding else "devices_responded",
        "role_candidates": candidates,
        "role_evidence": "ID pattern only; candidate is not physical proof",
        "motors": records,
        "saved_calibration_comparison": calibration_comparison,
    }


def inventory_report_status(inventory: dict[str, Any]) -> str:
    """Return a process-facing status without treating expected absent IDs as errors."""
    if inventory.get("device_status") != "devices_responded":
        return "error"
    if inventory.get("role_candidates") == ["unknown_or_partial"]:
        return "error"
    for motor in inventory.get("motors", []):
        if "ping_error" in motor:
            return "error"
        for field in motor.get("fields", {}).values():
            if "error" in field:
                return "error"
            if isinstance(field, dict) and any("error" in value for value in field.values() if isinstance(value, dict)):
                return "error"
    return "ok"


def timed_snapshot(bus: Any, calibration: dict[str, Any], include_control_profile: bool = False) -> dict[str, Any]:
    wall_start = time.time()
    mono_start = time.monotonic()
    inventory = inspect_bus(bus, calibration, include_control_profile)
    return {
        "source_host_time_start": wall_start,
        "source_host_time_end": time.time(),
        "monotonic_duration_s": time.monotonic() - mono_start,
        "inventory": inventory,
        "report_status": inventory_report_status(inventory),
    }


def compare_calibration(
    records: list[dict[str, Any]], saved: dict[str, Any], role: str = "unknown_or_partial"
) -> dict[str, Any]:
    """Compare EEPROM values read above with a saved JSON calibration, by ID."""
    if not saved:
        return {"status": "missing_saved_calibration", "motors": {}}
    result: dict[str, Any] = {}
    for record in records:
        motor_id = str(record["id"])
        candidates = [v for k, v in saved.items() if str(v.get("id", "")) == motor_id]
        if role == "left_arm_head_candidate":
            candidates = [
                v for k, v in saved.items() if str(v.get("id", "")) == motor_id and k.startswith(("left_arm_", "head_"))
            ]
        elif role == "right_arm_wheels_candidate":
            candidates = [
                v
                for k, v in saved.items()
                if str(v.get("id", "")) == motor_id and k.startswith(("right_arm_", "base_"))
            ]
        if len(candidates) > 1:
            result[motor_id] = {"status": "ambiguous_saved_motor"}
            continue
        expected = candidates[0] if candidates else None
        if expected is None:
            result[motor_id] = {"status": "missing_saved_motor"}
            continue
        fields = record.get("fields", {})
        actual = {
            key: fields.get(key, {}).get("value")
            for key in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")
        }
        wanted = {
            "Homing_Offset": expected.get("homing_offset"),
            "Min_Position_Limit": expected.get("range_min"),
            "Max_Position_Limit": expected.get("range_max"),
        }
        result[motor_id] = {"status": "match" if actual == wanted else "mismatch", "actual": actual, "expected": wanted}
    return {"status": "compared", "motors": result}


def close_read_only(bus: Any) -> str | None:
    """Close without the SDK's default torque-disabling writes."""
    try:
        if getattr(bus, "is_connected", False):
            bus.disconnect(disable_torque=False)
            return None
    except Exception as exc:  # noqa: BLE001  # cleanup must continue after partial connect
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
    handler = getattr(bus, "port_handler", None)
    close_port = getattr(handler, "closePort", None)
    if callable(close_port):
        try:
            close_port()
        except Exception as exc:  # noqa: BLE001  # preserve cleanup failure in artifact
            error = f"{type(exc).__name__}: {exc}"
    return error


def run_snapshot(
    bus_factory: Callable[[str], Any],
    ports: list[str],
    calibration: dict[str, Any],
    duration: float,
    include_control_profile: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(duration) or duration < 0 or duration > 5:
        raise ValueError("duration must be between 0 and 5 seconds")
    output: dict[str, Any] = {"source_host_time": time.time(), "no_camera_sync": True, "buses": []}
    for port in ports:
        bus = None
        started = time.time()
        try:
            bus = bus_factory(port)
            bus.connect(handshake=False)
            if hasattr(bus, "set_baudrate"):
                bus.set_baudrate(bus.default_baudrate)
            snapshots = [timed_snapshot(bus, calibration, include_control_profile)]
            if duration:
                time.sleep(duration)
                snapshots.append(timed_snapshot(bus, calibration, include_control_profile))
            output["buses"].append({"port": port, "source_host_time": started, "snapshots": snapshots})
        except Exception as exc:  # noqa: BLE001  # preserve per-port failure in artifact
            output["buses"].append({"port": port, "source_host_time": started, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if bus is not None:
                cleanup_error = close_read_only(bus)
                if output["buses"] and cleanup_error:
                    output["buses"][-1]["cleanup_error"] = cleanup_error
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-src", required=True, help="recovered LeRobot 0.4.3 src directory")
    parser.add_argument("--port", action="append", required=True, help="serial port; repeat for both buses")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, exclusive output directory")
    parser.add_argument("--include-control-profile", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.calibration and not args.calibration.exists():
        parser.error(f"calibration file does not exist: {args.calibration}")
    args.output_dir.mkdir(parents=True)
    FeetechMotorsBus, Motor, MotorNormMode = import_sdk(args.sdk_src)
    from lerobot.motors.feetech.feetech import DEFAULT_PROTOCOL_VERSION

    saved = json.loads(args.calibration.read_text()) if args.calibration and args.calibration.exists() else {}
    result = run_snapshot(
        lambda port: FeetechMotorsBus(port=port, motors=_motors(Motor, MotorNormMode)),
        args.port,
        saved,
        args.duration,
        args.include_control_profile,
    )
    result["sdk_defaults"] = {
        "default_baudrate": FeetechMotorsBus.default_baudrate,
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
    }
    (args.output_dir / "motor_inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    failed = any(
        "error" in bus
        or any(snapshot.get("report_status") != "ok" for snapshot in bus.get("snapshots", []))
        or "cleanup_error" in bus
        for bus in result["buses"]
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
