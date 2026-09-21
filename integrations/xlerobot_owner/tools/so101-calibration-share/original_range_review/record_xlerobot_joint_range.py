#!/usr/bin/env python3
"""Read-only range observations while a human manually moves one joint."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def record_joint_range(
    args: argparse.Namespace,
    inspector: Any,
    *,
    stop_event: Event | None = None,
    sleep_fn=None,
    now_fn=time.monotonic,
    ready_callback=None,
    sample_callback=None,
) -> tuple[dict[str, Any], int]:
    stop_event = stop_event or Event()
    if args.output_dir.exists():
        raise RuntimeError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    try:
        if not math.isfinite(args.duration) or not 5 <= args.duration <= 60:
            raise ValueError("duration must be finite and between 5 and 60 seconds")
        if not math.isfinite(args.interval) or args.interval < 0.05 or args.interval > args.duration:
            raise ValueError("interval must be finite, >= 0.05, and <= duration")
    except (TypeError, ValueError) as exc:
        report = {"status": "failed", "final_report": "observed_range_only", "error": str(exc), "samples": []}
        _write(args.output_dir / "joint_range.json", report)
        return report, 1
    try:
        saved = json.loads(args.calibration.read_text()) if args.calibration and args.calibration.exists() else {}
    except Exception as exc:  # noqa: BLE001  # persist malformed calibration evidence
        report = {
            "status": "failed",
            "final_report": "observed_range_only",
            "error": f"calibration_read_failed:{type(exc).__name__}: {exc}",
            "samples": [],
        }
        _write(args.output_dir / "joint_range.json", report)
        return report, 1
    if not isinstance(saved, dict):
        report = {
            "status": "failed",
            "final_report": "observed_range_only",
            "error": "calibration_json_must_be_object",
            "samples": [],
        }
        _write(args.output_dir / "joint_range.json", report)
        return report, 1
    calibration = saved.get(args.joint)
    if calibration is not None and not isinstance(calibration, dict):
        report = {
            "status": "failed",
            "final_report": "observed_range_only",
            "error": "calibration_joint_entry_must_be_object",
            "samples": [],
        }
        _write(args.output_dir / "joint_range.json", report)
        return report, 1
    if not (args.joint.startswith("left_arm_") or args.joint.startswith("right_arm_")):
        calibration = None
    report: dict[str, Any] = {
        "status": "failed",
        "final_report": "observed_range_only",
        "joint": args.joint,
        "source": "physical_manual_joint_observation",
        "action": None,
        "calibration_input": {
            "provided": bool(args.calibration),
            "path": str(args.calibration) if args.calibration else None,
        },
        "old_range": {"range_min": calibration.get("range_min"), "range_max": calibration.get("range_max")}
        if calibration
        else None,
        "samples": [],
        "observed_min": None,
        "observed_max": None,
        "error": None,
        "source_context": {"port": args.port, "sdk_src": getattr(args, "sdk_src", None), "joint": args.joint},
        "range_usable": False,
        "encoder_wrap_or_discontinuity": False,
        "inference_limits": "Observed range only; no full-range or safe-limit inference.",
    }
    if calibration is None:
        report["error"] = "joint_missing_from_calibration"
        _write(args.output_dir / "joint_range.json", report)
        return report, 1
    bus = None
    try:
        bus = inspector.FeetechMotorsBus(
            port=args.port,
            motors={
                args.joint: inspector.Motor(int(calibration["id"]), "sts3215", inspector.MotorNormMode.RANGE_M100_100)
            },
        )
        bus.connect(handshake=False)
        if hasattr(bus, "set_baudrate"):
            bus.set_baudrate(bus.default_baudrate)
        motor_id = int(calibration["id"])
        model_number = bus.ping(motor_id, num_retry=0, raise_on_error=False)
        report["motor_id"] = motor_id
        report["model_number"] = model_number
        if model_number != 777:
            report["error"] = f"unexpected_model_number:{model_number}"
        else:
            preflight = {}
            for field in (
                "Homing_Offset",
                "Min_Position_Limit",
                "Max_Position_Limit",
                "Operating_Mode",
                "Torque_Enable",
            ):
                preflight[field] = bus.read(field, args.joint, normalize=False)
            report["preflight"] = {
                "actual": preflight,
                "expected": {
                    "Homing_Offset": calibration.get("homing_offset"),
                    "Min_Position_Limit": calibration.get("range_min"),
                    "Max_Position_Limit": calibration.get("range_max"),
                    "Operating_Mode": 0,
                    "Torque_Enable": 0,
                },
            }
            if any(
                preflight[field] != report["preflight"]["expected"][field]
                for field in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")
            ):
                report["error"] = "calibration_eeprom_mismatch"
            elif preflight["Operating_Mode"] != 0:
                report["error"] = "operating_mode_not_position"
            elif preflight["Torque_Enable"] != 0:
                report["error"] = "torque_enabled_abort"
            else:
                if ready_callback:
                    ready_callback(report)
                started_wall = time.time()
                started_mono = now_fn()
                report["capture_started_at"] = started_wall
                while not stop_event.is_set() and now_fn() - started_mono < args.duration:
                    sample_started = now_fn()
                    sample: dict[str, Any] = {
                        "source_host_time": time.time(),
                        "monotonic": sample_started,
                        "fields": {},
                    }
                    try:
                        for field in ("Present_Position", "Present_Velocity", "Torque_Enable", "Moving"):
                            sample["fields"][field] = bus.read(field, args.joint, normalize=False)
                        if sample["fields"]["Torque_Enable"] != 0:
                            sample["error"] = "torque_enabled_abort"
                            report["samples"].append(sample)
                            report["error"] = sample["error"]
                            break
                        position = sample["fields"]["Present_Position"]
                        if (
                            report["samples"]
                            and abs(position - report["samples"][-1]["fields"].get("Present_Position", position)) > 2048
                        ):
                            report["encoder_wrap_or_discontinuity"] = True
                            report["range_usable"] = False
                        report["observed_min"] = (
                            position if report["observed_min"] is None else min(report["observed_min"], position)
                        )
                        report["observed_max"] = (
                            position if report["observed_max"] is None else max(report["observed_max"], position)
                        )
                    except Exception as exc:  # noqa: BLE001  # preserve partial raw sample
                        sample["error"] = f"{type(exc).__name__}: {exc}"
                        report["error"] = sample["error"]
                    report["samples"].append(sample)
                    if sample_callback:
                        sample_callback(sample, report)
                    if report["error"]:
                        break
                    if sleep_fn is None:
                        remaining = max(0.0, args.duration - (now_fn() - started_mono))
                        stop_event.wait(min(args.interval, remaining))
                    else:
                        sleep_fn(args.interval)
                report["capture_ended_at"] = time.time()
                if not report["error"]:
                    report["status"] = "complete" if not stop_event.is_set() else "stopped_partial"
                    if not report["samples"]:
                        report["status"] = "no_samples"
    except KeyboardInterrupt:
        stop_event.set()
        report["status"] = "stopped_partial"
        report["interrupted"] = True
        report["capture_ended_at"] = time.time()
    except Exception as exc:  # noqa: BLE001  # preserve setup/connection failure
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if bus is not None:
            cleanup_error = inspector.close_read_only(bus)
            if cleanup_error:
                report["cleanup_error"] = cleanup_error
                report["error"] = report["error"] or cleanup_error
    _write(args.output_dir / "joint_range.json", report)
    status = (
        0
        if report["status"] in {"complete", "stopped_partial"} and report["samples"] and not report.get("error")
        else 1
    )
    return report, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-src", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 5 <= args.duration <= 60:
        parser.error("--duration must be finite and between 5 and 60 seconds")
    if not math.isfinite(args.interval) or args.interval < 0.05:
        parser.error("--interval must be finite and at least 0.05 seconds")
    if not args.calibration.exists():
        parser.error(f"calibration file does not exist: {args.calibration}")
    import inspect_xlerobot_motors as inspector

    inspector.FeetechMotorsBus, inspector.Motor, inspector.MotorNormMode = inspector.import_sdk(args.sdk_src)
    stop_event = Event()

    def stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        _, status = record_joint_range(args, inspector, stop_event=stop_event)
    except Exception as exc:  # noqa: BLE001  # report CLI failure
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
