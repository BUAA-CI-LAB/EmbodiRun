#!/usr/bin/env python3
"""Inspect or set volatile SO101 arm acceleration while torque is disabled."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from embodirun_xlerobot_owner.hardware import ARM_NAMES, HardwareRobot


def _read_profile(robot: HardwareRobot) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for side in ("left", "right"):
        bus = robot._buses[side]
        for name in ARM_NAMES[side]:
            fields: dict[str, object] = {}
            errors: dict[str, str] = {}
            for field in (
                "Torque_Enable",
                "Present_Position",
                "Goal_Position",
                "Present_Velocity",
                "Moving",
                "Acceleration",
                "Maximum_Acceleration",
                "Lock",
            ):
                value, error = robot._read_register(bus, field, name)
                fields[field] = value
                if error is not None:
                    errors[field] = error
            result[name] = {"fields": fields, "errors": errors}
    return result


def _assert_safe(profile: dict[str, dict[str, object]]) -> None:
    failures: list[str] = []
    for name, record in profile.items():
        fields = record["fields"]
        if record["errors"]:
            failures.append(f"{name}: read errors {record['errors']}")
            continue
        if fields["Torque_Enable"] != 0:
            failures.append(f"{name}: Torque_Enable={fields['Torque_Enable']!r}")
        if fields["Present_Velocity"] != 0 or fields["Moving"] != 0:
            failures.append(f"{name}: moving velocity={fields['Present_Velocity']!r} flag={fields['Moving']!r}")
    if failures:
        raise RuntimeError("refusing volatile profile write: " + "; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="target", type=int)
    parser.add_argument("--motor", action="append", default=[])
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--delay-s", type=float, default=0.1)
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--disable-torque", action="append", default=[])
    args = parser.parse_args()
    if args.target is not None and not 0 <= args.target <= 254:
        parser.error("--set must be within [0, 254]")
    if args.attempts <= 0 or args.delay_s < 0:
        parser.error("--attempts must be positive and --delay-s non-negative")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_ports = dict(config["ports"])
    config["cameras"] = {}
    config["enable_base"] = False
    robot = HardwareRobot(config)
    try:
        robot.connect()
        before = _read_profile(robot)
        output: dict[str, object] = {"before": before, "writes": []}
        known = {name for side in ("left", "right") for name in ARM_NAMES[side]}
        unknown_disable = set(args.disable_torque) - known
        if unknown_disable:
            raise ValueError("unknown --disable-torque motors: " + ", ".join(sorted(unknown_disable)))
        for name in args.disable_torque:
            record = before[name]
            fields = record["fields"]
            if record["errors"]:
                raise RuntimeError(f"{name}: read errors {record['errors']}")
            if fields["Present_Velocity"] != 0 or fields["Moving"] != 0:
                raise RuntimeError(f"{name}: refusing torque disable while moving")
            side = "left" if name.startswith("left_") else "right"
            robot._buses[side].write("Torque_Enable", name, 0, normalize=False)
            time.sleep(args.delay_s)
            actual, error = robot._read_register(robot._buses[side], "Torque_Enable", name)
            writes = output["writes"]
            assert isinstance(writes, list)
            writes.append(
                {
                    "motor": name,
                    "field": "Torque_Enable",
                    "target": 0,
                    "actual": actual,
                    "read_error": error,
                }
            )
            if error is not None or actual != 0:
                raise RuntimeError(f"{name}: torque disable was not confirmed")
        if args.disable_torque:
            before = _read_profile(robot)
            output["after_torque_disable"] = before
        if args.snapshot_output is not None:
            failures: list[str] = []
            goals: dict[str, int] = {}
            cold: list[str] = []
            for name, record in before.items():
                fields = record["fields"]
                if record["errors"]:
                    failures.append(f"{name}: read errors {record['errors']}")
                    continue
                if fields["Torque_Enable"] == 0:
                    cold.append(name)
                elif fields["Torque_Enable"] != 1:
                    failures.append(f"{name}: Torque_Enable={fields['Torque_Enable']!r}")
                if fields["Present_Velocity"] != 0 or fields["Moving"] != 0:
                    failures.append(f"{name}: motor is not stationary")
                try:
                    goal = int(fields["Goal_Position"])
                    present = int(fields["Present_Position"])
                except (TypeError, ValueError) as exc:
                    failures.append(f"{name}: invalid position feedback: {exc}")
                    continue
                if fields["Torque_Enable"] == 1 and not name.endswith("_gripper") and abs(goal - present) > 16:
                    failures.append(f"{name}: Goal_Position={goal} differs from Present_Position={present}")
                if fields["Torque_Enable"] == 1:
                    goals[name] = goal
            if failures:
                raise RuntimeError("refusing held snapshot: " + "; ".join(failures))
            snapshot_path = args.snapshot_output
            used_path = snapshot_path.with_suffix(snapshot_path.suffix + ".used")
            if snapshot_path.exists() or used_path.exists():
                raise FileExistsError("snapshot path or its .used claim already exists")
            snapshot = {
                "source": "physical",
                "stop_confirmed": True,
                "created_monotonic_s": time.monotonic(),
                "ports": source_ports,
                "enable_base": False,
                "goals": goals,
                "cold_motors": cold,
            }
            file_descriptor = os.open(
                snapshot_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(snapshot, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            output["snapshot"] = str(snapshot_path)
        if args.target is not None:
            _assert_safe(before)
            selected = (
                set(args.motor) if args.motor else {name for side in ("left", "right") for name in ARM_NAMES[side]}
            )
            unknown = selected - known
            if unknown:
                raise ValueError("unknown motors: " + ", ".join(sorted(unknown)))
            writes = output["writes"]
            assert isinstance(writes, list)
            for side in ("left", "right"):
                bus = robot._buses[side]
                for name in ARM_NAMES[side]:
                    if name not in selected:
                        continue
                    actual = None
                    error = None
                    for attempt in range(1, args.attempts + 1):
                        bus.write("Acceleration", name, args.target, normalize=False)
                        time.sleep(args.delay_s)
                        actual, error = robot._read_register(bus, "Acceleration", name)
                        writes.append(
                            {
                                "motor": name,
                                "attempt": attempt,
                                "target": args.target,
                                "actual": actual,
                                "read_error": error,
                            }
                        )
                        if error is None and int(actual) == args.target:
                            break
                    else:
                        raise RuntimeError(f"{name}: acceleration remained {actual!r}; read_error={error}")
            output["after"] = _read_profile(robot)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
