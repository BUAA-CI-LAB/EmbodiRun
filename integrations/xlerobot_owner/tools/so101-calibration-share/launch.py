#!/usr/bin/env python3
"""Run one standard LeRobot SO101 calibration from a Mac or robot host."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ARMS = ("arm1", "arm2", "arm3", "arm4")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def load_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"config missing: copy config.example.json to {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return data


def required(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate(config: dict[str, Any], arm_name: str) -> dict[str, str]:
    mode = config.get("mode", "ssh")
    if mode not in {"ssh", "local"}:
        raise ValueError("mode must be 'ssh' or 'local'")
    arms = config.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("arms must contain exactly arm1, arm2, arm3, arm4")
    seen_ids: set[str] = set()
    for name in ARMS:
        entry = arms[name]
        if not isinstance(entry, dict):
            raise ValueError(f"{name} must be an object")
        arm_id = required(entry.get("id", name), f"{name}.id")
        port = required(entry.get("port"), f"{name}.port")
        if not SAFE_NAME.fullmatch(arm_id):
            raise ValueError(f"{name}.id contains unsafe characters")
        if "REPLACE_" in port:
            raise ValueError(f"{name}.port is still a placeholder; edit config before running")
        arm_type = entry.get("type", "so101_follower")
        if arm_type not in {"so101_follower", "so101_leader"}:
            raise ValueError(f"{name}.type must be so101_follower or so101_leader")
        if arm_id in seen_ids:
            raise ValueError("arm ids must be unique")
        seen_ids.add(arm_id)
    entry = arms[arm_name]
    calibration_dir = required(config.get("calibration_dir"), "calibration_dir")
    if not Path(calibration_dir).is_absolute():
        raise ValueError("calibration_dir must be an absolute path")
    item = {
        "mode": mode,
        "id": entry.get("id", arm_name),
        "port": entry["port"],
        "type": entry.get("type", "so101_follower"),
        "python": required(config.get("python", "python3"), "python"),
        "calibration_dir": calibration_dir,
        "sdk_src": config.get("sdk_src", "") or "",
    }
    if mode == "ssh":
        host = required(config.get("host"), "host")
        user = required(config.get("user"), "user")
        if (
            host.startswith("-")
            or user.startswith("-")
            or not SAFE_NAME.fullmatch(host)
            or not SAFE_NAME.fullmatch(user)
        ):
            raise ValueError("host and user must be simple names without shell/options")
        try:
            ssh_port = int(config.get("ssh_port", 22))
        except (TypeError, ValueError) as exc:
            raise ValueError("ssh_port must be an integer") from exc
        if not 1 <= ssh_port <= 65535:
            raise ValueError("ssh_port must be between 1 and 65535")
        item.update(host=host, user=user, ssh_port=str(ssh_port))
    return item


def calibration_args(item: dict[str, str]) -> list[str]:
    prefix = "robot" if item["type"] == "so101_follower" else "teleop"
    return [
        "-m",
        "lerobot.scripts.lerobot_calibrate",
        f"--{prefix}.type={item['type']}",
        f"--{prefix}.port={item['port']}",
        f"--{prefix}.id={item['id']}",
        f"--{prefix}.calibration_dir={item['calibration_dir']}",
    ]


def build_command(item: dict[str, str]) -> list[str]:
    command = [item["python"]]
    if item["sdk_src"]:
        command = ["env", f"PYTHONPATH={item['sdk_src']}", *command]
    return command + calibration_args(item)


def make_command(config: dict[str, Any], arm_name: str) -> tuple[dict[str, str], list[str], str]:
    item = validate(config, arm_name)
    command = build_command(item)
    if item["mode"] == "local":
        return item, command, "local host"
    ssh = ["ssh", "-tt", "-o", "StrictHostKeyChecking=ask", "-p", item["ssh_port"]]
    if config.get("identity_file"):
        ssh += ["-i", config["identity_file"]]
    ssh += [f"{item['user']}@{item['host']}", shlex.join(command)]
    return item, ssh, f"{item['user']}@{item['host']}:{item['ssh_port']}"


def run(config: dict[str, Any], arm_name: str, dry_run: bool) -> int:
    item, command, location = make_command(config, arm_name)
    expected_file = Path(item["calibration_dir"]) / f"{item['id']}.json"
    print(f"Calibrating {arm_name} ({item['id']}, {item['type']}) on {location}")
    print(f"Serial port: {item['port']}")
    print(f"Expected calibration file: {expected_file}")
    print("Command:", shlex.join(command))
    print(
        "Official LeRobot calibration writes calibration parameters and may operate the arm; this wrapper does not set motor IDs or configure torque."
    )
    if dry_run:
        print("Dry run: no SSH connection or subprocess was started.")
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        print(f"Error starting calibration command: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    arm = args.arm
    if arm is None:
        if not sys.stdin.isatty():
            parser.error("--arm is required when stdin is not a terminal")
        arm = input("Choose arm (arm1/arm2/arm3/arm4): ").strip()
        if arm not in ARMS:
            parser.error("arm must be arm1, arm2, arm3, or arm4")
    try:
        return run(load_config(args.config), arm, args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
