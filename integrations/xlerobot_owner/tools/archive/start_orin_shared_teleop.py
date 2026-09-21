"""Start the sole robot service using a fresh read-only held-pose handoff.

Refuses busy devices, moving motors or unverified calibration. Does not enable
motion, write registers, restart another service, or disable existing torque.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from embodirun_xlerobot_owner.hardware import WHEEL_NAMES, HardwareRobot
from embodirun_xlerobot_owner.robot import RemoteRobot
from embodirun_xlerobot_owner.stop_fault_handoff import validate_snapshot


def capture_idle(config, token_file, output):
    """Read the still-running service; no device access or stop command."""
    robot = HardwareRobot({**config, "cameras": {}, "allow_motion": False})
    client = RemoteRobot("http://127.0.0.1:8766", token_file.read_text().strip(), scope="base", timeout=3)
    before = client._request("status")
    samples = []
    for _ in range(3):
        observation, _ = client.read()
        samples.append({key: observation.get(key) for key in ("state_timestamp_ns", "state_cached", "errors", "raw")})
        time.sleep(0.06)
    evidence = {
        "source": "physical",
        "boot_id": robot._boot_id(),
        "created_monotonic_s": time.monotonic(),
        "status_before": before,
        "status_after": client._request("status"),
        "samples": samples,
    }
    robot._validate_prior_idle(evidence)
    output.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="prior-idle-", suffix=".json", dir=output)
    with os.fdopen(fd, "w") as handle:
        json.dump(evidence, handle, indent=2)
    print(name, flush=True)


def capture_stop_fault(config, token_file, output, legacy_arm_run=None, no_enable_since=False):
    """Read old API only; preserve its fault, never open devices or stop it."""
    robot = HardwareRobot({**config, "cameras": {}, "allow_motion": False})
    client = RemoteRobot("http://127.0.0.1:8766", token_file.read_text().strip(), scope="base", timeout=3)
    before = client._request("status")
    samples, previous = [], -1
    deadline = time.monotonic() + 5
    while len(samples) < 3 and time.monotonic() < deadline:
        observation, _ = client.read()
        stamp = observation.get("state_timestamp_ns")
        if type(stamp) is int and stamp > previous and observation.get("state_cached") is False:
            samples.append({k: observation.get(k) for k in ("state_timestamp_ns", "state_cached", "errors", "raw")})
            previous = stamp
        time.sleep(0.06)
    after = client._request("status")
    snapshot = {
        "source": "physical",
        "boot_id": robot._boot_id(),
        "ports": config["ports"],
        "created_monotonic_s": time.monotonic(),
        "status_before": before,
        "status_after": after,
        "samples": samples,
        "failed_stop": after.get("feedback"),
    }
    if legacy_arm_run is not None:
        snapshot["legacy_stop_only_review"] = {
            "last_arm_run": legacy_arm_run,
            "no_enable_attempts_since": no_enable_since,
        }
    validate_snapshot(robot, snapshot)
    output.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="stop-fault-", suffix=".json", dir=output)
    with os.fdopen(fd, "w") as handle:
        json.dump(snapshot, handle, indent=2)
    print(name, flush=True)
    return name


def require_idle_devices(config):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8766)) == 0:
            raise RuntimeError("8766 already in use; reuse the existing service, do not replace it")
    devices = list(config["ports"].values()) + list(config.get("cameras", {}).values())
    busy = subprocess.run(["fuser", *devices], capture_output=True, text=True, check=False)
    if busy.returncode == 0 or busy.stdout.strip():
        raise RuntimeError("hardware already in use: " + busy.stdout.strip())


def prepare(config, output, prior_idle=None):
    require_idle_devices(config)
    robot = HardwareRobot({**config, "cameras": {}, "allow_motion": False})
    try:
        if prior_idle is not None:
            robot._validate_prior_idle(prior_idle)
        robot.connect()
        records = {name: record for bus in robot._last_audit["buses"].values() for name, record in bus.items()}
        positions = [n for n in robot._motor_names_for_control() if n not in WHEEL_NAMES]
        goals, cold = {}, []
        for name in positions:
            fields = records[name]["fields"]
            if fields.get("Torque_Enable") == 1:
                goals[name] = fields["Goal_Position"]
            else:
                cold.append(name)
        wheel_torque = (
            {n: records[n]["fields"]["Torque_Enable"] for n in WHEEL_NAMES} if prior_idle is not None else None
        )
        if wheel_torque is not None and any(type(t) is not int or t not in (0, 1) for t in wheel_torque.values()):
            raise RuntimeError("invalid wheel torque state")
        _, audited, errors = robot._preflight(expected_hold=goals, expected_zero_wheels=wheel_torque)
        for name in WHEEL_NAMES:
            fields = audited[name]["fields"]
            if (prior_idle is None and fields.get("Torque_Enable") != 0) or fields.get("Goal_Velocity") != 0:
                errors.append(f"{name}: wheels must already be torque-off with zero goals")
        if errors:
            raise RuntimeError("; ".join(errors))
        stopped = robot._stop_locked("read-only fresh handoff", write_commands=False)
        if stopped.get("stop_confirmed") is not True:
            raise RuntimeError("stationary feedback not confirmed: " + str(stopped))
        snapshot = {
            "source": "physical",
            "stop_confirmed": True,
            "created_monotonic_s": time.monotonic(),
            "ports": config["ports"],
            "enable_base": False,
            "goals": goals,
            "cold_motors": cold,
            "stationary_wait": {n: r["stationary_wait"] for n, r in audited.items() if "stationary_wait" in r},
        }
        if prior_idle is not None:
            snapshot.update(wheel_torque=wheel_torque, prior_idle=prior_idle)
        output.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="held-", suffix=".json", dir=output)
        with os.fdopen(fd, "w") as handle:
            json.dump(snapshot, handle, indent=2)
        print(json.dumps({"handoff": name, "held_goals": goals, "cold_motors": cold, "register_writes": 0}), flush=True)
        return name
    finally:
        robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--capture-idle", action="store_true", help="save strict idle evidence via existing API, then exit"
    )
    mode.add_argument(
        "--capture-stop-fault", action="store_true", help="save unchanged failed all-stop evidence via API, then exit"
    )
    mode.add_argument(
        "--resume-stop-fault", type=Path, help="consume fresh fault evidence; start blocked, no stop/arm/torque writes"
    )
    parser.add_argument(
        "--legacy-arm-run", type=Path, help="old API only: reviewed result.json containing latest successful arm"
    )
    parser.add_argument(
        "--confirm-no-enable-since",
        action="store_true",
        help="operator confirms no enable ATTEMPTS since that run, in this old service",
    )
    parser.add_argument(
        "--prior-idle", type=Path, help="fresh pre-shutdown idle evidence; permits retaining zero-speed wheel torque"
    )
    args = parser.parse_args()
    if (
        (args.legacy_arm_run or args.confirm_no_enable_since)
        and not args.capture_stop_fault
        or bool(args.legacy_arm_run) != args.confirm_no_enable_since
    ):
        parser.error("legacy evidence and its operator assertion are paired, capture-stop-fault only")
    if args.prior_idle and (args.capture_idle or args.capture_stop_fault or args.resume_stop_fault):
        parser.error("--prior-idle cannot be mixed with capture or fault handoff")
    config = json.loads(args.config.read_text())
    if args.capture_idle:
        capture_idle(config, args.token_file, args.output)
        return
    if args.capture_stop_fault:
        capture_stop_fault(
            config,
            args.token_file,
            args.output,
            json.loads(args.legacy_arm_run.read_text()) if args.legacy_arm_run else None,
            args.confirm_no_enable_since,
        )
        return
    if args.resume_stop_fault:
        require_idle_devices(config)
        # Keep deployment independent of the older AGX __main__.py. The new
        # hardware + platform are already blocked before the app can serve.
        from aiohttp import web

        from embodirun_xlerobot_owner.server import Platform, create_app

        source = args.resume_stop_fault
        used = source.with_suffix(source.suffix + ".used")
        if used.exists():
            raise RuntimeError("fault restart snapshot was already claimed")
        source.rename(used)
        config["_resume_stop_fault"] = json.loads(used.read_text())
        platform = Platform(
            HardwareRobot(config), args.token_file.read_text().strip(), output=args.output / "episodes", robot_api=True
        )
        web.run_app(create_app(platform), host="127.0.0.1", port=8766, access_log=None)
        return
    prior_idle = json.loads(args.prior_idle.read_text()) if args.prior_idle else None
    snapshot = prepare(config, args.output, prior_idle)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            "-m",
            "embodirun_xlerobot_owner",
            "serve",
            "--mode",
            "hardware",
            "--robot-api",
            "--host",
            "127.0.0.1",
            "--port",
            "8766",
            "--hardware-config",
            str(args.config),
            "--resume-held-state",
            snapshot,
            "--token-file",
            str(args.token_file),
            "--output",
            str(args.output / "episodes"),
        ],
    )


if __name__ == "__main__":
    main()
