"""Start one local XLeRobot owner and two scope-bound Control services.

Private owner credentials are generated locally. The CLI never accepts or
prints a token. --write-only validates and writes configs without hardware.
A running owner must be stopped by its operator before starting this stack;
processes are not automatically recovered or restarted after faults.
"""

from __future__ import annotations

import argparse
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from embodirun.application.contracts import ControlServiceConfig
from embodirun.robots.sensors import SensorInput


def _private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def prepare(config: dict[str, Any], directory: Path, *, base_dir: Path) -> list[list[str]]:
    ports = [config["owner"]["port"], config["control"]["base_port"], config["control"]["manipulation_port"]]
    if len(set(ports)) != 3 or any(isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 65535 for p in ports):
        raise ValueError("owner, base and manipulation ports must be distinct valid ports")
    hardware = Path(config["owner"]["hardware_config"])
    hardware = hardware if hardware.is_absolute() else base_dir / hardware
    if not hardware.is_file():
        raise ValueError("owner.hardware_config does not exist")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    access = directory / "owner-access"
    if not access.exists():
        _private(access, secrets.token_urlsafe(32))
    credential = access.read_text().strip()
    owner_url = f"http://127.0.0.1:{ports[0]}"
    resources = (
        {
            "identity": "local:robot:xlerobot",
            "node": "local",
            "kind": "robot",
            "value": "xlerobot",
            "external_owner": True,
        },
    )
    common = {"url": owner_url, "token": credential}
    base = ControlServiceConfig.device_only(
        runtime_id="xlerobot-base",
        bind="127.0.0.1",
        port=ports[1],
        robot_id="xlerobot-base",
        robot_kind="lerobot.xlerobot",
        robot_options={**common, "scope": "base"},
        device_resources=resources,
    )
    inputs = tuple(
        SensorInput(
            sensor_id=f"owner-{camera}",
            name=name,
            kind="xlerobot",
            options={**common, "scope": "arms", "camera": camera},
        )
        for name, camera in config["cameras"].items()
    )
    arms = ControlServiceConfig(
        runtime_id="xlerobot-pi05",
        binding_kind="lerobot.xlerobot.pi05",
        bind="127.0.0.1",
        port=ports[2],
        inference_transport="http",
        inference_endpoint=config["model"]["endpoint"],
        inference_backend=config["model"].get("backend", "embodiinfer"),
        inference_options={},
        robot_id="xlerobot-arms",
        robot_kind="lerobot.xlerobot",
        robot_options={**common, "scope": "arms"},
        inputs=inputs,
        runtime_options={},
        device_resources=resources,
    )
    commands = [
        [
            sys.executable,
            "-m",
            "embodirun_xlerobot_owner",
            "serve",
            "--mode",
            "hardware",
            "--host",
            "127.0.0.1",
            "--port",
            str(ports[0]),
            "--token-file",
            str(access),
            "--robot-api",
            "--pair",
            "--hardware-config",
            str(hardware.resolve()),
            "--output",
            str(directory / "episodes"),
        ]
    ]
    for label, service in (("base", base), ("arms", arms)):
        path = directory / f"{label}-control.json"
        _private(path, service.to_json())
        commands.append(
            [
                sys.executable,
                "-m",
                "embodirun.services.control.server",
                "--config",
                str(path),
                "--state-dir",
                str(directory / f"{label}-state"),
            ]
        )
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(args.deployment.read_text())
    commands = prepare(config, args.state_dir.resolve(), base_dir=args.deployment.parent.resolve())
    if args.write_only:
        print("Validated private configs; no services started.")
        return 0
    # Refuse an occupied port; never terminate or replace somebody else's owner.
    for port in (config["owner"]["port"], *config["control"].values()):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    children = []
    logs = []
    try:
        for index, command in enumerate(commands):
            log = (args.state_dir / f"service-{index}.log").open("ab")
            logs.append(log)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        print("Owner and Control processes started. Logs:", args.state_dir, flush=True)
        print("Use the owner UI to confirm a physical stop before running the recipe.", flush=True)
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("a service exited; inspect logs and physical stop state before restarting")
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 130
    finally:
        # Stop Control services before the physical owner. Its existing cleanup
        # remains responsible for stopping hardware; process exit is not proof.
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("Service did not exit; inspect hardware and process", child.pid, file=sys.stderr)
        for log in logs:
            log.close()


def interrupted(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
