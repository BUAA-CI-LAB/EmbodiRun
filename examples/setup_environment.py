"""Install the same Python profiles for native recipes and container images."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12"
SNACK_PROJECT = ROOT / "examples/xlerobot_snack_delivery"
PROFILES = {
    "host": (".venv", None),
    "snack": (".venv-xlerobot-snack", "xlerobot_owner[hardware,camera,recording,web]"),
    "microduck": (".venv-microduck", "microduck_vln[full]"),
}


class _SetupCancelled(Exception):
    """Stop a setup command after a launcher cancellation signal."""


def _cancel_setup(_signum: int, _frame: object) -> None:
    raise _SetupCancelled


def _run_install(command: list[str], env: dict[str, str]) -> None:
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except (_SetupCancelled, KeyboardInterrupt):
        if os.name == "posix":
            # The session belongs to this uv command, including build helpers.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    time.sleep(1)
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait()
        raise


def environment_path(kind: str) -> Path:
    """Return the default isolated environment for a scene."""
    return ROOT / PROFILES[kind][0]


def environment_for_python(python: str) -> Path:
    """Find a venv target without resolving its Python symlink to a base interpreter."""
    path = Path(python).expanduser()
    if not path.is_absolute():
        path = Path(shutil.which(python) or python).absolute()
    if path.parent.name != "bin" or path.name not in {"python", "python3", "python3.12"}:
        raise ValueError("python must name a virtual environment's bin/python (or python3/python3.12)")
    environment = path.parent.parent
    if environment.exists() and not (environment / "pyvenv.cfg").is_file():
        raise ValueError(f"{environment} is not a virtual environment; choose a new venv directory or correct python")
    return environment


def install_commands(kind: str, environment: Path, *, minimal: bool = False, mode: str = "hardware") -> list[list[str]]:
    """Use the recipe lock for XLeRobot and retain the other integration profiles."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv 0.12.x, then repeat setup; see docs/en/installation.md")
    if mode not in {"software", "hardware"}:
        raise ValueError("setup mode must be software or hardware")
    if kind != "snack" and mode != "hardware":
        raise ValueError("--mode software is available only for the XLeRobot recipe")
    if kind == "snack":
        command = [
            uv,
            "sync",
            "--project",
            str(SNACK_PROJECT),
            "--python",
            PYTHON_VERSION,
            "--frozen",
            "--no-default-groups",
        ]
        if mode == "hardware" and not minimal:
            command.extend(["--extra", "hardware"])
        return [command]
    commands = [
        [
            uv,
            "sync",
            "--project",
            str(ROOT),
            "--python",
            PYTHON_VERSION,
            "--frozen",
            "--no-default-groups",
            "--group",
            "host",
        ]
    ]
    integration = PROFILES[kind][1]
    if integration and not minimal:
        command = [uv, "pip", "install"]
        command.extend(
            [
                "--python",
                str(environment / "bin/python"),
                "-e",
                str(ROOT / "integrations" / integration),
            ]
        )
        commands.append(command)
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--minimal", action="store_true", help="install only the locked Host dependencies")
    parser.add_argument("--mode", choices=("software", "hardware"), help="XLeRobot install mode (default: hardware)")
    args = parser.parse_args(argv)
    if args.profile != "snack" and args.mode is not None:
        parser.error("--mode is available only for the XLeRobot snack profile")
    mode = args.mode or "hardware"
    environment = (args.environment or environment_path(args.profile)).expanduser().absolute()
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
    signal.signal(signal.SIGTERM, _cancel_setup)
    try:
        print(f"Stage: install {mode if not args.minimal else 'software'} environment", flush=True)
        print(f"Python target: {environment / 'bin/python'}", flush=True)
        if args.profile == "snack":
            print(f"Dependency lock: {SNACK_PROJECT / 'uv.lock'}", flush=True)
        for command in install_commands(args.profile, environment, minimal=args.minimal, mode=mode):
            _run_install(command, env)
    except (_SetupCancelled, KeyboardInterrupt):
        print("Setup interrupted. Repeat setup with the same manifest to finish installation.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}. Correct the reported install error, then repeat setup.", file=sys.stderr)
        return 2
    print(f"Environment ready: {environment / 'bin/python'}")
    if args.profile == "snack" and mode == "hardware" and not args.minimal:
        print(f"SDK directory for hardware.local.json sdk_src: {environment / 'lib/python3.12/site-packages'}")
        print("Use this installed LeRobot source; calibration and device paths still need operator input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
