"""Install the same Python profiles for native recipes and container images."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12"
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


def install_commands(kind: str, environment: Path, *, minimal: bool = False) -> list[list[str]]:
    """Use the core lock and the owning integration's dependency declarations."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv 0.12.x, then repeat setup; see docs/en/installation.md")
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
        if kind == "snack":
            command.extend(["--torch-backend", "cpu"])
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
    args = parser.parse_args(argv)
    environment = (args.environment or environment_path(args.profile)).expanduser().resolve()
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
    signal.signal(signal.SIGTERM, _cancel_setup)
    try:
        for command in install_commands(args.profile, environment, minimal=args.minimal):
            _run_install(command, env)
    except (_SetupCancelled, KeyboardInterrupt):
        return 130
    print(f"Environment ready: {environment / 'bin/python'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
