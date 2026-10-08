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
MICRODUCK_PROJECT = ROOT / "examples/microduck_vln"
PROFILES = {
    "host": ".venv",
    "snack": ".venv-xlerobot-snack",
    "microduck": ".venv-microduck",
}
RECIPE_PROJECTS = {"snack": SNACK_PROJECT, "microduck": MICRODUCK_PROJECT}


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
    return ROOT / PROFILES[kind]


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


def recipe_mode(kind: str, mode: str | None = None) -> str:
    """Select the recipe's default profile and explain unsupported modes."""
    modes = {"snack": ("software", "hardware"), "microduck": ("software", "simulation")}
    if kind not in modes:
        if mode not in {None, "hardware"}:
            raise ValueError("--mode is available only for XLeRobot and MicroDuck setup/check")
        return "hardware"
    choices = modes[kind]
    selected = choices[-1] if mode is None else mode
    if selected not in choices:
        name = "MicroDuck" if kind == "microduck" else "XLeRobot"
        raise ValueError(f"{name} setup/check mode must be {' or '.join(choices)} (default: {choices[-1]})")
    return selected


def install_commands(
    kind: str, environment: Path, *, minimal: bool = False, mode: str | None = None
) -> list[list[str]]:
    """Install each profile from its own lock into the caller-selected environment."""
    if kind not in PROFILES:
        raise ValueError(f"Unknown install profile: {kind}")
    mode = recipe_mode(kind, mode)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv 0.12.x, then repeat setup; see docs/en/installation.md")
    if kind in RECIPE_PROJECTS:
        command = [
            uv,
            "sync",
            "--project",
            str(RECIPE_PROJECTS[kind]),
            "--python",
            PYTHON_VERSION,
            "--frozen",
            "--no-default-groups",
        ]
        if mode != "software" and not minimal:
            command.extend(["--extra", mode])
        return [command]
    return [
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--minimal", action="store_true", help="install only the profile's locked base dependencies")
    parser.add_argument(
        "--mode",
        help="XLeRobot: software/hardware (default hardware); MicroDuck: software/simulation (default simulation)",
    )
    args = parser.parse_args(argv)
    if args.profile not in RECIPE_PROJECTS and args.mode is not None:
        parser.error("--mode is available only for XLeRobot and MicroDuck profiles")
    try:
        mode = recipe_mode(args.profile, args.mode)
    except ValueError as error:
        parser.error(str(error))
    environment = (args.environment or environment_path(args.profile)).expanduser().absolute()
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
    signal.signal(signal.SIGTERM, _cancel_setup)
    try:
        print(f"Stage: install {mode if not args.minimal else 'software'} environment", flush=True)
        print(f"Python target: {environment / 'bin/python'}", flush=True)
        print(f"Dependency lock: {RECIPE_PROJECTS.get(args.profile, ROOT) / 'uv.lock'}", flush=True)
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
