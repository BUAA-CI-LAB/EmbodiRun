"""Exercise installed software Recipes through both public entrypoints (Python 3.12)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

PROJECTS = {"xlerobot": "xlerobot_snack_delivery", "microduck": "microduck_vln"}


def check(recipe: str, python: Path, source: Path, work: Path) -> dict:
    cli = [str(python.with_name("embodirun")), "example"]
    env = {
        **os.environ,
        "EMBODIRUN_SOURCE_ROOT": str(source),
        "EMBODIRUN_SCENE_PYTHON": str(python),
        "EMBODIRUN_EXAMPLE_PYTHON": str(python),
    }
    shell = ["bash", str(source / "examples/run.sh")]

    def run(argv: list[str], *, code: int = 0, overrides: dict | None = None, copied_hint: bool = False) -> str:
        inherited = {key: value for key, value in os.environ.items() if not key.startswith("EMBODIRUN_")}
        result = subprocess.run(
            argv,
            cwd=work,
            env={**(inherited if copied_hint else env), **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != code:
            raise RuntimeError(
                f"{shlex.join(argv)} returned {result.returncode}, expected {code}\n{result.stdout}\n{result.stderr}"
            )
        return result.stdout

    directory = work / "recipe with spaces"
    initialized = run([*cli, "init", recipe, "--output", str(directory)])
    manifest = str(directory / "example.local.yaml")
    hint = next(line.removeprefix("Next: ") for line in initialized.splitlines() if line.startswith("Next: "))
    run(shlex.split(hint), copied_hint=True)
    for action in ("validate", "plan"):
        for entry in (cli, shell):
            run([*entry, manifest, action])

    reports = []
    text_outputs = []
    for entry in (cli, shell):
        reports.append(json.loads(run([*entry, manifest, "check", "--mode", "software", "--json"])))
        text_outputs.append(run([*entry, manifest, "check", "--mode", "software"]))
    if reports[0] != reports[1] or text_outputs[0] != text_outputs[1]:
        raise RuntimeError("CLI and shell software checks disagree")
    report = reports[0]
    if report["status"] != "passed" or report["issues"] or report["environment"]["version"][:2] != [3, 12]:
        raise RuntimeError(f"Software prerequisites failed: {report}")
    boundary = "CUDA/EGL" if recipe == "microduck" else "Physical device connection"
    if not any(boundary in item for item in report["unverified"]):
        raise RuntimeError("Software report lost its simulation/hardware boundary")
    check_hint = next(
        line.split(": ", 1)[1]
        for line in initialized.splitlines()
        if line.startswith(("Check software inputs: ", "Check rehearsal inputs: "))
    )
    if json.loads(run(shlex.split(check_hint), copied_hint=True)) != report:
        raise RuntimeError("The printed software check command selects a different environment")
    missing = json.loads(
        run(
            [*cli, manifest, "check", "--mode", "software", "--json"],
            code=2,
            overrides={"EMBODIRUN_SCENE_PYTHON": str(work / "missing/bin/python")},
        )
    )
    if [item["code"] for item in missing["issues"]] != ["python_missing"]:
        raise RuntimeError(f"Missing environment diagnostic is incorrect: {missing}")
    if "setup --mode software" not in missing["issues"][0]["next_action"]:
        raise RuntimeError("Missing environment diagnostic lost its software setup hint")
    default = json.loads(run([*cli, manifest, "check", "--json"], code=2))
    expected_mode, issue = ("simulation", "asset_missing") if recipe == "microduck" else ("hardware", "fixture_route")
    if default["mode"] != expected_mode or not any(item["code"] == issue for item in default["issues"]):
        raise RuntimeError("Default check accepted an unprepared simulation/hardware Recipe")
    if (directory / "runs").exists():
        raise RuntimeError("Configuration/software checks created launcher outputs")

    checks = [
        "init/validate/plan outside the checkout with space-containing paths",
        "matching CLI/shell plain and JSON software checks",
        "copyable printed validate and software-check hints without inherited Recipe variables",
        "missing-environment diagnostic and default simulation/hardware rejection",
        "no launcher output from read-only checks",
    ]
    if recipe == "xlerobot":
        for entry in (cli, shell):
            before = set((directory / "runs").glob("*/run.json"))
            run([*entry, manifest, "dry-run"])
            created = set((directory / "runs").glob("*/run.json")) - before
            if len(created) != 1:
                raise RuntimeError("Fixture rehearsal did not create exactly one run report")
            path = created.pop()
            if json.loads(path.read_text())["status"] != "complete":
                raise RuntimeError("Fixture rehearsal failed")
            status = json.loads((path.parent / "result/status.json").read_text())
            if (
                status["status"] != "completed"
                or status["task_success"] != "unverified"
                or status["physical_success"] is not None
            ):
                raise RuntimeError(f"Fixture rehearsal lost its physical acceptance boundary: {status}")
            if list((directory / "runs").glob(".launcher-*.json")):
                raise RuntimeError("Fixture rehearsal left a launcher pointer")
        checks.append("both fixture rehearsals completed, retained unverified physical results and cleaned launchers")
    if any(path.stat().st_uid != os.getuid() for path in directory.rglob("*")):
        raise RuntimeError("Recipe files are not owned by the caller")
    checks.append("all generated files are owned by the caller")

    inventory = json.loads(
        run(
            [
                str(python),
                "-c",
                "import importlib.metadata as m, json; "
                "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))",
            ]
        )
    )
    packages = {re.sub(r"[-_.]+", "-", name).lower(): version for name, version in inventory.items()}
    lock = tomllib.loads((source / "examples" / PROJECTS[recipe] / "uv.lock").read_text())
    locked = {package["name"]: package["version"] for package in lock["package"]}
    if any(locked.get(name) != version for name, version in packages.items()):
        raise RuntimeError(f"Installed inventory differs from the Recipe lock: {packages}")
    if {"torch", "torchvision", "mujoco", "numpy", "pillow", "transformers", "embodirun-xlerobot-owner"}.intersection(
        packages
    ):
        raise RuntimeError("The software profile installed model/simulation/hardware dependencies")
    checks.append("installed package versions match the Recipe lock; no model/simulation/hardware dependencies")
    return {
        "recipe": recipe,
        "status": "passed",
        "python": report["environment"]["version"],
        "packages": dict(sorted(packages.items())),
        "checks": checks,
        "unverified": report["unverified"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=PROJECTS, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--work-dir", type=Path, help="parent for a temporary caller-owned Recipe directory")
    parser.add_argument("--report", type=Path, help="save the inventory for native/container comparison")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="recipe software ", dir=args.work_dir) as directory:
        report = check(args.recipe, args.python.absolute(), args.source_root.resolve(), Path(directory))
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
