"""Prepare linked local recipe files and report missing deployment inputs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from embodirun.deployment.config.loader import _load_yaml
from examples.setup_environment import recipe_mode

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "so101": "multi_robot_serving/example.yaml",
    "embodiinfer-http": "engine_comparison/embodiinfer-http.yaml",
    "embodiinfer-wireless": "engine_comparison/embodiinfer-wireless.yaml",
    "sglang-http": "engine_comparison/sglang-http.yaml",
    "xlerobot": "xlerobot_snack_delivery/example.yaml",
    "microduck": "microduck_vln/example.yaml",
}


def initialize(template: str, destination: Path, *, assets: Path | None = None) -> Path:
    """Copy a complete editable recipe without overwriting an existing directory."""
    source = ROOT / "examples" / TEMPLATES[template]
    data = _load_yaml(source)
    p = data["parameters"]
    files: dict[Path, bytes] = {}
    data.pop("python", None)
    data["output_dir"] = "runs"
    if data["kind"] == "rollout":
        files[Path("deployment.local.yaml")] = (source.parent / p["deployment"]).read_bytes()
        p["deployment"] = "deployment.local.yaml"
    elif data["kind"] == "snack":
        task = json.loads((source.parent / p["config"]).read_text())
        task["control"].pop("endpoints", None)
        for name, route in task["navigation"]["routes"].items():
            target = Path("routes") / f"{name}.local.json"
            files[target] = (source.parent / route).read_bytes()
            task["navigation"]["routes"][name] = target.as_posix()
        deployment = _load_yaml(source.parent / p["deployment"])
        hardware = source.parent / deployment["owner"]["hardware_config"]
        files[Path("hardware.local.json")] = hardware.read_bytes()
        deployment["owner"]["hardware_config"] = "hardware.local.json"
        files[Path("deployment.local.yaml")] = yaml.safe_dump(deployment, sort_keys=False).encode()
        files[Path("config.local.json")] = (json.dumps(task, indent=2) + "\n").encode()
        p.update(config="config.local.json", deployment="deployment.local.yaml")
    else:
        files[Path("assets.local.json")] = (source.parent / p["manifest"]).read_bytes()
        p["manifest"] = "assets.local.json"
        p["inference_root"] = str(ROOT / "third_party/embodiinfer")
        if assets is not None:
            assets = assets.expanduser().resolve()
            p.update(
                project_root=str(assets),
                checkpoint=str(assets / "lightnav_sft_v3/merged"),
                episodes=str(assets / "data/demo_microduck_vln.jsonl"),
            )
    if assets is not None and data["kind"] != "microduck":
        raise ValueError("--assets is only used by the microduck template")
    files[Path("example.local.yaml")] = yaml.safe_dump(data, sort_keys=False).encode()
    destination = destination.expanduser().resolve()
    # Exclusive directory creation protects an existing calibration or edited task.
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return destination / "example.local.yaml"


def init_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Create a complete local recipe with its file references already connected."
    )
    parser.add_argument("template", choices=TEMPLATES)
    parser.add_argument("--output", type=Path, help="new directory (default: examples/local/TEMPLATE)")
    parser.add_argument("--assets", type=Path, help="MicroDuck asset root; derives checkpoint and episode paths")
    args = parser.parse_args(argv)
    destination = (args.output or ROOT / "examples/local" / args.template).expanduser().resolve()
    try:
        path = initialize(args.template, destination, assets=args.assets)
    except FileExistsError:
        print(f"Recipe directory already exists: {destination}. Existing files were preserved.", file=sys.stderr)
        existing = destination / "example.local.yaml"
        if existing.is_file():
            print(f"Next: {_recipe_command(existing, 'validate')}", file=sys.stderr)
        else:
            print("Next: choose a new directory with init --output DIR.", file=sys.stderr)
        return 2
    print(f"Stage: recipe files created\nManifest: {path}\nNext: {_recipe_command(path, 'validate')}")
    if args.template == "xlerobot":
        print(f"Install rehearsal environment: {_recipe_command(path, 'setup', '--mode', 'software')}")
        print(f"Check rehearsal inputs: {_recipe_command(path, 'check', '--mode', 'software', '--json')}")
        print(f"Software rehearsal: {_recipe_command(path, 'dry-run')}")
    elif args.template == "microduck":
        print(f"Install software environment: {_recipe_command(path, 'setup', '--mode', 'software')}")
        print(f"Check software inputs: {_recipe_command(path, 'check', '--mode', 'software', '--json')}")
        print("Software checks do not require scene assets or a GPU; simulation setup/check is separate.")
    else:
        print(f"Next after validation: {_recipe_command(path, 'check')}")
    return 0


def placeholders(value: Any, location: str = "") -> list[str]:
    """Find unresolved template values, preserving the field names in diagnostics."""
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in placeholders(child, f"{location}.{key}".lstrip("."))]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in placeholders(child, f"{location}[{index}]")]
    if isinstance(value, str) and ("REPLACE_" in value or value.startswith(("/path/to/", "/absolute/path/to/"))):
        return [f"{location}: replace {value}"]
    return []


def _recipe_command(path: Path, *args: str) -> str:
    entrypoint = Path(sys.executable).absolute().with_name("embodirun")
    environment = [f"EMBODIRUN_SOURCE_ROOT={ROOT}"]
    if scene_python := os.environ.get("EMBODIRUN_SCENE_PYTHON"):
        environment.append(f"EMBODIRUN_SCENE_PYTHON={scene_python}")
    command = (
        [str(entrypoint)] if entrypoint.is_file() else ["uv", "run", "--frozen", "--project", str(ROOT), "embodirun"]
    )
    return shlex.join(["env", *environment, *command, "example", str(path), *args])


def _environment_details(python: str, packages: list[str]) -> dict[str, Any]:
    """Read metadata in the selected interpreter; do not import hardware or ML code."""
    script = """
import importlib.metadata, json, pathlib, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
try:
    sdk_src = str(pathlib.Path(importlib.metadata.distribution('lerobot').locate_file('lerobot')).parent)
except importlib.metadata.PackageNotFoundError:
    sdk_src = None
print(json.dumps({'version': list(sys.version_info[:3]), 'packages': versions, 'sdk_src': sdk_src}))
"""
    result = subprocess.run(
        [python, "-c", script, json.dumps(packages)], capture_output=True, text=True, check=True, timeout=15
    )
    return json.loads(result.stdout)


def check_report(example: Any, *, mode: str | None = None) -> dict[str, Any]:
    """Check installed metadata and local inputs without connecting to services/devices."""
    mode = recipe_mode(example.kind, mode)
    p = example.parameters
    issues: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "recipe": str(example.path),
        "kind": example.kind,
        "mode": mode,
        "status": "passed",
        "issues": issues,
        "verified": ["Manifest and task configuration structure"],
        "unverified": ["Live model/service availability"],
        "next_actions": [],
    }
    if example.kind != "microduck":
        report["unverified"].append("Physical device connection, calibration and stop feedback")

    def issue(code: str, location: str, message: str, action: str) -> None:
        issues.append({"code": code, "location": location, "message": message, "next_action": action})

    def unresolved(value: Any, location: str) -> None:
        for problem in placeholders(value, location):
            field, _, value_text = problem.partition(": replace ")
            issue("placeholder", field, f"replace {value_text}", f"Edit {field} with the value for this deployment.")

    setup = (
        _recipe_command(example.path, "setup", "--mode", mode)
        if example.kind == "snack" or (example.kind == "microduck" and mode == "software")
        else _recipe_command(example.path, "setup")
    )
    report["environment"] = {"python": example.python}
    if not Path(example.python).is_file() or shutil.which(example.python) is None:
        issue("python_missing", "python", f"Python is unavailable: {example.python}", setup)
    elif example.kind in {"snack", "microduck"}:
        try:
            report["environment"]["path"] = str(example.environment)
        except ValueError as error:
            issue(
                "python_environment",
                "python",
                str(error),
                f"Correct python in {example.path} or EMBODIRUN_SCENE_PYTHON to name a venv/bin/python, then repeat setup.",
            )
        packages = ["embodirun", "PyYAML", "paramiko", "rich"]
        if example.kind == "microduck":
            packages.append("embodirun-microduck")
            if mode == "simulation":
                packages += [
                    "numpy",
                    "Pillow",
                    "mujoco",
                    "onnxruntime",
                    "casadi",
                    "imageio-ffmpeg",
                    "torch",
                    "torchvision",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                    "safetensors",
                    "einops",
                    "accelerate",
                ]
        elif mode == "hardware":
            packages += [
                "embodirun-xlerobot-owner",
                "aiohttp",
                "cryptography",
                "lerobot",
                "opencv-python-headless",
                "numpy",
                "Pillow",
                "torch",
                "torchvision",
                "datasets",
                "pandas",
                "pyarrow",
                "av",
                "jsonlines",
                "feetech-servo-sdk",
                "pyserial",
                "deepdiff",
            ]
        try:
            details = _environment_details(example.python, packages)
            report["environment"].update(details)
            if details["version"][:2] != [3, 12]:
                name = "MicroDuck" if example.kind == "microduck" else "XLeRobot"
                issue("python_version", "python", f"The {name} recipe uses Python 3.12.", setup)
            missing = [name for name, version in details["packages"].items() if version is None]
            if missing:
                issue("dependencies_missing", "python", f"Missing installed packages: {', '.join(missing)}", setup)
            else:
                report["verified"].append(f"Installed {mode} package metadata in the selected Python")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            issue("environment_inspection", "python", f"Cannot inspect this Python: {error}", setup)
        report["unverified"].append("Optional dependency imports and external SDK compatibility")
    if example.kind == "rollout":
        path = example.resolve(p["deployment"])
        unresolved(_load_yaml(path), str(path))
        report["unverified"].append("SSH host keys, device paths and checkpoint features on the configured nodes")
    elif example.kind == "snack":
        from examples.xlerobot_snack_delivery.run import load_recipe_config

        deployment_path = example.resolve(p["deployment"])
        deployment = _load_yaml(deployment_path)
        task_path = example.resolve(p["config"])
        task = load_recipe_config(task_path, deployment_path)
        report["control_endpoints"] = task["control"]["endpoints"]
        report["verified"].append("Scoped local Control addresses derived from the deployment")
        task_values = dict(task)
        task_values["planning"] = dict(task.get("planning", {}))
        if task_values["planning"].get("mode", "recorded_route") != "diagram":
            task_values["planning"].pop("route_diagram", None)
        unresolved(task_values, str(task_path))
        for name, value in task["navigation"]["routes"].items():
            location = f"{task_path}.navigation.routes.{name}"
            action = f"Record and verify a room route, then edit {location} to point to that JSON."
            if not isinstance(value, str):
                if mode == "hardware" and isinstance(value, dict) and value.get("fixture") is True:
                    issue("fixture_route", location, "route is a software fixture", action)
                continue
            route_path = (task_path.parent / value).resolve()
            if not route_path.is_file():
                issue("route_missing", location, f"route does not exist: {route_path}", f"Correct {location}.")
            else:
                route = json.loads(route_path.read_text())
                if not isinstance(route, dict):
                    issue("route_invalid", str(route_path), "route JSON must be an object", f"Correct {route_path}.")
                elif mode == "hardware" and route.get("fixture") is True:
                    issue("fixture_route", str(route_path), "software fixture; not a hardware route", action)
        if not any(item["code"] in {"route_missing", "route_invalid"} for item in issues):
            report["verified"].append("Local route file availability and JSON syntax")
        report["unverified"] += [
            "Recorded route execution, handover pose and task outcome",
            "Two-arm checkpoint feature names, units and camera inputs for lerobot.xlerobot.pi05",
        ]
        if mode == "software":
            report["unverified"].append("Owner SDK, serial devices and cameras (unused by software rehearsal)")
        else:
            unresolved(deployment, str(deployment_path))
            hardware_path = (deployment_path.parent / deployment["owner"]["hardware_config"]).resolve()
            if not hardware_path.is_file():
                issue(
                    "hardware_config_missing",
                    str(hardware_path),
                    "owner hardware configuration is missing",
                    f"Correct {deployment_path}.owner.hardware_config to point to your hardware JSON.",
                )
            else:
                hardware = json.loads(hardware_path.read_text())
                if not isinstance(hardware, dict):
                    raise ValueError(f"{hardware_path} must contain a JSON object")
                for field in ("ports", "cameras"):
                    if not isinstance(hardware.get(field, {}), dict):
                        raise ValueError(f"{hardware_path}.{field} must be an object")
                unresolved(
                    {k: v for k, v in hardware.items() if k not in {"sdk_src", "calibration"}}, str(hardware_path)
                )
                sdk = hardware.get("sdk_src")
                recommended = report["environment"].get("sdk_src")
                sdk_action = (
                    f"Set {hardware_path}.sdk_src to {recommended} to use this environment's locked LeRobot."
                    if recommended
                    else f"Run {setup}, then set {hardware_path}.sdk_src to its reported SDK directory."
                )
                if not isinstance(sdk, str) or placeholders(sdk) or not Path(sdk).expanduser().is_dir():
                    issue(
                        "sdk_missing", f"{hardware_path}.sdk_src", f"SDK source directory is missing: {sdk}", sdk_action
                    )
                elif not any(
                    (Path(sdk).expanduser() / item).is_file()
                    for item in ("lerobot/motors/feetech/__init__.py", "lerobot/motors/feetech.py")
                ):
                    issue(
                        "sdk_layout",
                        f"{hardware_path}.sdk_src",
                        "SDK path must contain lerobot/motors/feetech",
                        sdk_action,
                    )
                elif not recommended or Path(sdk).expanduser().resolve() != Path(recommended).resolve():
                    report["unverified"].append(
                        "The explicit external sdk_src version is not guaranteed by the recipe lock"
                    )
                calibration = hardware.get("calibration")
                if calibration is None or (
                    isinstance(calibration, str)
                    and (placeholders(calibration) or not Path(calibration).expanduser().is_file())
                ):
                    issue(
                        "calibration_missing",
                        f"{hardware_path}.calibration",
                        f"calibration is missing: {calibration}",
                        f"Supply verified motor calibration in {hardware_path}.calibration before hardware use.",
                    )
                if not hardware.get("camera_roles_confirmed"):
                    issue(
                        "camera_roles_unconfirmed",
                        f"{hardware_path}.camera_roles_confirmed",
                        "camera_roles_confirmed is false; verify front/left_wrist/right_wrist mappings",
                        f"Verify each physical camera role, then update {hardware_path}.camera_roles_confirmed.",
                    )
                for role, source in hardware.get("ports", {}).items():
                    if isinstance(source, str) and not placeholders(source) and not Path(source).expanduser().exists():
                        issue(
                            "serial_path_missing",
                            f"{hardware_path}.ports.{role}",
                            f"serial path does not exist: {source}",
                            f"Inspect device enumeration and correct {hardware_path}.ports.{role}.",
                        )
                cameras = hardware.get("cameras", {})
                for feature, role in deployment.get("cameras", {}).items():
                    if role not in cameras:
                        issue(
                            "camera_role_missing",
                            f"{deployment_path}.cameras.{feature}",
                            f"missing hardware role {role}",
                            f"Connect {deployment_path}.cameras.{feature} to a role in {hardware_path}.cameras.",
                        )
                    elif cameras[role] is None:
                        issue(
                            "camera_source_missing",
                            f"{hardware_path}.cameras.{role}",
                            f"camera role {role} has no source",
                            f"Set {hardware_path}.cameras.{role} to the verified device index, path or stream.",
                        )
                    elif (
                        isinstance(cameras[role], str)
                        and "://" not in cameras[role]
                        and not placeholders(cameras[role])
                        and not Path(cameras[role]).expanduser().exists()
                    ):
                        issue(
                            "camera_path_missing",
                            f"{hardware_path}.cameras.{role}",
                            f"camera role {role} does not exist: {cameras[role]}",
                            f"Inspect camera enumeration and correct {hardware_path}.cameras.{role}.",
                        )
    else:
        report["unverified"] += [
            "CUDA/EGL availability and MicroDuck scene/model preflight",
            "External scene/checkpoint/episode assets and inference source compatibility",
            "Learned-model inference and navigation success",
        ]
        if mode == "simulation":
            for key in ("project_root", "inference_root", "checkpoint", "episodes", "manifest"):
                path = example.resolve(p[key])
                if not path.exists():
                    issue(
                        "asset_missing",
                        f"{example.path}.parameters.{key}",
                        f"asset does not exist: {path}",
                        f"Supply this asset and correct {example.path}.parameters.{key}.",
                    )
            inference_root = example.resolve(p["inference_root"])
            if inference_root.exists() and not (inference_root / "embodiinfer/__init__.py").is_file():
                initialize_source = shlex.join(
                    ["git", "-C", str(ROOT), "submodule", "update", "--init", "third_party/embodiinfer"]
                )
                issue(
                    "inference_source_missing",
                    f"{example.path}.parameters.inference_root",
                    f"EmbodiInfer entrypoint is missing: {inference_root / 'embodiinfer/__init__.py'}",
                    f"Initialize the pinned source with {initialize_source}, or select a compatible inference_root.",
                )
    report["status"] = "needs_attention" if issues else "passed"
    report["next_actions"] = list(dict.fromkeys(item["next_action"] for item in issues))
    if not issues:
        if example.kind == "microduck" and mode == "software":
            report["next_actions"] = [
                "For a simulation run, supply external assets and the pinned inference source, then use "
                + _recipe_command(example.path, "setup", "--mode", "simulation")
            ]
        else:
            report["next_actions"] = [
                _recipe_command(example.path, "dry-run")
                if example.kind == "snack" and mode == "software"
                else _recipe_command(example.path, "check", "--mode", "simulation")
                if example.kind == "microduck"
                else "Review the unverified device/model prerequisites with the operator before hardware startup."
            ]
    return report


def check_requirements(example: Any) -> list[str]:
    """Compatibility view of the local hardware prerequisites."""
    return [f"{item['location']}: {item['message']}" for item in check_report(example)["issues"]]
