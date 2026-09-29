"""Prepare linked local recipe files and report missing deployment inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from embodirun.deployment.config.loader import _load_yaml

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
    path = initialize(args.template, args.output or ROOT / "examples/local" / args.template, assets=args.assets)
    print(f"Created {path}\nNext: embodirun example {path} check")
    if args.template == "xlerobot":
        print(f"Software rehearsal: embodirun example {path} dry-run")
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


def check_requirements(example: Any) -> list[str]:
    """Check files and declarations locally; never open a camera or connect to a robot."""
    p = example.parameters
    problems: list[str] = []
    if example.kind == "rollout":
        path = example.resolve(p["deployment"])
        problems.extend(placeholders(_load_yaml(path), str(path)))
        print("Deployment structure checked. Device/calibration/checkpoint paths belong to their configured nodes.")
        print(
            "Still verify SSH host keys, calibration, camera-to-binding mappings and checkpoint features on those nodes."
        )
    elif example.kind == "snack":
        deployment_path = example.resolve(p["deployment"])
        deployment = _load_yaml(deployment_path)
        problems.extend(placeholders(deployment, str(deployment_path)))
        hardware_path = (deployment_path.parent / deployment["owner"]["hardware_config"]).resolve()
        if not hardware_path.is_file():
            return [*problems, f"{hardware_path}: owner hardware configuration is missing"]
        hardware = json.loads(hardware_path.read_text())
        problems.extend(placeholders(hardware, str(hardware_path)))
        for field in ("sdk_src", "calibration"):
            raw = hardware.get(field)
            if raw is None:
                problems.append(f"{hardware_path}: {field} is missing")
            elif isinstance(raw, str) and not placeholders(raw) and not Path(raw).expanduser().exists():
                problems.append(f"{hardware_path}: {field} does not exist: {raw}")
        if not hardware.get("camera_roles_confirmed"):
            problems.append(
                f"{hardware_path}: camera_roles_confirmed is false; verify front/left_wrist/right_wrist mappings"
            )
        cameras = hardware.get("cameras", {})
        for feature, role in deployment.get("cameras", {}).items():
            if role not in cameras:
                problems.append(f"{deployment_path}: cameras.{feature} refers to missing hardware role {role}")
            else:
                source = cameras[role]
                if source is None:
                    problems.append(f"{hardware_path}: camera role {role} has no source")
                elif (
                    isinstance(source, str)
                    and "://" not in source
                    and not placeholders(source)
                    and not Path(source).expanduser().exists()
                ):
                    problems.append(f"{hardware_path}: camera role {role} does not exist: {source}")
        task_path = example.resolve(p["config"])
        task = json.loads(task_path.read_text())
        problems.extend(placeholders(task, str(task_path)))
        for name, value in task["navigation"]["routes"].items():
            if not isinstance(value, str):
                if isinstance(value, dict) and value.get("fixture") is True:
                    problems.append(f"{task_path}: navigation.routes.{name} is a software fixture")
                continue
            route_path = (task_path.parent / value).resolve()
            if not route_path.is_file():
                problems.append(f"{task_path}: navigation.routes.{name} does not exist: {route_path}")
            else:
                route = json.loads(route_path.read_text())
                if route.get("fixture") is True:
                    problems.append(
                        f"{route_path}: software fixture; record and verify a room route before hardware use"
                    )
        print("Hardware files checked without starting the owner. Motion settings and confirmations are unchanged.")
        print(
            "Still verify physical stop feedback, handover pose and a two-arm checkpoint matching lerobot.xlerobot.pi05."
        )
    else:
        for key in ("project_root", "inference_root", "checkpoint", "episodes", "manifest"):
            path = example.resolve(p[key])
            if not path.exists():
                problems.append(f"{example.path}: parameters.{key} does not exist: {path}")
    if not Path(example.python).is_file() or shutil.which(example.python) is None:
        problems.append(f"Python is unavailable: {example.python}; run setup or correct the manifest's python field")
    return problems
