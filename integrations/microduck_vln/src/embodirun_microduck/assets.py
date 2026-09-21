"""Streaming checksums for the existing deployment assets (no weight copies)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_files(roots: dict[str, Path]) -> dict[str, Path]:
    project = roots["project"]
    result = {}
    groups = {
        "project": [
            project / "src/robots_assets",
            project / "src/sim/vln_mujoco",
            project / "src/val_2.json",
            project / "src/rlinf_integration_lightnav_env.py",
            project / "data/train/eval_val2_40_valid.jsonl",
            project / "data/demo_microduck_vln.jsonl",
        ],
        "vvla": [roots["vvla"] / "vvla"],
        "checkpoint": [p for p in roots["checkpoint"].iterdir() if p.is_file()],
    }
    for label, paths in groups.items():
        for path in paths:
            files = path.rglob("*") if path.is_dir() else [path]
            for file in files:
                if not file.is_file() or "__pycache__" in file.parts or file.name.startswith("."):
                    continue
                result[f"{label}/{file.relative_to(roots[label]).as_posix()}"] = file
    required = [
        "checkpoint/config.json",
        "checkpoint/model.safetensors.index.json",
        "project/src/robots_assets/alpha_walking.onnx",
        "project/src/robots_assets/mjcf_assets/robot_allcollisions.xml",
        "project/data/train/eval_val2_40_valid.jsonl",
        "project/data/demo_microduck_vln.jsonl",
    ]
    for name in required:
        if name not in result:
            raise FileNotFoundError(f"Missing required asset: {name}")
    index = json.loads((roots["checkpoint"] / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if f"checkpoint/{name}" not in result:
            raise FileNotFoundError(f"Missing checkpoint shard: {name}")
    return dict(sorted(result.items()))


def write_manifest(path: Path, roots: dict[str, Path]) -> int:
    files = asset_files(roots)
    entries = {key: {"md5": md5(file), "bytes": file.stat().st_size} for key, file in files.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "files": entries}, indent=2) + "\n", encoding="utf-8")
    return len(entries)


def verify_manifest(path: Path, roots: dict[str, Path]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing asset manifest: {path}; use --write-manifest only when provisioning trusted assets"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = asset_files(roots)
    if set(files) != set(manifest["files"]):
        raise RuntimeError("Asset inventory changed; inspect changes before explicitly refreshing the manifest")
    mismatches = []
    for key, file in files.items():
        expected = manifest["files"][key]
        if file.stat().st_size != expected["bytes"] or md5(file) != expected["md5"]:
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(f"Asset MD5 mismatch: {mismatches}")
    return {"files_verified": len(files), "manifest_md5": md5(path), "manifest": str(path)}
