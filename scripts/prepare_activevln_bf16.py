#!/usr/bin/env python3
"""Create a BF16 copy of the FP32 ActiveVLN snapshot, preserving its source.

Requires numpy, ml-dtypes and safetensors on the preparation computer, not CUDA.
The conversion uses round-to-nearest-even BF16, matching the selected BF16
inference precision. This is a precision conversion, not a quantized model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash shards with bounded memory on the supported Python 3.10 runtime."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verified_files(source: Path) -> list[Path]:
    """Require checksums for the config and every indexed model shard."""
    checksums = source / "SHA256SUMS"
    if not checksums.is_file():
        raise ValueError("source download is incomplete: SHA256SUMS is missing")
    expected_hashes = {}
    for line in checksums.read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or name.endswith(".partial")
            or name in {"SHA256SUMS", "download-manifest.json"}
            or name in expected_hashes
        ):
            raise ValueError(f"invalid source checksum entry: {name!r}")
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError(f"missing or unsafe source file: {name}")
        expected_hashes[name] = expected
    if "config.json" not in expected_hashes:
        raise ValueError("source download is incomplete: config.json has no checksum")
    # Check hashes before interpreting the source's config or shard index.
    for name, expected in expected_hashes.items():
        actual = sha256(source / name)
        if actual != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    if "model.safetensors.index.json" in expected_hashes:
        index = json.loads((source / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    available = {name for name in expected_hashes if name.endswith(".safetensors")}
    if not shards or shards != available:
        raise ValueError("source download is incomplete: model shards do not match the checksummed index")
    return [source / name for name in sorted(expected_hashes)]


def convert(source: Path, destination: Path) -> None:
    """Convert verified shards; publish SHA256SUMS only after all output is ready."""
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    if destination.exists():
        raise FileExistsError(f"choose a new destination: {destination}")
    paths = verified_files(source)
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Qwen2_5_VLForConditionalGeneration"]:
        raise ValueError("expected an ActiveVLN Qwen2.5-VL checkpoint")
    destination.mkdir(parents=True)
    total_size = 0
    for path in paths:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".safetensors":
            shutil.copy2(path, target)
            continue
        tensors = {}
        with safe_open(path, framework="numpy") as reader:
            metadata = reader.metadata()
            for key in reader.keys():  # noqa: SIM118 - safetensors reader is not a dict/iterable
                tensor = reader.get_tensor(key)
                if tensor.dtype == np.float32:
                    tensor = tensor.astype(ml_dtypes.bfloat16)
                tensors[key] = tensor
                total_size += tensor.nbytes
        partial = target.with_suffix(".safetensors.partial")
        save_file(tensors, partial, metadata=metadata)
        partial.replace(target)
        del tensors
        print(f"converted {path.name}", flush=True)
    config["torch_dtype"] = "bfloat16"
    if isinstance(config.get("vision_config"), dict):
        config["vision_config"]["torch_dtype"] = "bfloat16"
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index_path = destination / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index.setdefault("metadata", {})["total_size"] = total_size
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    provenance = {"source": str(source.resolve()), "precision": "bfloat16", "rounding": "nearest-even"}
    if (source / "download-manifest.json").exists():
        manifest = json.loads((source / "download-manifest.json").read_text())
        provenance.update(repo=manifest["repo"], revision=manifest["revision"])
    (destination / "conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        value = sha256(path)
        hashes.append(f"{value}  {path.relative_to(destination).as_posix()}\n")
    (destination / "SHA256SUMS").write_text("".join(hashes))


def main() -> None:
    """Prepare an independently checksummed inference copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
