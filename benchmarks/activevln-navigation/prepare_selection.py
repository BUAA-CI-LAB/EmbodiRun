"""Select reproducible scene-proportional validation subsets before model inference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def stratified_sample(
    episodes: list[dict[str, Any]], count: int, seed: int, dataset: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Give every scene a slot, then fill the largest proportional quota deficits."""
    if not 1 <= count <= len(episodes):
        raise ValueError("sample count must be positive and no greater than the eligible population")
    if len({str(ep["episode_id"]) for ep in episodes}) != len(episodes):
        raise ValueError("duplicate episode IDs in population")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[episode["scene_id"]].append(episode)
    total = len(episodes)
    quotas = {scene: max(1, len(rows) * count // total) for scene, rows in groups.items()}
    if sum(quotas.values()) > count:
        raise ValueError("sample count cannot accommodate minimum-one proportional scene quotas")
    while sum(quotas.values()) < count:
        scene = min(groups, key=lambda key: (-(len(groups[key]) * count - quotas[key] * total), key))
        quotas[scene] += 1
    selected = []
    for scene in sorted(groups):
        rows = sorted(groups[scene], key=lambda ep: (int(ep["episode_id"]), str(ep["episode_id"])))
        scene_seed = int.from_bytes(hashlib.sha256(f"{seed}:{dataset}:{scene}".encode()).digest(), "big")
        picked = random.Random(scene_seed).sample(rows, quotas[scene])
        selected.extend(sorted(picked, key=lambda ep: int(ep["episode_id"])))
    return selected, quotas


def prepare(
    source: Path, scenes: Path, dataset: str, count: int, seed: int, preflight: Path | None = None
) -> dict[str, Any]:
    """Validate unchanged navmeshes and record every exclusion before selecting episodes."""
    raw = source.read_bytes()
    data = json.loads(gzip.decompress(raw) if source.suffix == ".gz" else raw)
    population = data["episodes"]
    if dataset == "rxr":
        population = [ep for ep in population if ep["instruction"]["language"].startswith("en")]
    if preflight is not None:
        previous = json.loads(preflight.read_text())
        if previous["dataset"] != dataset or previous["split"] != "val_unseen":
            raise ValueError("preflight dataset or split differs")
        if hashlib.sha256(raw).hexdigest() != previous["source_sha256"]:
            raise ValueError("source differs from the reachability preflight")
        for scene, expected in previous["navmesh_sha256"].items():
            geometry = (scenes / scene).resolve()
            if not geometry.is_relative_to(scenes.resolve()):
                raise ValueError(f"unsafe scene path: {scene}")
            if hashlib.sha256(geometry.with_suffix(".navmesh").read_bytes()).hexdigest() != expected:
                raise ValueError(f"navmesh differs from preflight: {scene}")
        excluded_ids = {str(row["episode_id"]) for row in previous["excluded_before_model_inference"]}
        eligible = [ep for ep in population if str(ep["episode_id"]) not in excluded_ids]
        if len(population) != previous["candidate_episodes"] or len(eligible) != previous["eligible_episodes"]:
            raise ValueError("population differs from reachability preflight")
        if any(ep["scene_id"] not in previous["navmesh_sha256"] for ep in population):
            raise ValueError("population contains a scene without a checked navmesh")
        metadata = {k: v for k, v in previous.items() if k not in ("episodes", "scene_quotas")}
        return selected_manifest(metadata, eligible, count, seed, dataset, preflight)

    import habitat_sim
    import numpy as np

    eligible, excluded = [], []
    pathfinders, mesh_hashes = {}, {}
    for episode in population:
        scene = episode["scene_id"]
        geometry = (scenes / scene).resolve()
        if not geometry.is_relative_to(scenes.resolve()):
            raise ValueError(f"unsafe scene path: {scene}")
        if scene not in pathfinders:
            navmesh = geometry.with_suffix(".navmesh")
            if not geometry.is_file() or not navmesh.is_file():
                raise FileNotFoundError(f"missing scene or navmesh: {geometry}")
            finder = habitat_sim.PathFinder()
            if not finder.load_nav_mesh(str(navmesh)):
                raise ValueError(f"cannot load navmesh: {navmesh}")
            pathfinders[scene] = finder
            mesh_hashes[scene] = hashlib.sha256(navmesh.read_bytes()).hexdigest()
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(episode["start_position"], dtype=np.float32)
        path.requested_end = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
        if not pathfinders[scene].find_path(path) or not math.isfinite(path.geodesic_distance):
            excluded.append({"episode_id": str(episode["episode_id"]), "reason": "no finite start-to-goal path"})
        else:
            eligible.append(episode)
    metadata = {
        "dataset": dataset,
        "split": "val_unseen",
        "language": "English",
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "selection": "",
        "sampling_method_version": "",
        "seed": seed,
        "expected_episodes": count,
        "candidate_episodes": len(population),
        "eligible_episodes": len(eligible),
        "excluded_before_model_inference": excluded,
        "navmesh_sha256": mesh_hashes,
        "habitat_sim_version": habitat_sim.__version__,
        "omitted_unused_fields": ["instruction.timed_instruction"],
    }
    return selected_manifest(metadata, eligible, count, seed, dataset)


def selected_manifest(
    metadata: dict[str, Any],
    eligible: list[dict[str, Any]],
    count: int,
    seed: int,
    dataset: str,
    preflight: Path | None = None,
) -> dict[str, Any]:
    """Freeze the measured minimum-one scene selection and its reachability provenance."""
    selected, quotas = stratified_sample(eligible, count, seed, dataset)
    metadata.update(
        expected_episodes=count,
        seed=seed,
        selection="proportional scene quotas with minimum one per scene; largest remaining deficit; seeded sampling without replacement; scene then episode-ID order",
        sampling_method_version="sha256-seed-dataset-scene-python-random-min1-v1",
        scene_quotas=quotas,
        eligible_scene_counts=dict(Counter(ep["scene_id"] for ep in eligible)),
    )
    if preflight is not None:
        metadata["reachability_preflight_manifest_sha256"] = hashlib.sha256(preflight.read_bytes()).hexdigest()
    metadata["episodes"] = [
        dict(ep, instruction={k: v for k, v in ep["instruction"].items() if k != "timed_instruction"})
        for ep in selected
    ]
    return metadata


def main() -> None:
    """Write immutable R2R/RxR manifests in a fresh benchmark artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2r-source", type=Path)
    parser.add_argument("--rxr-source", type=Path)
    parser.add_argument("--datasets", choices=("r2r", "rxr"), nargs="+", default=["r2r", "rxr"])
    parser.add_argument(
        "--preflight-root", type=Path, help="Reuse checked 1024-episode manifests after hash validation"
    )
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sources = {"r2r": args.r2r_source, "rxr": args.rxr_source}
    for dataset in args.datasets:
        if sources[dataset] is None:
            parser.error(f"--{dataset}-source is required for this selection")
    args.output.mkdir(parents=True, exist_ok=False)
    for dataset in args.datasets:
        preflight = args.preflight_root / f"selected-{dataset}-navigation-1024.json" if args.preflight_root else None
        selection = prepare(sources[dataset], args.scenes, dataset, args.episode_count, args.seed, preflight)
        path = args.output / f"selected-{dataset}-navigation-{args.episode_count}.json"
        path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in selection.items()
                    if key not in ("episodes", "excluded_before_model_inference", "navmesh_sha256")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
