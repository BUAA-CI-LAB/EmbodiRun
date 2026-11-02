"""Audit completed NX navigation traces and summarize paired quality and model timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from metrics import compare, primitive_actions


def read(path: Path) -> Any:
    """Read one retained JSON evidence artifact."""
    return json.loads(path.read_text())


def timing_summary(values: list[float]) -> dict[str, float]:
    """Use linear percentiles over measured calls, without including warmup."""
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError("timings must be nonempty, finite and nonnegative")

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {"mean": statistics.mean(ordered), "p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


def summarize_stage(root: Path, stage: str, *, split: str | None = None) -> dict[str, Any]:
    """Require all declared episodes and verify actions, distances, timings and model-call evidence."""
    folder = root / stage
    if split is None:
        available = [name for name in ("r2r", "rxr") if (folder / f"{name}.json").exists()]
        if len(available) != 1:
            raise ValueError("stage requires exactly one dataset report or an explicit split")
        split = available[0]
    report = read(folder / f"{split}.json")
    count = report.get("expected_episodes", 48)
    if report["status"] != "complete" or len(report["episodes"]) != count:
        raise ValueError(f"{stage} is incomplete; expected {count} episodes")
    if len({str(ep["episode_id"]) for ep in report["episodes"]}) != count:
        raise ValueError("duplicate completed episode IDs")
    details, http = [], []
    invocations: dict[tuple[int, int], list[tuple[dict, dict, int]]] = {}
    grammar = report["model_service"]["config"].get("action_space", "r2r")
    for summary in report["episodes"]:
        episode = read(folder / split / f"episode-{summary['episode_id']}.json")
        for key in (
            "episode_id",
            "status",
            "success",
            "spl",
            "steps",
            "stopped",
            "path_length_m",
            "distance_to_goal_m",
        ):
            if summary[key] != episode[key]:
                raise ValueError(f"summary/trace mismatch: {key}")
        position, steps, length, stopped = episode["start_position"], 0, 0.0, False
        distances = [episode["shortest_path_m"]]
        previous_cache = 0
        for trace in episode["trace"]:
            if trace["before_position"] != position:
                raise ValueError("discontinuous execution trace")
            batch = trace["model"]["tensor_batch"]
            service_index = episode.get("model_service_index", summary.get("model_service_index", 0))
            invocations.setdefault((service_index, batch["index"]), []).append((episode, trace, previous_cache))
            previous_cache = trace["model"]["cache_length"]
            http.append(trace["http_roundtrip_ms"])
            if "invalid_action" in trace:
                if episode["termination"] != "invalid_action":
                    raise ValueError("invalid action was not reported as such")
                continue
            expected = primitive_actions(trace["model"]["rows"], action_space=grammar)[: 500 - steps]
            if [row["action"] for row in trace["executed"]] != expected:
                raise ValueError("executed primitives differ from model actions")
            for row in trace["executed"]:
                if stopped:
                    raise ValueError("action executed after STOP")
                length += math.dist(position, row["position"])
                position, stopped = row["position"], row["action"] == 0
                steps += 1
                if row["distance_to_goal_m"] is not None:
                    distances.append(row["distance_to_goal_m"])
        if steps != episode["steps"] or stopped != episode["stopped"] or abs(length - episode["path_length_m"]) > 1e-4:
            raise ValueError("primitive trajectory does not reproduce episode metrics")
        episode["oracle_success"] = min(distances) < 3.0
        details.append(episode)
    # Also recompute STOP/SPL evidence with the independent metric validator.
    compare(report, report)
    evidence = [json.loads(line) for line in (folder / "tensor-batches.jsonl").read_text().splitlines()]
    indexed = {(row.get("model_service_index", 0), row["index"]): row for row in evidence}
    if len(indexed) != len(evidence) or indexed.keys() != invocations.keys():
        raise ValueError("server calls differ from retained tensor invocations")
    timings, forward = [], []
    for identity in sorted(invocations):
        rows, call = invocations[identity], indexed[identity]
        size = call["observations"]
        if call["status"] != "complete" or size != len(rows) or size != len(call["rows"]):
            raise ValueError("server tensor batch occupancy differs from retained trajectories")
        sessions = {row["session_id"]: row for row in call["rows"]}
        if len(sessions) != size or len({row["slot"] for row in call["rows"]}) != size:
            raise ValueError("server tensor batch contains duplicate sessions or slots")
        for episode, trace, previous_cache in rows:
            batch = trace["model"]["tensor_batch"]
            if batch["observations"] != size or not 1 <= size <= batch["capacity"]:
                raise ValueError("response tensor batch metadata disagrees with server evidence")
            if trace["timing"]["e2e_ms"] != call["latency_ms"] or any(
                trace["timing"][key] != value for key, value in call["model_timing_ms"].items()
            ):
                raise ValueError("response model timings disagree with server evidence")
            session = episode.get("session_id")
            if session is None and size != 1:
                raise ValueError("batched trajectory is missing its session identity")
            server_row = call["rows"][0] if session is None else sessions.get(session)
            if (
                server_row is None
                or server_row["step_id"] != trace["turn"]
                or server_row["cache_length"] != previous_cache
            ):
                raise ValueError("model session history differs from its episode trace")
            if "benchmark_slot" in episode and episode["benchmark_slot"] != server_row["slot"]:
                raise ValueError("model slot differs from its episode trace")
        timings.append(call["latency_ms"])
        forward.append(call["model_timing_ms"]["pure_inference_ms"])
    starts = [row for call in evidence for row in call["rows"] if row["step_id"] == 0]
    if len(starts) != count or any(row["cache_length"] != 0 for row in starts):
        raise ValueError("episode histories were not reset independently")
    distances = [ep["distance_to_goal_m"] for ep in details if ep["distance_to_goal_m"] is not None]
    return {
        "stage": stage,
        "status": "complete",
        "episodes": count,
        "successes": sum(ep["success"] for ep in details),
        "sr": statistics.mean(ep["success"] for ep in details),
        "spl": statistics.mean(ep["spl"] for ep in details),
        "navigation_error_m": statistics.mean(distances) if len(distances) == count else None,
        "unreachable_final_positions": count - len(distances),
        "oracle_sr": statistics.mean(ep["oracle_success"] for ep in details),
        "mean_path_length_m": statistics.mean(ep["path_length_m"] for ep in details),
        "mean_primitive_steps": statistics.mean(ep["steps"] for ep in details),
        "invalid_action_episode_ids": [ep["episode_id"] for ep in details if ep["termination"] == "invalid_action"],
        "model_calls": len(http),
        "model_invocations": len(timings),
        "mean_batch_occupancy": len(http) / len(timings),
        "calls_per_inference_second": 1000 * len(http) / sum(timings),
        "amortized_ms_per_call": sum(timings) / len(http),
        "model_e2e_ms": timing_summary(timings),
        "model_forward_ms": timing_summary(forward),
        "http_roundtrip_ms": timing_summary(http),
        "model_inference_seconds": sum(timings) / 1000,
        "episode_wall_seconds": sum(ep["wall_seconds"] for ep in details),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in evidence),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in evidence),
        "runtime": indexed[max(indexed)]["runtime"],
        "manifest_sha256": report["manifest_sha256"],
        "report_sha256": hashlib.sha256((folder / f"{split}.json").read_bytes()).hexdigest(),
        "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": report.get("selection"),
        "protocol": report["protocol"],
        "model_service": report["model_service"],
        "calibration_overlap_episode_ids": report.get("calibration_overlap_episode_ids", []),
        "timing_scope": "Decoded CPU RGB to CPU actions; shared tensor elapsed time counted once. Excludes disk, warmup, HTTP transport and simulation. Amortized time describes throughput, not request latency; closed-loop workloads can differ.",
    }


def paired_summary(root: Path, datasets: tuple[str, ...] = ("r2r", "rxr")) -> dict[str, Any]:
    """Join complete pairs, requiring identical sources, settings and initial observations."""
    result: dict[str, Any] = {"acceptance_threshold": None, "datasets": {}}
    if not datasets or any(split not in ("r2r", "rxr") for split in datasets) or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be a nonempty unique selection of r2r/rxr")
    for split in datasets:
        stages = [f"{precision}-{split}" for precision in ("bf16", "int8")]
        reports = [read(root / stage / f"{split}.json") for stage in stages]
        pair = compare(*reports, max_fewer=0)
        for key in ("max_fewer_successes", "success_rate_within_tolerance", "latency_admission_evaluated"):
            pair.pop(key)
        ready = [r["model_service"] for r in reports]
        for key in ("source_python_sha256",):
            if ready[0]["environment"][key] != ready[1]["environment"][key]:
                raise ValueError("paired model sources differ")
        if ready[0]["allocator_config"] != ready[1]["allocator_config"]:
            raise ValueError("paired allocators differ")
        for report, service in zip(reports, ready, strict=True):
            if service.get("navigation_manifest_sha256", report["manifest_sha256"]) != report["manifest_sha256"]:
                raise ValueError("service prewarm and navigation episode selections differ")
        for key in ("source_snapshot", "adapter_sha256", "loading_adapter_sha256", "float32_matmul_precision"):
            if ready[0].get(key) != ready[1].get(key):
                raise ValueError(f"paired services differ: {key}")
        # Output paths are run-local; the manifest's contents are compared above.
        ignored = {"precision_experiment", "output_dir", "navigation_manifest"}
        configs = [{k: v for k, v in r["config"].items() if k not in ignored} for r in ready]
        if configs[0] != configs[1]:
            raise ValueError("paired runtime configurations differ")
        for summary in reports[0]["episodes"]:
            episodes = [read(root / stage / split / f"episode-{summary['episode_id']}.json") for stage in stages]
            for key in ("start_position", "start_rotation", "shortest_path_m", "habitat_sim_version"):
                if episodes[0][key] != episodes[1][key]:
                    raise ValueError(f"paired initial condition differs: {key}")
            if episodes[0]["trace"][0]["image_sha256"] != episodes[1]["trace"][0]["image_sha256"]:
                raise ValueError("paired initial RGB differs")
        pair["profiles"] = {stage.split("-")[0]: read(root / stage / "summary.json") for stage in stages}
        pair["paired_initial_observation_audits"] = len(reports[0]["episodes"])
        result["datasets"][split] = pair
    return result


def main() -> None:
    """Summarize one complete stage or compare the two complete dataset pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", help="Completed stage folder name, or compare for the legacy precision pairs")
    parser.add_argument("--split", choices=("r2r", "rxr"))
    parser.add_argument("--datasets", choices=("r2r", "rxr"), nargs="+", default=["r2r", "rxr"])
    args = parser.parse_args()
    result = (
        paired_summary(args.root, tuple(args.datasets))
        if args.stage == "compare"
        else summarize_stage(args.root, args.stage, split=args.split)
    )
    output = args.root / "comparison.json" if args.stage == "compare" else args.root / args.stage / "summary.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "status": "complete", "episodes": result.get("episodes")}))


if __name__ == "__main__":
    main()
