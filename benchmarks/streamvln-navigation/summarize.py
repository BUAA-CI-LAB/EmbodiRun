"""Summarize complete closed-loop reports without substituting replay for SR/SPL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from metrics import episode_metrics


def distribution(values: list[float]) -> dict[str, float]:
    """Report empirical quantiles and total-time-derived request throughput."""
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timings must be nonempty, finite and nonnegative")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return dict(mean=statistics.fmean(values), p50=percentile(0.5), p95=percentile(0.95), p99=percentile(0.99))


def timed_invocations(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count a tensor batch's shared elapsed time once, including partial tails."""
    if not any("tensor_batch" in turn["model"] for turn in calls):
        return calls
    batches: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for turn in calls:
        batch = turn["model"].get("tensor_batch")
        if batch is None:
            raise ValueError("cannot mix tensor-batch and serialized timing records")
        identity = (turn.get("model_service_index", 0), batch["index"])
        batches.setdefault(identity, []).append(turn)
    result = []
    model_fields = ("e2e_ms", "prefill_ms", "decode_ms", "gpu_inference_ms", "pure_inference_ms")
    for identity in sorted(batches):
        rows = batches[identity]
        batch = rows[0]["model"]["tensor_batch"]
        if len(rows) != batch["observations"] or not 1 <= len(rows) <= batch["capacity"]:
            raise ValueError("tensor batch is missing rows or exceeds its declared capacity")
        # HTTP service policy_ms includes per-request queueing and is not a
        # shared model timer, even when the requests execute in one batch.
        measured = {key: rows[0]["timing"][key] for key in model_fields if key in rows[0]["timing"]}
        if any(
            row["model"]["tensor_batch"] != batch
            or {key: row["timing"][key] for key in model_fields if key in row["timing"]} != measured
            for row in rows
        ):
            raise ValueError("tensor batch rows disagree on shared timing or occupancy")
        result.append(rows[0])
    return result


def load_complete(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Require every declared episode and validate summaries against raw evidence."""
    report = json.loads(path.read_text())
    count = report["expected_episodes"]
    rows = report["episodes"]
    if report["status"] != "complete" or len(rows) != count:
        raise ValueError(f"{path}: requires {count} complete episodes")
    traces = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in traces or row["status"] != "complete":
            raise ValueError("duplicate or incomplete episode")
        raw = json.loads((path.parent / report["dataset"].lower() / f"episode-{episode_id}.json").read_text())
        calculated = episode_metrics(
            raw["stopped"], raw["distance_to_goal_m"], raw["shortest_path_m"], raw["path_length_m"]
        )
        if calculated["success"] != row["success"] or abs(calculated["spl"] - row["spl"]) > 1e-9:
            raise ValueError("summary disagrees with episode evidence")
        if any(turn["primitive_step"] != turn["model"]["stream_step_id"] for turn in raw["trace"]):
            raise ValueError("model and simulator primitive history differ")
        traces[episode_id] = raw
    calls = [turn for episode in traces.values() for turn in episode["trace"]]
    invocations = timed_invocations(calls)
    timing_names = ("e2e_ms", "prefill_ms", "decode_ms", "gpu_inference_ms", "pure_inference_ms")
    if "tensor_batch" in invocations[0]["model"]:
        fields = set(invocations[0]["timing"]).intersection(timing_names)
        if "e2e_ms" not in fields or any(
            set(turn["timing"]).intersection(timing_names) != fields for turn in invocations
        ):
            raise ValueError("tensor invocations require consistent timing fields including e2e_ms")
        timing_names = tuple(name for name in timing_names if name in fields)
    elif any(name not in turn["timing"] for turn in invocations for name in timing_names):
        raise ValueError("serialized reports require all measured stage timings")
    latencies = {name: distribution([turn["timing"][name] for turn in invocations]) for name in timing_names}
    inference_seconds = sum(turn["timing"]["e2e_ms"] for turn in invocations) / 1000
    ready = report["model_service"]
    summary = dict(
        report=str(path),
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        episodes=count,
        successes=sum(row["success"] for row in rows),
        sr=sum(row["success"] for row in rows) / count,
        spl=statistics.fmean(row["spl"] for row in rows),
        calls=len(calls),
        model_invocations=len(invocations),
        mean_batch_occupancy=len(calls) / len(invocations),
        amortized_ms_per_call=1000 * inference_seconds / len(calls),
        primitive_steps=sum(row["steps"] for row in rows),
        inference_seconds=inference_seconds,
        calls_per_inference_second=len(calls) / inference_seconds,
        latency_ms=latencies,
        http_roundtrip_ms=distribution([turn["http_roundtrip_ms"] for turn in calls]),
        cuda_peak_allocated_mib=max(turn["model"]["cuda_peak_allocated_mib"] for turn in calls),
        environment=ready["environment"],
        config=ready["config"],
        graph_stats=invocations[-1]["model"]["graph_stats"],
        latency_note="Closed-loop workloads can differ. Latencies describe whole model invocations; tensor batches are counted once. Amortized per-call time is a throughput metric. Check service and renderer provenance for GPU placement.",
    )
    return report, traces, summary


def compare(left: tuple, right: tuple) -> dict[str, Any]:
    """Compare paired navigation outcomes and locate the earliest output differences."""
    baseline, baseline_traces, baseline_summary = left
    candidate, candidate_traces, candidate_summary = right
    for key in ("manifest_sha256", "protocol", "expected_episodes"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"incomparable reports: {key}")
    if baseline_traces.keys() != candidate_traces.keys():
        raise ValueError("episode identities differ")
    lost, gained, tokens, actions = [], [], {}, {}
    for identity, before in baseline_traces.items():
        after = candidate_traces[identity]
        if before["trace"][0]["image_sha256"] != after["trace"][0]["image_sha256"]:
            raise ValueError(f"episode {identity}: initial simulator observations differ")
        if before["success"] and not after["success"]:
            lost.append(identity)
        if after["success"] and not before["success"]:
            gained.append(identity)
        for first, second in zip(before["trace"], after["trace"]):
            if first["model"]["tokens"] != second["model"]["tokens"] and identity not in tokens:
                tokens[identity] = first["primitive_step"]
            if first["model"]["data"] != second["model"]["data"]:
                actions[identity] = first["primitive_step"]
                break
        else:
            if len(before["trace"]) != len(after["trace"]):
                actions[identity] = "different trajectory length"
    count = baseline["expected_episodes"]
    return dict(
        baseline_report=baseline_summary["report"],
        candidate_report=candidate_summary["report"],
        sr_change_percentage_points=(len(gained) - len(lost)) / count * 100,
        lost_success_episode_ids=lost,
        gained_success_episode_ids=gained,
        first_different_token_primitive_steps=tokens,
        first_different_action_primitive_steps=actions,
        exact_action_trajectories=count - len(actions),
        lossless_certified=False,
        note="Closed-loop agreement alone does not establish numerical losslessness; quantization is a separate precision experiment.",
    )


def main() -> None:
    """Write complete-run summaries and every paired comparison, including devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [load_complete(path) for path in args.reports]
    result = dict(
        summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=[item[2] for item in loaded],
        comparisons=[compare(left, right) for index, left in enumerate(loaded) for right in loaded[index + 1 :]],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps([dict(episodes=r[2]["episodes"], sr=r[2]["sr"], spl=r[2]["spl"]) for r in loaded]))


if __name__ == "__main__":
    main()
