"""Execution and paired quality checks for the selected StreamVLN episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def primitive_actions(rows: list[list[float]]) -> list[int]:
    """Validate StreamVLN's metre/degree action chunk without changing its plan."""
    if not 1 <= len(rows) <= 4 or any(len(row) != 2 for row in rows):
        raise ValueError("expected one to four [action,value] rows")
    result = []
    units = {0: 0.0, 1: 0.25, 2: 15.0, 3: 15.0}
    for kind, value in rows:
        if not all(math.isfinite(x) for x in (kind, value)) or int(kind) != kind:
            raise ValueError("action entries must be finite with integer action IDs")
        kind = int(kind)
        if kind not in units or value != units[kind]:
            raise ValueError("action exceeds the StreamVLN grammar")
        result.append(kind)
        if kind == 0:
            break
    return result


def episode_metrics(stopped: bool, distance: float | None, shortest: float, length: float) -> dict[str, Any]:
    """Require STOP within 3 m; compute SPL using simulator geodesic start distance."""
    if not math.isfinite(shortest) or shortest < 0 or not math.isfinite(length) or length < 0:
        raise ValueError("path distances must be finite and nonnegative")
    success = bool(stopped and distance is not None and math.isfinite(distance) and distance < 3.0)
    denominator = max(shortest, length)
    return {"success": success, "spl": float(success) * (shortest / denominator if denominator else 1.0)}


def compare(reference: dict[str, Any], candidate: dict[str, Any], max_fewer: int = 0) -> dict[str, Any]:
    """Compare complete paired episodes without substituting action agreement for SR."""
    if type(max_fewer) is not int or max_fewer < 0:
        raise ValueError("max_fewer must be a nonnegative integer")
    for key in ("dataset", "manifest_sha256", "protocol"):
        if reference[key] != candidate[key]:
            raise ValueError(f"evaluation conditions differ: {key}")
    count = reference.get("expected_episodes", 48)
    if type(count) is not int or count < 1 or candidate.get("expected_episodes", 48) != count:
        raise ValueError("evaluation conditions differ: expected_episodes")
    indexed = []
    for report in (reference, candidate):
        rows = report["episodes"]
        if report["status"] != "complete" or len(rows) != count:
            raise ValueError(f"quality admission requires all {count} episodes")
        byid = {str(row["episode_id"]): row for row in rows}
        if len(byid) != count or any(row["status"] != "complete" for row in rows):
            raise ValueError("duplicate or unfinished episodes")
        for row in rows:
            calculated = episode_metrics(
                row["stopped"], row["distance_to_goal_m"], row["shortest_path_m"], row["path_length_m"]
            )
            if calculated["success"] != row["success"] or abs(calculated["spl"] - row["spl"]) > 1e-9:
                raise ValueError("episode summary does not agree with raw metric evidence")
        indexed.append(byid)
    left, right = indexed
    if left.keys() != right.keys():
        raise ValueError("episode identities differ")
    lost = [key for key in left if left[key]["success"] and not right[key]["success"]]
    gained = [key for key in left if not left[key]["success"] and right[key]["success"]]
    counts = [sum(row["success"] for row in group.values()) for group in indexed]
    return {
        "dataset": reference["dataset"],
        "episodes": count,
        "baseline_successes": counts[0],
        "candidate_successes": counts[1],
        "baseline_sr": counts[0] / count,
        "candidate_sr": counts[1] / count,
        "sr_change_percentage_points": (counts[1] - counts[0]) / count * 100,
        "baseline_spl": sum(r["spl"] for r in left.values()) / count,
        "candidate_spl": sum(r["spl"] for r in right.values()) / count,
        "lost_success_episode_ids": lost,
        "gained_success_episode_ids": gained,
        "max_fewer_successes": max_fewer,
        "success_rate_within_tolerance": counts[0] - counts[1] <= max_fewer,
        "latency_admission_evaluated": False,
    }


def main() -> None:
    """Write an auditable SR/SPL comparison of two complete closed-loop reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-fewer-successes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = args.reference.read_bytes(), args.candidate.read_bytes()
    result = compare(json.loads(reference), json.loads(candidate), args.max_fewer_successes)
    result["reference_sha256"] = hashlib.sha256(reference).hexdigest()
    result["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success_rate_within_tolerance"] else 1)


if __name__ == "__main__":
    main()
