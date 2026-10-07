"""Run a finite evaluation manifest against existing HTTP inference replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodirun.application.simulation.contracts import EpisodeRequest, SimulationServiceConfig
from embodirun.application.simulation.evaluation import EvaluationConfig, EvaluationRunner


def main(argv: list[str] | None = None) -> int:
    """Parse jobs, run process-isolated simulators and save the complete report."""
    parser = argparse.ArgumentParser(prog="embodirun-evaluate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.config.read_text())
    service = SimulationServiceConfig.from_json(json.dumps(manifest["simulation"]))
    config = EvaluationConfig(**manifest.get("evaluation", {}))
    requests = [EpisodeRequest.from_payload(item) for item in manifest["episodes"]]
    report = EvaluationRunner(service, config).run(requests)
    report["manifest"] = manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
