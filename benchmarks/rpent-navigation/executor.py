"""Keep simulator observations and recurrent inference together on the GPU host.

The evaluator talks JSON over stdin/stdout (optionally SSH). Only bounded skill
feedback travels to the planner host; metrics use a separate evaluator command.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agents.rpent.navigation.skill import NavigationSkill
from simulator import HabitatBackend, SimulatorProcess

from embodirun.bindings.unitree.go2.activevln.policy import ActiveVLNPolicy, check_capabilities
from embodirun.model_services.backends.embodiinfer.compat import connect_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--habitat-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = connect_client(args.endpoint, args.token_file.read_text().strip(), timeout_s=180)
    check_capabilities(client)
    simulator = SimulatorProcess(
        [args.habitat_python, "-u", str(Path(__file__).with_name("habitat_worker.py"))], args.output
    )
    skill = backend = None
    timings = []
    root = Path(__file__).resolve().parents[2]
    sources = [*Path(__file__).parent.glob("*.py"), *(root / "agents/rpent/navigation").glob("*.py")]
    fingerprints = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    try:
        for line in sys.stdin:
            command = json.loads(line)
            op = command.pop("op")
            try:
                if op == "close":
                    break
                if op == "reset":
                    if skill is not None:
                        skill.close()
                    backend = HabitatBackend(simulator, command["episode"], command["scenes"])
                    timings = []
                    skill = NavigationSkill(
                        backend,
                        lambda events=timings: ActiveVLNPolicy(
                            client, robot_id="habitat-sim", event_sink=events.append
                        ),
                        max_runtime_s=command["max_runtime_s"],
                    )
                    result = {"ready": True, "source_sha256": fingerprints}
                elif skill is None:
                    raise RuntimeError("reset an episode before using it")
                elif op in ("observe", "navigate"):
                    feedback = skill.observe() if op == "observe" else skill.navigate(**command)
                    result = asdict(feedback)
                    result["png"] = base64.b64encode(feedback.png).decode()
                elif op == "finish":
                    skill.finish(**command)
                    result = {"finished": True}
                elif op == "result":
                    result = {"metrics": backend.metrics(), "trajectory": backend.trajectory, "timings": timings}
                elif op == "close_episode":
                    skill.close()
                    result = {"closed": True}
                else:
                    raise ValueError("unknown executor command")
                if skill is not None:
                    result.update(model_calls=skill.model_calls, subgoals=skill.subgoals, finished=skill.finished)
                print(json.dumps(result, allow_nan=False), flush=True)
            except Exception as error:
                print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
    finally:
        if skill is not None:
            skill.close()
        simulator.close()


if __name__ == "__main__":
    main()
