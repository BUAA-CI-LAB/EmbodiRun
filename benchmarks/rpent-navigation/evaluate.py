"""Paired ActiveVLN / RPent+ActiveVLN closed-loop R2R evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from agents.rpent.navigation.events import EventPolicy
from agents.rpent.navigation.images import JPEG_QUALITY
from agents.rpent.navigation.skill import NavigationSkill
from simulator import HabitatBackend, SimulatorProcess

from embodirun.bindings.unitree.go2.activevln.policy import ActiveVLNPolicy, check_capabilities
from embodirun.model_services.backends.embodiinfer.compat import connect_client


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(output, episodes, arms):
    results = {arm: [] for arm in arms}
    for index in range(len(episodes)):
        for arm in arms:
            path = output / f"{index:03d}-{arm}" / "result.json"
            if path.exists():
                results[arm].append(json.loads(path.read_text()))
    summary = {"expected_per_arm": len(episodes), "arms": {}}
    for arm, rows in results.items():
        valid = [row for row in rows if row.get("status") == "complete"]
        summary["arms"][arm] = {
            "completed": len(valid),
            "errors": len(rows) - len(valid),
            "successes": sum(row["success"] for row in valid),
            "sr": sum(row["success"] for row in valid) / len(valid) if valid else None,
            "spl": sum(row["spl"] for row in valid) / len(valid) if valid else None,
            "elapsed_s": sum(row["elapsed_s"] for row in rows),
            "task_elapsed_s": sum(row["task_elapsed_s"] for row in valid)
            if valid and all("task_elapsed_s" in row for row in valid)
            else None,
            "episode_setup_s": sum(row["episode_setup_s"] for row in valid)
            if valid and all("episode_setup_s" in row for row in valid)
            else None,
        }
    if set(arms) == {"baseline", "rpent"}:
        indexed = {
            arm: {row["episode_id"]: row for row in rows if row["status"] == "complete"}
            for arm, rows in results.items()
        }
        ids = indexed["baseline"].keys() & indexed["rpent"].keys()
        summary["paired"] = {
            "count": len(ids),
            "rpent_only_success": sum(
                indexed["rpent"][key]["success"] and not indexed["baseline"][key]["success"] for key in ids
            ),
            "baseline_only_success": sum(
                indexed["baseline"][key]["success"] and not indexed["rpent"][key]["success"] for key in ids
            ),
        }
    save(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument(
        "--simulator-command", type=Path, required=True, help="JSON argv array; supports SSH to an isolated worker"
    )
    parser.add_argument("--endpoint", help="model endpoint when the evaluator owns the local simulator")
    parser.add_argument(
        "--remote-executor", action="store_true", help="simulator-command launches executor.py on the GPU host"
    )
    parser.add_argument("--token-env", default="ACTIVEVLN_BENCHMARK_TOKEN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=("baseline", "rpent"), default=["baseline", "rpent"])
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-backend", choices=("codex", "shared-model"), default="codex")
    parser.add_argument("--shared-tool-examples", action="store_true", help="include fixed tool-format examples")
    parser.add_argument(
        "--shared-plan-tools", action="store_true", help="enable model-selected advancement of saved ordered goals"
    )
    parser.add_argument(
        "--shared-source-tools", action="store_true", help="enable literal task-span plans and current-goal references"
    )
    parser.add_argument(
        "--shared-tools",
        nargs="+",
        choices=("observe", "navigate", "navigate_task", "advance_goal", "finish", "plan_task", "continue_goal"),
        default=["observe", "navigate", "finish"],
    )
    parser.add_argument("--planner-timeout-s", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--profile", choices=("event", "legacy"), default="event")
    parser.add_argument("--event-max-chunks", type=int, default=32)
    args = parser.parse_args()
    if "rpent" in args.arms and args.planner_backend == "codex" and not args.planner_model:
        parser.error("--planner-model is required for the Codex backend")
    if args.planner_backend == "shared-model" and (not args.endpoint or args.profile != "event"):
        parser.error("shared-model requires --endpoint and --profile event")
    if args.planner_backend == "shared-model" and args.planner_model:
        parser.error("shared-model uses the navigation checkpoint; omit --planner-model")
    if args.shared_tool_examples and (
        args.planner_backend != "shared-model" or not {"navigate", "finish"} <= set(args.shared_tools)
    ):
        parser.error("--shared-tool-examples requires shared-model with navigate and finish")
    if args.shared_plan_tools and (args.planner_backend != "shared-model" or args.profile != "event"):
        parser.error("--shared-plan-tools requires shared-model and the event profile")
    if "advance_goal" in args.shared_tools and not args.shared_plan_tools:
        parser.error("advance_goal requires --shared-plan-tools")
    if args.shared_source_tools and (args.planner_backend != "shared-model" or args.profile != "event"):
        parser.error("--shared-source-tools requires shared-model and the event profile")
    if {"plan_task", "continue_goal"} & set(args.shared_tools) and not args.shared_source_tools:
        parser.error("plan_task and continue_goal require --shared-source-tools")
    event_policy = EventPolicy(max_chunks=args.event_max_chunks)
    if args.count < 1 or args.offset < 0 or len(set(args.arms)) != len(args.arms):
        parser.error("invalid selection or duplicate arm")
    episodes = json.loads(args.manifest.read_text())["episodes"][args.offset : args.offset + args.count]
    if len(episodes) != args.count or len({str(e["episode_id"]) for e in episodes}) != args.count:
        raise ValueError("selection must contain exactly count distinct episode IDs")
    command = json.loads(args.simulator_command.read_text())
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError("simulator command must be a nonempty argv array")
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2]
    sources = sorted([*Path(__file__).parent.glob("*.py"), *(source_root / "agents/rpent/navigation").glob("*.py")])
    protocol = {
        "version": 1,
        "remote_executor": args.remote_executor,
        "dataset": "R2R val-unseen",
        "seed": 42,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "offset": args.offset,
        "count": args.count,
        "episode_ids": [str(e["episode_id"]) for e in episodes],
        "max_primitive_steps": 500,
        "max_model_calls": 500,
        "success": "final STOP and geodesic < 3m; planner claims never count as ground truth",
        "local_stop": "counts as one primitive step; only baseline local STOP terminates the episode",
        "planner_model": args.planner_model,
        "planner_backend": args.planner_backend,
        "shared_tools": args.shared_tools if args.planner_backend == "shared-model" else None,
        "shared_tool_examples": args.shared_tool_examples,
        "shared_plan_tools": args.shared_plan_tools,
        "shared_source_tools": args.shared_source_tools,
        "planner_reasoning": "low" if args.planner_backend == "codex" else None,
        "profile": args.profile,
        "initial_observation_attached": args.profile == "event",
        "event_policy": asdict(event_policy) if args.profile == "event" else None,
        "planner_images": {"format": "jpeg", "quality": JPEG_QUALITY, "resize": False}
        if args.profile == "event"
        else {"format": "png", "resize": False},
        "planner_timeout_s": args.planner_timeout_s,
        "max_tool_calls": args.max_tool_calls,
        "memory": "isolated per episode; no cross-episode transfer",
        "timing": {
            "planner_import_s": "once per evaluator start, before episodes; retained in startup.jsonl",
            "episode_setup_s": "executor initialization and simulator episode reset",
            "task_elapsed_s": "after reset through policy/planner execution and final metric read",
            "elapsed_s": "setup plus task; excludes final artifact writes and session cleanup",
        },
        "arms": args.arms,
        "observations": "RGB, executed discrete actions, collision count, consumed step count; no map, coordinates or evaluation metrics",
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("protocol/source changed: use a new output directory")
    save(protocol_path, protocol)
    planner_import_started = time.monotonic()
    if "rpent" in args.arms:
        from agents.rpent.navigation.toolkit import NavigationToolkit, shared_tool_examples

        if args.planner_backend == "shared-model":
            from agents.rpent.navigation.shared_model import SharedModelHttpClient, SharedModelPlanner
        else:
            from agents.rpent.navigation.planner import solve
    with (args.output / "startup.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "planner_import_s": time.monotonic() - planner_import_started,
                    "planner_backend": args.planner_backend if "rpent" in args.arms else None,
                }
            )
            + "\n"
        )
    client = None
    if not args.remote_executor:
        if not args.endpoint:
            parser.error("--endpoint is required without --remote-executor")
        client = connect_client(args.endpoint, os.environ.get(args.token_env), timeout_s=180)
        check_capabilities(client)
    simulator = SimulatorProcess(command, args.output)
    try:
        for index, episode in enumerate(episodes):
            # Alternate ordering to reduce time/temperature effects.
            arms = args.arms if index % 2 == 0 else list(reversed(args.arms))
            for arm in arms:
                output = args.output / f"{index:03d}-{arm}"
                output.mkdir(exist_ok=True)
                if (output / "result.json").exists():
                    prior = json.loads((output / "result.json").read_text())
                    if prior["status"] != "complete":
                        raise RuntimeError(f"previous failed attempt requires an explicit new run: {output}")
                    continue
                started = time.monotonic()
                save(
                    args.output / "state.json",
                    {
                        "status": "running",
                        "index": index,
                        "episode_id": str(episode["episode_id"]),
                        "arm": arm,
                        "pid": os.getpid(),
                    },
                )
                timings = []
                if args.remote_executor:
                    from remote import RemoteSkill

                    skill = backend = RemoteSkill(
                        simulator,
                        episode,
                        args.scenes,
                        max_runtime_s=args.planner_timeout_s,
                        expected_sources=protocol["source_sha256"],
                        profile=args.profile if arm == "rpent" else "legacy",
                    )
                else:
                    backend = HabitatBackend(simulator, episode, args.scenes)
                    skill = NavigationSkill(
                        backend,
                        lambda events=timings: ActiveVLNPolicy(
                            client, robot_id="habitat-sim", event_sink=events.append
                        ),
                        max_runtime_s=args.planner_timeout_s,
                    )
                task_started = time.monotonic()
                result = {
                    "episode_id": str(episode["episode_id"]),
                    "arm": arm,
                    "episode_setup_s": task_started - started,
                }
                try:
                    instruction = episode["instruction"]
                    if isinstance(instruction, dict):
                        instruction = instruction["instruction_text"]
                    if arm == "baseline":
                        feedback = skill.navigate(instruction, max_chunks=500, terminal_stop=True)
                        result["termination"] = feedback.reason
                    else:
                        toolkit = NavigationToolkit(
                            skill,
                            output / "agent",
                            max_tool_calls=args.max_tool_calls,
                            profile=args.profile,
                            event_policy=event_policy,
                            initial_task=instruction,
                            plan_tools=args.shared_plan_tools,
                            source_tools=args.shared_source_tools,
                        )
                        if args.planner_backend == "shared-model":
                            capabilities = (
                                backend.model_capabilities if args.remote_executor else check_capabilities(client)
                            )
                            identity = capabilities.get("shared_text", {}).get("model_instance")
                            if not isinstance(identity, str) or not identity:
                                raise RuntimeError("navigation executor did not attest its shared model identity")
                            transport = SharedModelHttpClient(
                                args.endpoint, os.environ.get(args.token_env), expected_model_instance=identity
                            )
                            try:
                                initial_image, feedback = toolkit.initial_observation()
                                planner = SharedModelPlanner(
                                    transport,
                                    initial_image=initial_image.read_bytes(),
                                    initial_feedback=feedback,
                                    timeout_s=args.planner_timeout_s,
                                    check_cancelled=toolkit.raise_if_cancelled,
                                    allowed_tools=tuple(args.shared_tools),
                                    examples=shared_tool_examples(plan_tools="advance_goal" in args.shared_tools)
                                    if args.shared_tool_examples
                                    else (),
                                ).solve(
                                    system_prompt=toolkit.system_prompt,
                                    user_message=instruction,
                                    toolkit=toolkit,
                                    max_turns=args.max_tool_calls,
                                )
                            finally:
                                transport.close()
                            save(
                                output / "agent" / "planner.json",
                                {"messages": planner.messages, "stats": planner.stats},
                            )
                        else:
                            planner = solve(
                                toolkit,
                                instruction,
                                output / "agent",
                                model=args.planner_model,
                                timeout_s=args.planner_timeout_s,
                            )
                        result.update(
                            planner_stats=planner.stats,
                            planner_error=planner.error,
                            planner_finish=toolkit.finish_result,
                            tool_calls=toolkit.tool_calls,
                        )
                        algorithm_failure = args.planner_backend == "shared-model" and planner.stats.get(
                            "failure_kind"
                        ) in {"invalid_tool_call", "turn_budget", "episode_deadline"}
                        if planner.error and not algorithm_failure:
                            raise RuntimeError(planner.error)
                        if algorithm_failure:
                            result["termination"] = planner.stats["failure_kind"]
                        elif toolkit.finish_result is None:
                            result["termination"] = "planner_returned_without_finish"
                        else:
                            result["termination"] = toolkit.finish_result["status"]
                    result.update(
                        status="complete",
                        **backend.metrics(),
                        model_calls=skill.model_calls,
                        subgoals=skill.subgoals,
                    )
                    finished = time.monotonic()
                    result.update(elapsed_s=finished - started, task_elapsed_s=finished - task_started)
                except BaseException as error:
                    finished = time.monotonic()
                    result.update(
                        status="error",
                        error=repr(error),
                        elapsed_s=finished - started,
                        task_elapsed_s=finished - task_started,
                    )
                    save(output / "result.json", result)
                    save(
                        args.output / "state.json",
                        {
                            "status": "error",
                            "index": index,
                            "episode_id": str(episode["episode_id"]),
                            "arm": arm,
                            "error": repr(error),
                        },
                    )
                    summarize(args.output, episodes, args.arms)
                    raise
                finally:
                    skill.close()
                save(output / "result.json", result)
                save(output / "trajectory.json", backend.trajectory)
                save(output / "model-timings.json", backend.timings if args.remote_executor else timings)
                if args.remote_executor:
                    save(output / "transport-timings.json", backend.transport_timings)
                print(json.dumps(summarize(args.output, episodes, args.arms)), flush=True)
        save(args.output / "state.json", {"status": "complete"})
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
