"""Run paired closed-loop StreamVLN navigation against frozen HTTP model services."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import select
import subprocess
import time
import uuid
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

from metrics import episode_metrics, primitive_actions

from embodirun.model_services.backends.embodiinfer.http import EmbodiInferHttpClient
from embodirun.model_services.contracts import ImagePayload, PolicyObservation, PolicyResult

PROTOCOL = {
    "evaluation": "independent closed-loop trajectories; identical declared episode selections per dataset",
    "checkpoint_action_grammar": "StreamVLN symbolic actions, up to four primitive actions per response",
    "success": "STOP and finite geodesic distance strictly below 3 metres",
    "max_primitive_steps": 500,
    "forward_m": 0.25,
    "turn_degrees": 15,
    "rgb_width": 640,
    "rgb_height": 480,
    "hfov_degrees": 90,
    "camera_height_m": 1.25,
    "agent_height_m": 1.5,
    "agent_radius_m": 0.1,
    "allow_sliding": True,
    "seed": 42,
    "spl": "success * initial geodesic / max(initial geodesic, travelled Euclidean path)",
    "history": "reset per episode; own responses; 32 primitive-action windows; all intervening observations; eight slow visual memories",
    "invalid_action": "episode failure with diagnostic; infrastructure errors fail the run",
    "observation_encoding": "lossless PNG over HTTP; decode excluded from policy timer",
    "physical_gpu": 1,
}


def digest(path: Path) -> str:
    """Hash a frozen source report or episode manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    """Commit reports atomically after each finished episode."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_selection(root: Path, split: str, count: int) -> tuple[Path, list[dict[str, Any]]]:
    """Require the full declared manifest, with unique IDs, before running any episode."""
    manifest = root / f"selected-{split}-navigation-{count}.json"
    episodes = json.loads(manifest.read_text())["episodes"]
    if len(episodes) != count:
        raise ValueError(f"{split} manifest must contain exactly {count} episodes, got {len(episodes)}")
    if len({str(episode["episode_id"]) for episode in episodes}) != count:
        raise ValueError(f"{split} manifest contains duplicate episode IDs")
    return manifest, episodes


class SimulatorProcess:
    """Keep Habitat in its compatible interpreter with a bounded JSON exchange."""

    def __init__(self, python: Path, output: Path, gpu: str = "1") -> None:
        self.log = (output / "habitat.log").open("a")
        self.process = subprocess.Popen(
            [str(python), "-u", str(Path(__file__).with_name("habitat_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH="", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet"),
        )

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        """Send one simulation command; never reuse an observation after failure."""
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.process.stdout], [], [], min(5, max(0, deadline - time.monotonic())))
            if ready:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"Habitat exited: {self.process.poll()}; inspect habitat.log")
                if line.startswith("{"):
                    return json.loads(line)
                self.log.write(line)
                self.log.flush()
        raise TimeoutError("Habitat did not return an observation within 180 seconds")

    def close(self) -> None:
        """Reap only this runner's simulator process."""
        if self.process.poll() is None:
            with suppress(BrokenPipeError):
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=15)
        self.log.close()


def episode_steps(
    client: EmbodiInferHttpClient,
    simulator: SimulatorProcess,
    episode: dict[str, Any],
    scenes: Path,
    output: Path,
    slot: int = 0,
) -> Generator[PolicyObservation, PolicyResult, dict[str, Any]]:
    """Yield each live observation and execute the corresponding policy response."""
    observation = simulator.request({"op": "reset", "episode": episode, "scenes_dir": str(scenes), "seed": 42})
    session = client.open_session(robot_id="habitat-sim", action_space="streamvln.action_chunk.v1")
    result = {
        "episode_id": str(episode["episode_id"]),
        "session_id": session.session_id,
        "benchmark_slot": slot,
        "status": "running",
        "trace": [],
        "start_position": observation["position"],
        "start_rotation": observation["rotation"],
        "habitat_sim_version": observation["habitat_sim_version"],
    }
    started = time.monotonic()
    termination = "step_limit"
    try:
        while not observation["stopped"] and observation["steps"] < 500:
            image = base64.b64decode(observation["png"])
            if not result["trace"]:
                (output / f"episode-{episode['episode_id']}-initial.png").write_bytes(image)
            turn = len(result["trace"])
            frames = [base64.b64decode(frame) for frame in observation["frames"]]
            request = PolicyObservation(
                session_id=session.session_id,
                request_id=uuid.uuid4().hex,
                step_id=turn,
                instruction=episode["instruction"]["instruction_text"],
                state={"benchmark_slot": slot},
                images=tuple(
                    ImagePayload(f"observation.images.rgb.{i}", "image/png", frame) for i, frame in enumerate(frames)
                ),
                metadata={"stream_step_id": observation["steps"]},
            )
            start = time.monotonic()
            response = yield request
            roundtrip_ms = (time.monotonic() - start) * 1000
            if len(response.actions) != 1 or response.actions[0].kind != "action_chunk":
                raise ValueError("model service returned the wrong action space")
            values = dict(response.actions[0].values)
            if values["stream_step_id"] != observation["steps"]:
                raise RuntimeError("model history does not match the current primitive step")
            record = {
                "turn": turn,
                "request_id": request.request_id,
                "http_roundtrip_ms": roundtrip_ms,
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "primitive_frame_sha256": [hashlib.sha256(frame).hexdigest() for frame in frames],
                "primitive_step": observation["steps"],
                "before_position": observation["position"],
                "model": values,
                "timing": dict(response.timing),
            }
            result["trace"].append(record)
            try:
                if not values["parsed_valid"]:
                    raise ValueError("model response did not parse as a valid action chunk")
                actions = primitive_actions(values["data"])
            except ValueError as error:
                record["invalid_action"] = str(error)
                termination = "invalid_action"
                break
            observation = simulator.request({"op": "step", "actions": actions})
            record["executed"] = observation["trajectory"]
            record["distance_to_goal_m"] = observation["distance_to_goal_m"]
            if observation["stopped"]:
                termination = "stop"
            if (turn + 1) % 20 == 0:
                print(
                    f"episode={episode['episode_id']} turns={turn + 1} steps={observation['steps']} distance={observation['distance_to_goal_m']}",
                    flush=True,
                )
        result.update(
            {
                key: observation[key]
                for key in ("stopped", "steps", "distance_to_goal_m", "shortest_path_m", "path_length_m")
            }
        )
        result.update(
            episode_metrics(
                observation["stopped"],
                observation["distance_to_goal_m"],
                observation["shortest_path_m"],
                observation["path_length_m"],
            )
        )
        result.update(status="complete", termination=termination, turns=len(result["trace"]))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            client.close(session.session_id)
        finally:
            result["wall_seconds"] = time.monotonic() - started
            save(output / f"episode-{episode['episode_id']}.json", result)
    return result


def run_episode(
    client: EmbodiInferHttpClient, simulator: SimulatorProcess, episode: dict[str, Any], scenes: Path, output: Path
) -> dict[str, Any]:
    """Drive one complete closed-loop episode using the same step generator."""
    steps = episode_steps(client, simulator, episode, scenes, output)
    try:
        request = next(steps)
        while True:
            try:
                response = client.step(request)
            except Exception as error:
                steps.throw(error)
                raise
            request = steps.send(response)
    except StopIteration as done:
        return done.value
    finally:
        steps.close()


def run_batched_episodes(
    client: EmbodiInferHttpClient,
    simulators: list[SimulatorProcess],
    episodes: list[dict[str, Any]],
    scenes: Path,
    output: Path,
) -> Iterator[dict[str, Any]]:
    """Submit one synchronous round per live slot, refilling only completed episodes.

    The server must confirm all requests in a round shared one tensor invocation.
    HTTP arrival order is irrelevant: the adapter sorts by benchmark_slot.
    """
    pending = iter(episodes)
    active: dict[int, tuple[Any, PolicyObservation]] = {}
    try:
        with ThreadPoolExecutor(max_workers=len(simulators)) as executor:
            while True:
                for slot, simulator in enumerate(simulators):
                    if slot not in active:
                        episode = next(pending, None)
                        if episode is not None:
                            steps = episode_steps(client, simulator, episode, scenes, output, slot)
                            active[slot] = (steps, next(steps))
                if not active:
                    return
                slots = sorted(active)
                futures = {slot: executor.submit(client.step, active[slot][1]) for slot in slots}
                responses = {slot: futures[slot].result() for slot in slots}
                batches = [response.actions[0].values["tensor_batch"] for response in responses.values()]
                if len({batch["index"] for batch in batches}) != 1 or any(
                    batch["observations"] != len(slots) for batch in batches
                ):
                    raise RuntimeError("HTTP round split across tensor batches; reject incomparable schedule")
                for slot in slots:
                    steps, _ = active[slot]
                    try:
                        active[slot] = (steps, steps.send(responses[slot]))
                    except StopIteration as done:
                        del active[slot]
                        yield done.value
    except BaseException as error:
        for steps, _ in active.values():
            with suppress(BaseException):
                steps.throw(error)
        raise
    finally:
        for steps, _ in active.values():
            steps.close()


def main() -> None:
    """Run a specified frozen policy through both complete episode selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--serve-script", type=Path)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=48, help="Expected full manifest size per dataset")
    parser.add_argument("--episode-limit", type=int, help="Optional partial smoke-test limit")
    parser.add_argument("--datasets", choices=("r2r", "rxr"), nargs="+", default=["r2r", "rxr"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--endpoint", help="Existing versioned HTTP model service; no local model launch")
    parser.add_argument("--service-ready", type=Path, help="Provenance emitted by that model service")
    parser.add_argument("--token-file", type=Path, help="Private file containing the service token")
    parser.add_argument("--simulator-gpu", default="1", help="CUDA_VISIBLE_DEVICES for Habitat only")
    args = parser.parse_args()
    if not args.preflight_only:
        if args.endpoint:
            if not all((args.service_ready, args.token_file, args.config)):
                parser.error("external service requires --service-ready, --token-file and --config")
        elif not all((args.source, args.config, args.reference_report, args.model_python, args.serve_script)):
            parser.error("local service requires source, config, reference-report, model-python and serve-script")
    if args.episode_count < 1:
        parser.error("episode-count must be positive")
    if args.episode_limit is None:
        args.episode_limit = args.episode_count
    if not 1 <= args.episode_limit <= args.episode_count:
        parser.error("episode-limit must be in [1,episode-count]")
    protocol = dict(PROTOCOL, physical_gpu=args.simulator_gpu, episode_count=args.episode_count)
    selections = {split: load_selection(args.root, split, args.episode_count) for split in args.datasets}
    if args.batch_size > 1 and (args.resume or len(args.datasets) != 1):
        parser.error("tensor batch evaluation needs one split and a fresh run for deterministic refill")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    simulator = SimulatorProcess(args.habitat_python, args.output, args.simulator_gpu)
    simulators = [simulator]
    process = None
    model_log = None
    try:
        if args.preflight_only:
            episode = selections[args.datasets[0]][1][0]
            result = simulator.request({"op": "reset", "episode": episode, "scenes_dir": str(args.scenes), "seed": 42})
            (args.output / "initial.png").write_bytes(base64.b64decode(result.pop("png")))
            moved = simulator.request({"op": "step", "actions": [1, 2, 3]})
            moved.pop("png")
            stopped = simulator.request({"op": "step", "actions": [0]})
            stopped.pop("png")
            save(
                args.output / "preflight.json",
                {"initial": result, "moved": moved, "stop": stopped, "protocol": protocol},
            )
            print(json.dumps({"preflight": "passed", "initial_distance": result["distance_to_goal_m"]}), flush=True)
            return
        if args.endpoint:
            token = args.token_file.read_text().strip()
            if not token:
                raise ValueError("empty service token")
            ready = json.loads(args.service_ready.read_text())
            endpoint = args.endpoint
        else:
            raise ValueError("use --endpoint with an independently frozen model service")
        original = (
            json.loads(args.reference_report.read_text())
            if args.reference_report
            else {"config": json.loads(args.config.read_text())}
        )
        if args.reference_report and (
            ready["environment"]["source_python_sha256"] != original["environment"]["source_python_sha256"]
        ):
            raise ValueError("model source differs from the original latency report")
        for key, value in original["config"].items():
            if key not in ("validate_data_only", "output_dir") and ready["config"].get(key) != value:
                raise ValueError(f"model configuration differs from the original latency report: {key}")
        client = EmbodiInferHttpClient(endpoint, token=token, timeout_s=240)
        if ready["config"].get("batch_size", 1) != args.batch_size:
            raise ValueError("server and simulator batch sizes differ")
        if args.batch_size > 1:
            for slot in range(1, args.batch_size):
                slot_output = args.output / f"simulator-{slot}"
                slot_output.mkdir()
                simulators.append(SimulatorProcess(args.habitat_python, slot_output, args.simulator_gpu))
        for split in args.datasets:
            manifest, full_selection = selections[split]
            episodes = full_selection[: args.episode_limit]
            selection_metadata = {k: v for k, v in json.loads(manifest.read_text()).items() if k != "episodes"}
            calibration = ready["config"].get("precision_experiment", {})
            calibration_dataset = calibration.get("calibration_dataset") or {}
            same_calibration_split = (split == "r2r" and not selection_metadata.get("split")) or (
                calibration_dataset.get("name", "").lower() == split
                and calibration_dataset.get("split") == selection_metadata.get("split")
            )
            output = args.output / split
            output.mkdir(exist_ok=True)
            report_path = args.output / f"{split}.json"
            report = {
                "dataset": split.upper() if split == "r2r" else "RxR",
                "manifest_sha256": digest(manifest),
                "selection": selection_metadata,
                "protocol": protocol,
                "status": "running",
                "expected_episodes": args.episode_count,
                "episodes": [],
                "model_service": ready,
                "model_services": [ready],
                "latency_reference_sha256": digest(args.reference_report) if args.reference_report else None,
                "config_sha256": digest(args.config),
                "calibration_overlap_episode_ids": sorted(
                    {str(ep["episode_id"]) for ep in episodes}.intersection(
                        str(value)
                        for value in ready["config"].get("precision_experiment", {}).get("calibration_episode_ids", [])
                    )
                )
                if same_calibration_split
                else [],
                "batch_size": args.batch_size,
                "runner_sha256": digest(Path(__file__)),
                "simulator_worker_sha256": digest(Path(__file__).with_name("habitat_worker.py")),
            }
            previous_rows = {}
            if args.resume and report_path.exists():
                previous = json.loads(report_path.read_text())
                for key in ("protocol", "manifest_sha256", "latency_reference_sha256", "config_sha256"):
                    if previous[key] != report[key]:
                        raise ValueError(f"resume conditions changed: {key}")
                report["model_services"] = previous.get("model_services", [previous["model_service"]]) + [ready]
                previous_rows = {str(row["episode_id"]): row for row in previous["episodes"]}
            current_service = len(report["model_services"]) - 1
            if args.batch_size > 1:
                save(report_path, report)
                try:
                    for result in run_batched_episodes(client, simulators, episodes, args.scenes, output):
                        report["episodes"].append({k: v for k, v in result.items() if k != "trace"})
                        report["successes"] = sum(row["success"] for row in report["episodes"])
                        report["sr"] = report["successes"] / len(report["episodes"])
                        report["spl"] = sum(row["spl"] for row in report["episodes"]) / len(report["episodes"])
                        save(report_path, report)
                        print(
                            f"{split} {len(report['episodes'])}/{len(episodes)} success={report['successes']}",
                            flush=True,
                        )
                except BaseException as error:
                    report.update(status="infrastructure_failure", error=f"{type(error).__name__}: {error}")
                    save(report_path, report)
                    raise
                report["status"] = "complete" if len(episodes) == args.episode_count else "partial_smoke"
                save(report_path, report)
                continue
            for episode in episodes:
                path = output / f"episode-{episode['episode_id']}.json"
                if args.resume and path.exists():
                    result = json.loads(path.read_text())
                    if result["status"] != "complete":
                        raise ValueError(
                            "resume refuses an unfinished episode; preserve diagnostics and retry explicitly"
                        )
                    result["model_service_index"] = previous_rows.get(str(episode["episode_id"]), {}).get(
                        "model_service_index", max(0, current_service - 1)
                    )
                else:
                    try:
                        result = run_episode(client, simulator, episode, args.scenes, output)
                    except BaseException as error:
                        report.update(status="infrastructure_failure", error=f"{type(error).__name__}: {error}")
                        save(report_path, report)
                        raise
                    result["model_service_index"] = current_service
                    save(path, result)
                report["episodes"].append({k: v for k, v in result.items() if k != "trace"})
                report["successes"] = sum(row["success"] for row in report["episodes"])
                report["sr"] = report["successes"] / len(report["episodes"])
                report["spl"] = sum(row["spl"] for row in report["episodes"]) / len(report["episodes"])
                save(report_path, report)
                print(
                    f"{split} {len(report['episodes'])}/{len(episodes)} episode={episode['episode_id']} success={result['success']} SR={report['sr']:.4f} SPL={report['spl']:.4f}",
                    flush=True,
                )
            report["status"] = "complete" if len(episodes) == args.episode_count else "partial_smoke"
            save(report_path, report)
    finally:
        for worker in simulators:
            worker.close()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        if model_log is not None:
            model_log.close()


if __name__ == "__main__":
    main()
