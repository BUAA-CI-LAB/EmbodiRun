#!/usr/bin/env python3
"""MicroDuck VLN example using the Deploy client and a managed inference service."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from embodirun_microduck.assets import md5, verify_manifest, write_manifest


def atomic_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "third_party/embodiinfer",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--episodes", type=Path, default=None, help="JSONL episode file; defaults to the two fixed-spawn demos"
    )
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=60, help="Primitive-action cap, including STOP (not inference turns)"
    )
    parser.add_argument("--max-action-physics-steps", type=int, default=1200)
    parser.add_argument("--success-radius", type=float, default=1.0)
    parser.add_argument(
        "--success-hold-steps", type=int, default=3, help="Consecutive action endpoints in radius, including STOP"
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--sample", action="store_true", help="Use seeded temperature=.2/top_p=.8 sampling; default is greedy"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, choices=[5, 10, 20], default=10)
    parser.add_argument(
        "--output", type=Path, default=None, help="A NEW output directory; existing directories are rejected"
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--write-manifest", action="store_true", help="Provision trusted asset MD5 inventory, without loading a GPU"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Verify assets, package versions, CUDA, EGL, ONNX, MPC and encoder"
    )
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.checkpoint = (args.checkpoint or args.project_root / "lightnav_sft_v3/merged").resolve()
    args.episodes = (args.episodes or args.project_root / "data/demo_microduck_vln.jsonl").resolve()
    args.manifest = (args.manifest or Path(__file__).resolve().parent / "assets.md5.json").resolve()
    for name in (
        "num_episodes",
        "max_steps",
        "max_action_physics_steps",
        "success_hold_steps",
        "max_new_tokens",
        "max_context",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.episode_offset < 0 or not math.isfinite(args.success_radius) or args.success_radius <= 0:
        parser.error("episode offset must be nonnegative and success radius positive/finite")
    if any(not math.isfinite(v) or v <= 0 for v in (args.startup_timeout, args.request_timeout)):
        parser.error("Timeouts must be positive and finite")
    return args


def load_episodes(args):
    records = []
    for line in args.episodes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records.extend(record.get("episodes", [record]))
    end = args.episode_offset + args.num_episodes
    if end > len(records):
        raise ValueError(f"Requested episodes [{args.episode_offset}:{end}], file contains {len(records)}")
    selected = records[args.episode_offset : end]
    for record in selected:
        if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
            raise ValueError("Every episode needs a nonempty instruction")
        for name, length in [("start", 3), ("goal", 2)]:
            if len(record[name]) != length or not all(math.isfinite(float(v)) for v in record[name]):
                raise ValueError(f"Invalid episode {name}: {record[name]}")
        if int(record.get("max_steps", 60)) <= 0:
            raise ValueError("Episode max_steps must be positive")
    return selected


def preflight(args, output, roots):
    import imageio_ffmpeg
    import torch
    import transformers

    from embodirun_microduck.runtime import DemoSimulation, EpisodeVideo

    if transformers.__version__ != "4.51.3":
        raise RuntimeError(
            f"Expected transformers 4.51.3, got {transformers.__version__}; follow examples/microduck_vln/README.md"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run on a GPU node or set parameters.slurm: auto in the example YAML")
    logging.info("Verifying all asset MD5 checksums (including checkpoint shards)")
    checked = verify_manifest(args.manifest, roots)
    checked.update(
        {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": platform.node(),
            "python": sys.version,
            "executable": sys.executable,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                    "mujoco",
                    "onnxruntime",
                    "casadi",
                    "imageio-ffmpeg",
                )
            },
            "episodes_md5": md5(args.episodes),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-version"], check=True, stdout=subprocess.DEVNULL)
    sim = DemoSimulation(args.project_root, max_physics_steps=args.max_action_physics_steps, fps=args.fps)
    try:
        sim.reset([6.5, 13.8, 0.0])
        start = time.perf_counter()
        frame = sim.render()
        checked["first_render_ms"] = (time.perf_counter() - start) * 1000
        from PIL import Image

        Image.fromarray(frame).save(output / "preflight_first_person.jpg")
        Image.fromarray(sim.render(third=True)).save(output / "preflight_third_person.jpg")
        start = time.perf_counter()
        for _ in range(200):
            sim._step((0.0, 0.0))
        checked["physics_200_steps_wall_s"] = time.perf_counter() - start
        checked["physics_realtime_factor"] = 200 * sim.dt / checked["physics_200_steps_wall_s"]
        sim.controller.solve(__import__("numpy").zeros(3), __import__("numpy").tile([0.1, 0, 0], (5, 1)), (0, 0), 0.30)
        video = EpisodeVideo(output / "preflight_encoder.mp4", args.fps, ffmpeg)
        try:
            canvas = video.compose(
                frame,
                sim.render(True),
                instruction="GPU / EGL / MPC / encoder preflight",
                episode_id="preflight",
                action="idle",
                step=0,
                distance=0,
                sim_time=1,
                inference_ms=0,
            )
            video.write(canvas, repeats=args.fps)
        finally:
            video.close()
        checked["checks"] = [
            "asset_md5",
            "cuda",
            "egl_first_person",
            "egl_third_person",
            "onnx_physics",
            "casadi_ipopt",
            "ffmpeg_h264",
        ]
        checked["ffmpeg"] = ffmpeg
    finally:
        sim.close()
    atomic_json(output / "preflight.json", checked)
    logging.info("Preflight passed: %s; 200 physics steps %.3fs", checked["gpu"], checked["physics_200_steps_wall_s"])
    return checked


def run_episode(args, sim, client, record, index, output, ffmpeg):
    import io
    import uuid

    from PIL import Image

    from embodirun.model_services.contracts import ImagePayload, PolicyObservation
    from embodirun_microduck.protocol import ACTION_SPACE, IMAGE_FIELD, action_text, decode_actions
    from embodirun_microduck.runtime import EpisodeVideo

    episode_id = str(record.get("id", f"ep{index:03d}"))
    # Numeric filenames avoid trusting dataset IDs as paths.
    prefix = f"episode_{index:03d}"
    session = None
    max_steps = min(args.max_steps, int(record.get("max_steps", 60)))
    result = {
        "episode_id": episode_id,
        "instruction": record["instruction"],
        "start": record["start"],
        "goal": record["goal"],
        "steps": 0,
        "turns": 0,
        "success": False,
        "stop_seen": False,
        "termination": None,
        "trace": [],
        "status": "running",
        "max_steps": max_steps,
    }
    hud = {"action": "initial observation", "step": 0, "inference_ms": 0}
    video = None
    start_s = time.perf_counter()
    hold = 0
    try:
        session = client.open_session(robot_id="microduck-sim", action_space=ACTION_SPACE)
        sim.reset(record["start"])
        if not args.no_video:
            video = EpisodeVideo(output / f"{prefix}.mp4", args.fps, ffmpeg)

            def canvas(banner=None):
                return video.compose(
                    sim.render(),
                    sim.render(True),
                    instruction=record["instruction"],
                    episode_id=episode_id,
                    distance=sim.distance(record["goal"]),
                    sim_time=float(sim.robot.data.time),
                    banner=banner,
                    **hud,
                )

            intro = canvas(
                [
                    "MicroDuck VLN / SFT-v3 / EmbodiRun HTTP + ActiveVLN + MPC",
                    "First-person observations, model actions, and MPC execution.",
                    "Playback follows simulation time; inference pauses are omitted.",
                ]
            )
            video.write(intro, repeats=2 * args.fps)
            intro.save(output / f"{prefix}_preview.jpg", quality=85)
            sim.callback = lambda: video.write(canvas())
        while result["steps"] < max_steps and not result["stop_seen"]:
            frame = sim.render()
            encoded = io.BytesIO()
            Image.fromarray(frame).save(encoded, format="PNG")
            observation = PolicyObservation(
                session_id=session.session_id,
                request_id=uuid.uuid4().hex,
                step_id=result["turns"],
                instruction=record["instruction"],
                state={},
                images=(ImagePayload(IMAGE_FIELD, "image/png", encoded.getvalue()),),
            )
            requested = time.perf_counter()
            response = client.step(observation)
            if (
                response.action_space != ACTION_SPACE
                or len(response.actions) != 1
                or response.actions[0].kind != "discrete_chunk"
            ):
                raise ValueError("Inference returned an incompatible action contract")
            values = response.actions[0].values
            latency_ms = float(response.timing.get("policy_ms", 0))
            trace = {
                "turn": result["turns"],
                "request_id": response.request_id,
                "session_id": response.session_id,
                "session_revision": response.session_revision,
                "raw_text": values.get("raw_text"),
                "stop_reason": values.get("stop_reason"),
                "action_rows": values.get("rows"),
                "latency_ms": latency_ms,
                "http_roundtrip_ms": (time.perf_counter() - requested) * 1000,
                "timing": dict(response.timing),
                "actions": [],
            }
            result["trace"].append(trace)
            result["turns"] += 1
            if values.get("valid") is not True:
                result["termination"] = "invalid_model_output"
                break
            try:
                actions = decode_actions(values.get("rows"))
            except ValueError as exc:
                result["termination"] = "invalid_model_output"
                trace["validation_error"] = str(exc)
                break
            for action in actions:
                if result["steps"] >= max_steps:
                    break
                hud.update(action=action_text(action), step=result["steps"] + 1, inference_ms=latency_ms)
                details = sim.execute(action)
                result["steps"] += 1
                distance = sim.distance(record["goal"])
                hold = hold + 1 if distance < args.success_radius else 0
                trace["actions"].append({"command": action_text(action), "distance": distance, **details})
                if action[0] == 0:
                    result["stop_seen"] = True
                    result["success"] = distance < args.success_radius and hold >= args.success_hold_steps
                    result["termination"] = "success" if result["success"] else "stop_outside_success_condition"
                    break
            logging.info(
                "%s turn=%d steps=%d distance=%.2fm infer=%.0fms text=%s",
                episode_id,
                result["turns"],
                result["steps"],
                sim.distance(record["goal"]),
                latency_ms,
                trace["raw_text"],
            )
            atomic_json(output / f"{prefix}.json", result)
        if result["termination"] is None:
            result["termination"] = "max_steps"
        result.update(
            status="complete",
            distance=sim.distance(record["goal"]),
            pose=list(sim.pose()),
            path_length_m=sim.path_length,
            simulation_s=float(sim.robot.data.time),
            physics_steps=sim.physics_steps,
            mpc_solve_count=len(sim.mpc_ms),
            mpc_mean_ms=sum(sim.mpc_ms) / max(1, len(sim.mpc_ms)),
        )
        shortest = math.dist(record["start"][:2], record["goal"])
        result["spl_euclidean"] = (shortest / max(shortest, sim.path_length, 1e-9)) if result["success"] else 0.0
        if video:
            video.write(
                canvas(
                    [
                        f"Episode finished: {result['termination']}",
                        f"Success: {result['success']} | distance: {result['distance']:.2f} m",
                        f"Actions: {result['steps']} | inference turns: {result['turns']}",
                    ]
                ),
                repeats=args.fps,
            )
            result["video"] = str(video.path.name)
            result["video_frames"] = video.frames
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        primary_error = sys.exc_info()[1]
        sim.callback = None
        result["session_cleared"] = False
        if session is not None:
            try:
                client.close(session.session_id)
                result["session_cleared"] = True
            except Exception as exc:
                result.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        result["wall_s"] = time.perf_counter() - start_s
        try:
            if video:
                video.close()
        except BaseException as exc:
            result.update(status="failed", video_cleanup_error=f"{type(exc).__name__}: {exc}")
            if primary_error is None:
                raise
            logging.exception("Video cleanup also failed; preserving the episode error")
        finally:
            atomic_json(output / f"{prefix}.json", result)
    if result["status"] == "failed":
        raise RuntimeError(result.get("cleanup_error", result.get("error", "Episode failed")))
    return result


def main():
    args = parse_args()
    roots = {"project": args.project_root, "embodiinfer": args.inference_root, "checkpoint": args.checkpoint}
    if args.write_manifest:
        print(f"Wrote {write_manifest(args.manifest, roots)} asset checksums to {args.manifest}", flush=True)
        return 0
    # The shared examples runner manages Slurm and the selected environment.
    for key, value in {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    for path in reversed([args.project_root / "src", args.project_root / "src/sim"]):
        sys.path.insert(0, str(path))
    records = load_episodes(args)
    output = (
        args.output
        or Path(__file__).resolve().parents[2]
        / "artifacts/microduck_vln"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")],
    )
    logging.info("OUTPUT=%s", output)
    started = time.perf_counter()
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "protocol": {
            "controller": "existing CasADi MPC; synchronous 10 Hz; v_max=.30, w_max=1.5",
            "success": "STOP and last success_hold_steps action endpoints strictly inside success_radius",
            "max_steps_unit": "primitive action, including STOP",
            "spl": "Euclidean approximation; no geodesic ground truth; not standard benchmark SPL",
            "history": "HTTP session per episode; DELETE clears recurrent state; no periodic context reset",
        },
        "records": [],
    }
    atomic_json(output / "results.json", report)
    sim = None

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        check = preflight(args, output, roots)
        report["preflight"] = check
        if args.check_only:
            report["status"] = "preflight_passed"
        else:
            import numpy as np

            from embodirun_microduck.process import managed_service
            from embodirun_microduck.runtime import DemoSimulation

            sim = DemoSimulation(args.project_root, max_physics_steps=args.max_action_physics_steps, fps=args.fps)
            loading = time.perf_counter()
            with managed_service(args, output) as client:
                report["model_load_s"] = time.perf_counter() - loading
                report["service_capabilities"] = client.capabilities()
                for index, record in enumerate(records, start=args.episode_offset):
                    result = run_episode(args, sim, client, record, index, output, check["ffmpeg"])
                    report["records"].append(result)
                    atomic_json(output / "results.json", report)
            latencies = [turn["latency_ms"] for record in report["records"] for turn in record["trace"]]
            report["summary"] = {
                "episodes": len(records),
                "sr": float(np.mean([r["success"] for r in report["records"]])),
                "spl_euclidean": float(np.mean([r["spl_euclidean"] for r in report["records"]])),
                "mean_distance_m": float(np.mean([r["distance"] for r in report["records"]])),
                "inference_mean_ms": float(np.mean(latencies)),
                "inference_p50_ms": float(np.percentile(latencies, 50)),
                "inference_p95_ms": float(np.percentile(latencies, 95)),
                "http_roundtrip_mean_ms": float(
                    np.mean([t["http_roundtrip_ms"] for r in report["records"] for t in r["trace"]])
                ),
            }
            report["status"] = "complete"
        # Recheck the protected training adapter after each run.
        report["training_adapter_md5_after"] = md5(args.project_root / "src/rlinf_integration_lightnav_env.py")
        original = json.loads(args.manifest.read_text())["files"]["project/src/rlinf_integration_lightnav_env.py"][
            "md5"
        ]
        if report["training_adapter_md5_after"] != original:
            raise RuntimeError("Protected training adapter changed during the run")
        logging.info("DONE status=%s summary=%s", report["status"], report.get("summary"))
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        logging.exception("Demo failed; diagnostic results preserved at %s", output)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if sim is not None:
            sim.close()
        report["wall_s"] = time.perf_counter() - started
        atomic_json(output / "results.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
