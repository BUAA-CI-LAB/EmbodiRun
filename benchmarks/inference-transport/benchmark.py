"""Compare HTTP and WirelessComm from the actual SO101 control nodes.

Workers replay immutable observations. They never construct a robot or camera
adapter. SSH only distributes inputs and coordinates the measurement barrier.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import select
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "embodirun.so101.transport-benchmark.v1"
IMAGE_FIELDS = ("observation.images.front", "observation.images.wrist")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_observations(path):
    """Preload encoded images once, preserving image bytes and sample order."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError("each observation needs a non-empty instruction")
        state = row["state"]
        joints, gripper = state["joint_positions_deg"], state["gripper_position"]
        if not isinstance(joints, list) or len(joints) != 5:
            raise ValueError("SO101 observations require five joint positions")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in [*joints, gripper]
        ):
            raise ValueError("SO101 joint and gripper values must be finite numbers")
        if set(row["images"]) != set(IMAGE_FIELDS):
            raise ValueError(f"images must contain exactly {IMAGE_FIELDS}")
        images = []
        for name in IMAGE_FIELDS:
            image_path = path.parent / row["images"][name]
            data = image_path.read_bytes()
            if data.startswith(b"\xff\xd8\xff"):
                mime_type = "image/jpeg"
            elif data.startswith(b"\x89PNG\r\n\x1a\n"):
                mime_type = "image/png"
            else:
                raise ValueError(f"not an encoded JPEG or PNG: {image_path}")
            images.append({"name": name, "mime_type": mime_type, "data": base64.b64encode(data).decode("ascii")})
        rows.append(
            {
                "instruction": row["instruction"],
                "state": state,
                "images": images,
                "source": row.get("source", "user-supplied"),
            }
        )
    if not rows:
        raise ValueError("observation manifest is empty")
    return rows


def distribution(values):
    if not values:
        return None
    values = sorted(values)

    def percentile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"mean": statistics.mean(values), "p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


def measure(client, mapper, session, row, frames, *, step_id, request_id, expected_actions):
    from embodirun.robots import RobotObservation

    start = time.perf_counter_ns()
    request = mapper.map_observation(
        RobotObservation(time.time(), row["state"]),
        session_id=session.session_id,
        request_id=request_id,
        step_id=step_id,
        instruction=row["instruction"],
        frames=frames,
    )
    rpc_start = time.perf_counter_ns()
    result = client.step(request)
    rpc_end = time.perf_counter_ns()
    actions = mapper.map_result(result)
    if len(actions) != expected_actions:
        raise ValueError(f"expected {expected_actions} actions, received {len(actions)}")
    end = time.perf_counter_ns()
    return {
        "e2e_ms": (end - start) / 1e6,
        "rpc_ms": (rpc_end - rpc_start) / 1e6,
        "action_rows": len(actions),
        "server_timing_ms": dict(result.timing),
        "policy_revision": result.policy_revision,
    }


def worker(payload, emit, wait_start, wait_shutdown=None):
    from embodirun.bindings.lerobot.so101.pi05 import Pi05SO101Mapper
    from embodirun.robots.sensors.cameras import CameraFrame
    from embodirun.services.inference.factory import build_inference_client

    rows = payload["observations"]
    frames = [
        tuple(
            CameraFrame(image["name"], image["mime_type"], base64.b64decode(image["data"], validate=True))
            for image in row["images"]
        )
        for row in rows
    ]
    mapper = Pi05SO101Mapper()
    inference = payload["inference"]
    # The measured path is the device LAN for both protocols, never an HTTP proxy.
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    with tempfile.TemporaryDirectory(prefix="so101-benchmark-") as directory:
        options = dict(inference["options"])
        if payload["wireless_config"] is not None:
            path = Path(directory) / "client.json"
            path.write_text(canonical(payload["wireless_config"]))
            options["comm_config"] = str(path)
        client = build_inference_client(
            inference["transport"],
            inference["endpoint"],
            options,
            backend=inference["backend"],
            timeout_s=payload["timeout_s"],
        )
        session = None
        report = None
        try:
            capabilities = client.capabilities()
            session = client.open_session(robot_id=payload["robot_id"], action_space=mapper.policy_action_space)
            run_id = uuid.uuid4().hex
            for index in range(payload["warmup"]):
                sample = index % len(rows)
                measure(
                    client,
                    mapper,
                    session,
                    rows[sample],
                    frames[sample],
                    step_id=index,
                    request_id=f"{run_id}-warmup-{index}",
                    expected_actions=payload["expected_actions"],
                )
            emit(
                {
                    "event": "ready",
                    "runtime": payload["runtime"],
                    "hostname": platform.node(),
                    "python": platform.python_version(),
                    "dataset_sha256": digest(rows),
                    "capabilities": capabilities,
                }
            )
            wait_start()
            samples = []
            start = time.perf_counter()
            for index in range(payload["requests"]):
                sample = index % len(rows)
                attempt_start = time.perf_counter()
                try:
                    result = measure(
                        client,
                        mapper,
                        session,
                        rows[sample],
                        frames[sample],
                        step_id=payload["warmup"] + index,
                        request_id=f"{run_id}-measure-{index}",
                        expected_actions=payload["expected_actions"],
                    )
                    samples.append({"index": index, "sample": sample, "status": "ok", **result})
                except Exception as error:
                    samples.append(
                        {
                            "index": index,
                            "sample": sample,
                            "status": "error",
                            "elapsed_ms": (time.perf_counter() - attempt_start) * 1000,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    # A timed-out request may already have advanced the server session.
                    # Stop instead of retrying or discarding failed attempts.
                    break
            elapsed = time.perf_counter() - start
            successful = [row for row in samples if row["status"] == "ok"]
            emit({"event": "done", "runtime": payload["runtime"], "elapsed_s": elapsed, "completed": len(successful)})
            report = {
                "event": "report",
                "runtime": payload["runtime"],
                "status": "ok" if len(successful) == payload["requests"] else "failed",
                "attempted": len(samples),
                "completed": len(successful),
                "failed": len(samples) - len(successful),
                "elapsed_s": elapsed,
                "calls_per_s": len(successful) / elapsed,
                "e2e_ms": distribution([row["e2e_ms"] for row in successful]),
                "rpc_ms": distribution([row["rpc_ms"] for row in successful]),
                "encoded_image_bytes": [sum(len(frame.data) for frame in item) for item in frames],
                "samples": samples,
            }
        finally:
            try:
                if session is not None:
                    try:
                        client.close(session.session_id)
                    except Exception as error:
                        if report is None:
                            raise
                        report["status"] = "failed"
                        report["cleanup_error"] = f"{type(error).__name__}: {error}"
                if report is not None:
                    emit(report)
                    if wait_shutdown is not None:
                        wait_shutdown()
            finally:
                shutdown = getattr(client, "shutdown", None)
                if shutdown is not None:
                    shutdown()


def worker_main():
    def emit(value):
        print(canonical(value), flush=True)

    def wait_start():
        if json.loads(sys.stdin.readline()) != {"event": "start"}:
            raise ValueError("coordinator did not send the start signal")

    def wait_shutdown():
        if json.loads(sys.stdin.readline()) != {"event": "shutdown"}:
            raise ValueError("coordinator did not send the shutdown signal")

    try:
        worker(json.loads(sys.stdin.readline()), emit, wait_start, wait_shutdown)
    except Exception as error:
        emit({"event": "error", "error": f"{type(error).__name__}: {error}"})
        return 1
    return 0


def read_event(process, expected, timeout):
    deadline = time.monotonic() + timeout
    pending = getattr(process, "_benchmark_pending", b"")
    while b"\n" not in pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
            raise TimeoutError(f"worker timed out before {expected}")
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            raise RuntimeError(f"worker exited before {expected}; inspect its stderr log")
        pending += chunk
    line, process._benchmark_pending = pending.split(b"\n", 1)
    event = json.loads(line)
    if event.get("event") != expected:
        raise RuntimeError(f"expected {expected}, received {event}")
    return event


def ssh_command(connection, python, source, extra_options):
    argv = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connection.connect_timeout_s:g}",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=" + ("accept-new" if connection.accept_new_host_key else "yes"),
        "-p",
        str(connection.port),
    ]
    if connection.identity_file:
        argv += ["-i", os.path.expanduser(connection.identity_file)]
    for option in extra_options:
        argv += ["-o", option]
    if connection.proxy_command:
        argv += ["-o", "ProxyCommand=" + connection.proxy_command]
    # shlex protects the remote shell; observations travel over stdin, not argv.
    return [*argv, f"{connection.username}@{connection.host}", shlex.join([python, "-u", "-c", source, "worker"])]


def run(args):
    from embodirun.services.host.cli.context import state_path
    from embodirun.services.host.config import config_digest, load_config
    from embodirun.services.host.plan import build_plan
    from embodirun.services.host.state import StateStore

    config = load_config(args.config)
    plan = build_plan(config)
    state = StateStore(state_path(args.state_dir, plan.name)).load()
    if state is None or state.config_digest != config_digest(config):
        raise ValueError("initialize this exact configuration with embodirun init first")
    runtimes = [runtime for runtime in plan.runtimes if not args.runtime or runtime.runtime_id in args.runtime]
    if not runtimes or (args.runtime and set(args.runtime) != {r.runtime_id for r in runtimes}):
        raise ValueError("unknown or empty runtime selection")
    if any(r.binding != "lerobot.so101.pi05" for r in runtimes):
        raise ValueError("this benchmark requires SO101/PI0.5 runtimes")
    if len({r.model for r in runtimes}) != 1:
        raise ValueError("clients must share one Thor inference service")
    model = config.models[runtimes[0].model]
    if model.backend != "vvla":
        raise ValueError("this comparison requires the VVLA backend")
    if state.services[model.model_id].status != "running":
        raise ValueError("start the model service with embodirun up first")
    rows = load_observations(args.observations)
    source = Path(__file__).read_text()
    services = {s.service_id: s for s in plan.services}
    jobs = []
    for runtime in runtimes:
        if state.services[runtime.service_id].status != "stopped":
            raise ValueError("stop control services with embodirun down --target control first")
        node = config.nodes[runtime.node]
        environment = state.environments[runtime.environment_id]
        if environment.status != "ready":
            raise ValueError(f"client environment is not ready: {runtime.runtime_id}")
        if node.connection.kind != "ssh":
            raise ValueError("workers must execute on the configured control nodes via SSH")
        service = services[runtime.service_id]
        control = json.loads(service.control_config_json)
        payload = {
            "runtime": runtime.runtime_id,
            "robot_id": runtime.target_id,
            "inference": control["inference"],
            "observations": rows,
            "wireless_config": json.loads(service.wireless_config_json) if service.wireless_config_json else None,
            "warmup": args.warmup,
            "requests": args.requests,
            "timeout_s": args.timeout,
            "expected_actions": 50,
        }
        argv = ssh_command(node.connection, environment.path + "/bin/python", source, args.ssh_option)
        jobs.append((runtime, payload, argv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    comparable_model = {
        key: value for key, value in model.options.items() if key not in {"transport", "transport_options", "server"}
    }
    comparison = {
        "dataset_sha256": digest(rows),
        "requests_per_client": args.requests,
        "warmup_per_client": args.warmup,
        "timeout_s": args.timeout,
        "model": comparable_model,
        "deploy_commit": state.deploy_commit,
        "inference_commit": state.inference_commit,
        "benchmark_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "clients": {r.runtime_id: config.nodes[r.node].connection.host for r in runtimes},
    }
    report = {
        "schema": SCHEMA,
        "transport": model.transport,
        "status": "failed",
        "comparison": comparison,
        "dataset_samples": len(rows),
        "configuration_sha256": config_digest(config),
        "transport_options": model.transport_options,
        "ready": [],
        "clients": [],
        "errors": [],
    }
    processes, logs = [], []
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    cancelling = threading.Event()
    try:

        def launch(job):
            runtime, payload, argv = job
            log = args.output.with_name(args.output.stem + "." + runtime.runtime_id + ".stderr.log").open("x")
            logs.append(log)
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            processes.append(process)
            if cancelling.is_set():
                process.terminate()
                raise RuntimeError("benchmark cancelled")
            process.stdin.write((canonical(payload) + "\n").encode())
            process.stdin.flush()
            return process, read_event(process, "ready", (args.warmup + 2) * args.timeout + 30)

        launched = list(pool.map(launch, jobs))
        report["ready"] = [ready for _, ready in launched]
        if any(ready["dataset_sha256"] != comparison["dataset_sha256"] for _, ready in launched):
            raise ValueError("worker inputs differ from coordinator inputs")
        # Both remote workers have already loaded inputs and finished warmup.
        start = time.perf_counter()
        for process, _ in launched:
            process.stdin.write(b'{"event":"start"}\n')
            process.stdin.flush()

        def collect(item):
            process, _ = item
            read_event(process, "done", args.requests * args.timeout + 30)
            finished = time.perf_counter()
            result = read_event(process, "report", args.timeout + 10)
            return finished, result

        collected = list(pool.map(collect, launched))
        elapsed = max(finished for finished, _ in collected) - start
        report["clients"] = [result for _, result in collected]
        report["coordinator_elapsed_s"] = elapsed
        report["aggregate_calls_per_s"] = sum(r["completed"] for r in report["clients"]) / elapsed
        report["status"] = "ok" if all(r["status"] == "ok" for r in report["clients"]) else "failed"
        # Keep transports alive until every client has closed its server session.
        # A peer disconnect must not race another client's final RPC.
        for process, _ in launched:
            process.stdin.write(b'{"event":"shutdown"}\n')
            process.stdin.flush()
            process.stdin.close()
        for process, _ in launched:
            if process.wait(timeout=args.timeout + 10) != 0:
                report["status"] = "failed"
                error = read_event(process, "error", 5)
                report["errors"].append(error["error"])
    except BaseException as error:
        report["status"] = "failed"
        report["errors"].append(f"{type(error).__name__}: {error}")
        if not isinstance(error, Exception):
            raise
    finally:
        cancelling.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin and not process.stdin.closed:
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.close()
        pool.shutdown(wait=True, cancel_futures=True)
        for process in processes:
            process.stdout.close()
        for log in logs:
            log.close()
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "ok" else 1


def compare(args):
    reports = [json.loads(path.read_text()) for path in (args.http, args.wireless)]
    if [r.get("transport") for r in reports] != ["http", "wireless"]:
        raise ValueError("provide HTTP and WirelessComm reports in the matching arguments")
    if any(r.get("schema") != SCHEMA or r.get("status") != "ok" for r in reports):
        raise ValueError("both reports must be complete, successful benchmark runs")
    if reports[0]["comparison"] != reports[1]["comparison"]:
        raise ValueError("inputs, clients, revisions, model settings or request counts differ")
    print("| Protocol | Client | Calls | Mean ms | P50 ms | P95 ms | P99 ms | Calls/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for report in reports:
        for client in report["clients"]:
            latency = client["e2e_ms"]
            print(
                f"| {report['transport']} | {client['runtime']} | {client['completed']} | "
                f"{latency['mean']:.2f} | {latency['p50']:.2f} | {latency['p95']:.2f} | "
                f"{latency['p99']:.2f} | {client['calls_per_s']:.3f} |"
            )
    for report in reports:
        print(f"\n{report['transport']} aggregate: {report['aggregate_calls_per_s']:.3f} calls/s")
    return 0


def fixture(args):
    """Create reproducible synthetic PNGs for transport smoke testing only."""
    args.output.mkdir(parents=True, exist_ok=False)

    def png(seed):
        rng = random.Random(seed)
        pixels = b"".join(b"\0" + rng.randbytes(640 * 3) for _ in range(480))

        def chunk(kind, body):
            return struct.pack("!I", len(body)) + kind + body + struct.pack("!I", zlib.crc32(kind + body))

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!2I5B", 640, 480, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels))
            + chunk(b"IEND", b"")
        )

    for index, name in enumerate(("front.png", "wrist.png")):
        (args.output / name).write_bytes(png(index))
    row = {
        "instruction": "Pick up the cube.",
        "source": "synthetic transport fixture, not a public dataset",
        "state": {"joint_positions_deg": [0.0] * 5, "gripper_position": 0.0},
        "images": dict(zip(IMAGE_FIELDS, ("front.png", "wrist.png"))),
    }
    path = args.output / "observations.jsonl"
    path.write_text(canonical(row) + "\n")
    print(path)
    return 0


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("worker", help="internal worker using the coordinator's stdin protocol")
    p = commands.add_parser("run", help="run synchronized replay clients on the configured Orin nodes")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--state-dir", type=Path)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--runtime", action="append", help="default: all configured runtimes")
    p.add_argument("--requests", type=positive, default=200)
    p.add_argument("--warmup", type=positive, default=20)
    p.add_argument("--timeout", type=positive, default=60)
    p.add_argument("--ssh-option", action="append", default=[], help="OpenSSH -o option, repeatable")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("compare", help="compare compatible successful reports")
    p.add_argument("--http", type=Path, required=True)
    p.add_argument("--wireless", type=Path, required=True)
    p = commands.add_parser("fixture", help="create synthetic observations for smoke testing")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return {"run": run, "compare": compare, "fixture": fixture, "worker": lambda _: worker_main()}[args.command](
            args
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
