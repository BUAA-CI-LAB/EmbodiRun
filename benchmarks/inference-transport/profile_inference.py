"""Attribute replay latency using existing reports or an instrumented PI0.5 server.

The server wraps one adapter instance, preserving its lock and CUDA event timers.
It starts one transport per stdin command and keeps the model loaded between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path


class AdapterProfile:
    """Measure non-overlapping adapter stages without adding CUDA synchronizations."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.local = threading.local()
        self.originals = []
        profiler = self
        original_lock = adapter._lock

        class Lock:
            def __enter__(self):
                start = time.perf_counter_ns()
                original_lock.acquire()
                profiler.local.record["profile_lock_wait_ms"] = (time.perf_counter_ns() - start) / 1e6

            def __exit__(self, *args):
                original_lock.release()

        self.replace(adapter, "_lock", Lock())
        for owner, method, key in (
            (adapter, "_state_vector", "profile_state_ms"),
            (adapter, "_image_tensor_stack", "profile_images_ms"),
            (adapter._processor, "prepare", "profile_prepare_ms"),
            (adapter._core, "execute", "profile_engine_wall_ms"),
            (adapter._processor, "restore_actions", "profile_restore_ms"),
        ):
            self.wrap(owner, method, key)
        original_infer = adapter.infer

        @functools.wraps(original_infer)
        def infer(request):
            record = self.local.record = {}
            start = time.perf_counter_ns()
            try:
                result = original_infer(request)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                exclusive = sum(
                    record[key]
                    for key in (
                        "profile_lock_wait_ms",
                        "profile_state_ms",
                        "profile_images_ms",
                        "profile_prepare_ms",
                        "profile_engine_wall_ms",
                        "profile_restore_ms",
                    )
                )
                record["profile_adapter_other_ms"] = elapsed - exclusive
                record["profile_adapter_wall_ms"] = elapsed
                return replace(result, timing={**result.timing, **record})
            finally:
                del self.local.record

        self.replace(adapter, "infer", infer)

    def replace(self, owner, name, value):
        self.originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def wrap(self, owner, name, key):
        original = getattr(owner, name)

        @functools.wraps(original)
        def call(*args, **kwargs):
            start = time.perf_counter_ns()
            result = original(*args, **kwargs)
            self.local.record[key] = (time.perf_counter_ns() - start) / 1e6
            if name == "execute":
                if len(result) != 1:
                    raise ValueError("PI0.5 profiling requires B=1")
                for stage in ("prefill_ms", "decode_ms", "e2e_ms"):
                    self.local.record["profile_core_" + stage] = float(result[0].timing[stage])
            return result

        self.replace(owner, name, call)

    def close(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def event(kind, **values):
    print("PROFILE " + json.dumps({"event": kind, **values}, allow_nan=False), flush=True)


def serve(args):
    import os

    from vvla.engine.serve.factory import build_serving_adapter
    from vvla.engine.serve.http_server import PolicyHttpService, create_http_server
    from vvla.engine.serve.service import PolicyService
    from vvla.engine.serve.wireless_server import _run

    adapter = build_serving_adapter(args)
    profiler = AdapterProfile(adapter)
    active = None

    def stop():
        nonlocal active
        if active is None:
            return None
        protocol, server, thread, loop = active
        active = None
        failure = None
        if protocol == "http":
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
        else:

            async def finish():
                if not server.done():
                    server.cancel()
                result = (await asyncio.gather(server, return_exceptions=True))[0]
                if isinstance(result, Exception):
                    return f"{type(result).__name__}: {result}"
                return None

            failure = asyncio.run_coroutine_threadsafe(finish(), loop).result(timeout=15)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
            loop.close()
        if thread.is_alive():
            raise RuntimeError("transport thread did not stop")
        return failure

    try:
        event("loaded", pid=os.getpid(), action_space=adapter.action_space)
        for line in sys.stdin:
            command = json.loads(line)
            action = command["action"]
            if action == "stop":
                event("stopped", transport_error=stop())
            elif action == "exit":
                break
            elif action == "start":
                if active is not None:
                    raise ValueError("stop the previous transport before starting another")
                protocol = command["transport"]
                if protocol == "http":
                    server = create_http_server(
                        PolicyHttpService(adapter, token=None, max_body_bytes=64 * 1024 * 1024),
                        host=args.host,
                        port=args.port,
                    )
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    active = (protocol, server, thread, None)
                elif protocol == "wireless":
                    loop = asyncio.new_event_loop()
                    thread = threading.Thread(target=loop.run_forever, daemon=True)
                    thread.start()

                    async def start():
                        task = asyncio.create_task(_run(args, PolicyService(adapter)))
                        # _run reaches its first receive wait after binding its listener.
                        await asyncio.sleep(0.2)
                        if task.done():
                            await task
                        return task

                    server = asyncio.run_coroutine_threadsafe(start(), loop).result(timeout=15)
                    active = (protocol, server, thread, loop)
                else:
                    raise ValueError(f"unknown transport: {protocol}")
                event("started", transport=protocol, pid=os.getpid())
            else:
                raise ValueError(f"unknown action: {action}")
    finally:
        error = stop()
        profiler.close()
        event("exited", transport_error=error)


def report_summary(report):
    if report["status"] != "ok":
        raise ValueError("profiling requires a successful measurement report")
    rows = [row for client in report["clients"] for row in client["samples"]]
    if not rows or any(row["status"] != "ok" for row in rows):
        raise ValueError("report contains missing or failed requests")
    result = {
        "transport": report["transport"],
        "requests": len(rows),
        "clients": len(report["clients"]),
        "aggregate_calls_per_s": report["aggregate_calls_per_s"],
    }
    measures = {"e2e_ms": [], "client_mapping_ms": [], "rpc_outside_policy_ms": []}
    keys = set(rows[0]["server_timing_ms"])
    measures.update({key: [] for key in keys})
    for row in rows:
        if set(row["server_timing_ms"]) != keys:
            raise ValueError("inconsistent server timing fields")
        measures["e2e_ms"].append(row["e2e_ms"])
        measures["client_mapping_ms"].append(row["e2e_ms"] - row["rpc_ms"])
        measures["rpc_outside_policy_ms"].append(row["rpc_ms"] - row["server_timing_ms"]["policy_ms"])
        for key in keys:
            measures[key].append(row["server_timing_ms"][key])
    spec = importlib.util.spec_from_file_location("so101_benchmark", Path(__file__).with_name("benchmark.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    result["timing_ms"] = {key: bench.distribution(values) for key, values in sorted(measures.items())}
    return result


def analyze(args):
    summaries = []
    for path in args.reports:
        report = json.loads(path.read_text())
        summary = report_summary(report)
        summary["report"] = str(path)
        summary["comparison"] = report["comparison"]
        summaries.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump({"schema": "embodirun.so101.profile.v1", "reports": summaries}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for summary in summaries:
        print(summary["report"], json.dumps({key: value["mean"] for key, value in summary["timing_ms"].items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--reports", type=Path, nargs="+", required=True)
    analysis.add_argument("--output", type=Path, required=True)
    server = sub.add_parser("serve")
    server.add_argument("--checkpoint", required=True)
    server.add_argument("--adapter-config", required=True)
    server.add_argument("--comm-config", required=True)
    server.add_argument("--device", default="cuda:0")
    server.add_argument("--dtype", default="bfloat16")
    server.add_argument("--num-steps", type=int, default=10)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8000)
    server.set_defaults(
        policy="pi05",
        max_batch=1,
        no_cuda_graph=False,
        capture_full_loop=True,
        token=None,
        max_images=8,
        max_image_bytes=16 * 1024 * 1024,
        max_in_flight=64,
    )
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
