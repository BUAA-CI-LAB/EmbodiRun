"""Parallel evaluation with explicit rendering placement and HTTP replicas."""

from __future__ import annotations

import dataclasses
import math
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any
from urllib.parse import urlsplit

from .contracts import EpisodeRequest, EpisodeResult, SimulationServiceConfig
from .service import SimulationService


@dataclass(frozen=True)
class EvaluationConfig:
    """Place simulator workers separately from already-running inference services.

    Workers and their statically assigned episodes stay on one HTTP replica.
    All endpoints must serve identical weights and adapter contracts. An empty
    endpoint list uses the simulation service's existing endpoint. Rendering
    placement applies to MuJoCo EGL; None preserves the simulator's own setup.
    worker_timeout_s bounds startup and the time between completed episodes for
    each worker, including reset and cleanup, not just an individual HTTP call.
    """

    workers: int = 8
    inference_endpoints: tuple[str, ...] = ()
    render_gpu_id: int | None = None
    worker_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        if self.render_gpu_id is not None and (type(self.render_gpu_id) is not int or self.render_gpu_id < 0):
            raise ValueError("render_gpu_id must be a nonnegative integer or None")
        if (
            isinstance(self.worker_timeout_s, bool)
            or not isinstance(self.worker_timeout_s, (int, float))
            or not math.isfinite(self.worker_timeout_s)
            or self.worker_timeout_s <= 0
        ):
            raise ValueError("worker_timeout_s must be finite and positive")
        if not isinstance(self.inference_endpoints, (list, tuple)):
            raise ValueError("inference_endpoints must be a list or tuple of HTTP URLs")
        endpoints = tuple(self.inference_endpoints)
        for endpoint in endpoints:
            if not isinstance(endpoint, str) or endpoint != endpoint.strip():
                raise ValueError("inference_endpoints must contain HTTP URLs")
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("inference_endpoints must contain credential-free HTTP URLs")
            _ = parsed.port
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("inference_endpoints must be unique")
        object.__setattr__(self, "inference_endpoints", endpoints)


def _worker(
    service_config: SimulationServiceConfig,
    render_gpu_id: int | None,
    jobs: Sequence[tuple[int, EpisodeRequest]],
    connection: Connection,
    service_factory: Callable[[SimulationServiceConfig], SimulationService],
) -> None:
    try:
        if render_gpu_id is not None:
            os.environ.update(MUJOCO_EGL_DEVICE_ID=str(render_gpu_id), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        service = service_factory(service_config)
        try:
            for index, request in jobs:
                started = time.monotonic()
                result = service.execute(request)
                connection.send(("result", index, result.to_payload(), time.monotonic() - started))
        finally:
            service.close()
        connection.send(("done",))
    except BaseException:
        with suppress(BrokenPipeError, OSError):
            connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class EvaluationRunner:
    """Run finite simulation jobs on a fixed pool of process-isolated workers.

    Reuses SimulationService's episode/session lifecycle and original inference
    clients. There is no per-step coordinator, CPU quota, or inference batching
    implementation here. Failure aborts the run without retrying an episode on
    another replica. Results are returned in input order only after all workers
    close successfully. Custom service factories must be spawn-picklable.
    """

    def __init__(
        self,
        service: SimulationServiceConfig,
        config: EvaluationConfig,
        *,
        service_factory: Callable[[SimulationServiceConfig], SimulationService] = SimulationService,
    ) -> None:
        if service.inference_transport != "http":
            raise ValueError("parallel evaluation currently requires HTTP inference")
        if (
            config.render_gpu_id is not None
            and service_factory is SimulationService
            and service.simulator_kind not in {"libero", "vlabench"}
        ):
            raise ValueError("render_gpu_id currently supports the LIBERO and VLABench MuJoCo adapters")
        self.service, self.config, self._service_factory = service, config, service_factory

    def run(self, requests: Sequence[EpisodeRequest]) -> dict[str, Any]:
        """Run each request once; reject invalid identities before starting workers."""
        requests = list(requests)
        if any(request.runtime_id != self.service.runtime_id for request in requests):
            raise ValueError("all requests must target the configured runtime")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("episode request IDs must be unique")
        endpoints = self.config.inference_endpoints or (self.service.inference_endpoint,)
        count = min(self.config.workers, len(requests))
        ctx = mp.get_context("spawn")
        processes = []
        readers: dict[Connection, int] = {}
        activity: dict[int, float] = {}
        results: dict[int, dict[str, Any]] = {}
        completed = False
        started = time.monotonic()
        try:
            for worker_id in range(count):
                config = dataclasses.replace(
                    self.service,
                    simulator_id=f"{self.service.simulator_id}-{worker_id}",
                    inference_endpoint=endpoints[worker_id % len(endpoints)],
                )
                jobs = [(index, requests[index]) for index in range(worker_id, len(requests), count)]
                reader, writer = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=_worker,
                    args=(config, self.config.render_gpu_id, jobs, writer, self._service_factory),
                    name=f"embodirun-eval-{worker_id}",
                )
                try:
                    process.start()
                except BaseException:
                    reader.close()
                    raise
                finally:
                    writer.close()
                processes.append(process)
                readers[reader] = worker_id
                activity[worker_id] = time.monotonic()
            while readers:
                oldest = min(readers.values(), key=activity.__getitem__)
                remaining = self.config.worker_timeout_s - (time.monotonic() - activity[oldest])
                if remaining <= 0:
                    raise TimeoutError(f"evaluation worker {oldest} exceeded worker_timeout_s")
                for reader in wait(list(readers), timeout=remaining):
                    worker_id = readers[reader]
                    try:
                        message = reader.recv()
                    except EOFError:
                        raise RuntimeError(f"evaluation worker {worker_id} exited without completion") from None
                    activity[worker_id] = time.monotonic()
                    if message[0] == "error":
                        raise RuntimeError(f"evaluation worker {worker_id} failed:\n{message[1]}")
                    if message[0] == "done":
                        reader.close()
                        del readers[reader]
                    elif message[0] == "result":
                        _, index, payload, duration = message
                        if index in results or index not in range(worker_id, len(requests), count):
                            raise RuntimeError("unexpected or duplicate episode completion")
                        result = EpisodeResult.from_payload(payload)
                        if (result.request_id, result.runtime_id) != (
                            requests[index].request_id,
                            self.service.runtime_id,
                        ):
                            raise RuntimeError("episode result identity does not match request")
                        results[index] = {
                            **payload,
                            "worker": worker_id,
                            "inference_endpoint": endpoints[worker_id % len(endpoints)],
                            "wall_s": duration,
                        }
                    else:
                        raise RuntimeError("unexpected evaluation worker message")
            if len(results) != len(requests):
                raise RuntimeError("evaluation completed with missing results")
            completed = True
        finally:
            for process in processes:
                if not completed and process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            for reader in readers:
                reader.close()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("evaluation worker failed during cleanup")
        return {
            "results": [results[index] for index in range(len(requests))],
            "wall_s": time.monotonic() - started,
            "evaluation": dataclasses.asdict(self.config),
        }
