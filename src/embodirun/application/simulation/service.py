"""Application service for one bounded simulator policy runtime.

This module owns episode validation, simulator and inference lifecycle, and
wireless client reuse. HTTP parsing and listener lifetime remain in
``services.simulation.server``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from embodirun.bindings import binding_definition
from embodirun.model_services import build_inference_client
from embodirun.simulators import simulator_definition

from .contracts import EpisodeRequest, EpisodeResult, SimulationServiceConfig
from .runtime import SimulationRuntime


class SimulationServiceError(RuntimeError):
    """A configured simulation episode cannot be executed."""


class SimulationService:
    """Execute one episode at a time for exactly one configured runtime."""

    def __init__(
        self,
        config: SimulationServiceConfig,
        *,
        client_factory: Callable[[SimulationServiceConfig, float], Any] | None = None,
        runtime_factory: Callable[..., Any] = SimulationRuntime,
    ) -> None:
        self.config = config
        self.client_factory = client_factory or create_inference_client
        self.runtime_factory = runtime_factory
        self._episode_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._wireless_client: Any | None = None

    def health(self) -> dict[str, Any]:
        client = self._inference_client(2.0)
        try:
            health = client.health()
        finally:
            self._release_inference_client(client)
        if health.get("status") != "ok":
            raise SimulationServiceError(f"inference service at {self.config.inference_endpoint} is not healthy")
        return {"status": "ok", "runtime_id": self.config.runtime_id}

    def execute(self, request: EpisodeRequest) -> EpisodeResult:
        if request.runtime_id != self.config.runtime_id:
            raise SimulationServiceError(f"runtime {request.runtime_id!r} is not served here")
        if not self._episode_lock.acquire(blocking=False):
            raise SimulationServiceError("simulation service is already executing")
        try:
            return self._execute_locked(request)
        finally:
            self._episode_lock.release()

    def _execute_locked(self, request: EpisodeRequest) -> EpisodeResult:
        try:
            binding = binding_definition(self.config.binding_kind)
            definition = simulator_definition(self.config.simulator_kind)
        except (KeyError, TypeError) as error:
            raise SimulationServiceError(str(error)) from error
        if binding.robot_kind != definition.embodiment_kind:
            raise SimulationServiceError(
                f"binding {binding.kind!r} targets {binding.robot_kind!r}, not "
                f"simulator embodiment {definition.embodiment_kind!r}"
            )
        if request.chunk_steps > binding.maximum_chunk_steps:
            raise SimulationServiceError(f"chunk_steps exceeds binding maximum {binding.maximum_chunk_steps}")
        try:
            simulator_config = definition.config_factory(
                self.config.simulator_id,
                self.config.simulator_options,
            )
        except (TypeError, ValueError) as error:
            raise SimulationServiceError(
                f"simulator {self.config.simulator_id!r} configuration is invalid: {error}"
            ) from error

        client: Any | None = None
        simulator: Any | None = None
        runtime: Any | None = None
        try:
            client = self._inference_client(request.inference_timeout_s)
            simulator = definition.adapter_type(simulator_config)
            runtime = self.runtime_factory(
                simulator,
                client,
                mapper=binding.mapper_factory(),
                chunk_steps=request.chunk_steps,
            )
            health = client.health()
            if health.get("status") != "ok":
                raise SimulationServiceError("inference service is not healthy")
            outcome = runtime.run(
                instruction=request.prompt,
                max_policy_steps=request.max_policy_steps,
                task=request.task,
                seed=request.seed,
            )
        finally:
            try:
                if runtime is not None:
                    runtime.close()
            finally:
                try:
                    if simulator is not None:
                        simulator.close()
                finally:
                    if client is not None:
                        self._release_inference_client(client)
        return EpisodeResult(
            request_id=request.request_id,
            runtime_id=request.runtime_id,
            policy_steps=outcome.policy_steps,
            environment_steps=outcome.environment_steps,
            total_reward=outcome.total_reward,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
        )

    def close(self) -> None:
        """Release the process-owned wireless connection, when configured."""

        with self._client_lock:
            client = self._wireless_client
            self._wireless_client = None
        if client is not None:
            _shutdown_client(client)

    def _inference_client(self, timeout_s: float) -> Any:
        if self.config.inference_transport != "wireless":
            return self.client_factory(self.config, timeout_s)
        with self._client_lock:
            if self._wireless_client is None:
                self._wireless_client = self.client_factory(self.config, timeout_s)
            client = self._wireless_client
        with_timeout = getattr(client, "with_timeout", None)
        return with_timeout(timeout_s) if callable(with_timeout) else client

    def _release_inference_client(self, client: Any) -> None:
        if self.config.inference_transport != "wireless":
            _shutdown_client(client)


def create_inference_client(
    config: SimulationServiceConfig,
    timeout_s: float,
) -> Any:
    return build_inference_client(
        config.inference_transport,
        config.inference_endpoint,
        config.inference_options,
        backend=config.inference_backend,
        timeout_s=timeout_s,
    )


def _shutdown_client(client: Any) -> None:
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown()


__all__ = ["SimulationService", "SimulationServiceError", "create_inference_client"]
