"""Versioned Host-to-Simulation configuration and episode messages."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SIMULATION_CONFIG_SCHEMA = "embodirun.simulation.config.v1"
EPISODE_REQUEST_SCHEMA = "embodirun.simulation.episode.v1"
EPISODE_RESULT_SCHEMA = "embodirun.simulation.result.v1"
ERROR_SCHEMA = "embodirun.simulation.error.v1"


class SimulationContractError(ValueError):
    """A simulation service message is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class SimulationServiceConfig:
    runtime_id: str
    binding_kind: str
    bind: str
    port: int
    inference_transport: str
    inference_endpoint: str
    inference_options: Mapping[str, Any]
    simulator_id: str
    simulator_kind: str
    simulator_options: Mapping[str, Any]
    inference_backend: str = "embodiinfer"

    def __post_init__(self) -> None:
        for name in (
            "runtime_id",
            "binding_kind",
            "bind",
            "inference_backend",
            "inference_transport",
            "inference_endpoint",
            "simulator_id",
            "simulator_kind",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SimulationContractError(f"{name} must not be empty")
        if self.bind not in {"127.0.0.1", "localhost", "::1"}:
            raise SimulationContractError("simulation service must bind to a loopback address")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise SimulationContractError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise SimulationContractError("port must be between 1 and 65535")
        if self.inference_transport not in {"http", "wireless"}:
            raise SimulationContractError("inference transport must be http or wireless")
        if self.inference_backend not in {"embodiinfer", "sglang"}:
            raise SimulationContractError("inference backend must be embodiinfer or sglang")
        for name in ("inference_options", "simulator_options"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": SIMULATION_CONFIG_SCHEMA,
                "runtime_id": self.runtime_id,
                "binding": self.binding_kind,
                "server": {"bind": self.bind, "port": self.port},
                "inference": {
                    "backend": self.inference_backend,
                    "transport": self.inference_transport,
                    "endpoint": self.inference_endpoint,
                    "options": dict(self.inference_options),
                },
                "simulator": {
                    "id": self.simulator_id,
                    "type": self.simulator_kind,
                    "options": dict(self.simulator_options),
                },
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> SimulationServiceConfig:
        try:
            root = _mapping(json.loads(value), "simulation config")
        except json.JSONDecodeError as error:
            raise SimulationContractError(f"simulation config is invalid JSON: {error}") from error
        if root.get("schema") != SIMULATION_CONFIG_SCHEMA:
            raise SimulationContractError("unsupported simulation config schema")
        server = _mapping(root.get("server"), "simulation config.server")
        inference = _mapping(root.get("inference"), "simulation config.inference")
        simulator = _mapping(root.get("simulator"), "simulation config.simulator")
        return cls(
            runtime_id=_string(root, "runtime_id", "simulation config"),
            binding_kind=_string(root, "binding", "simulation config"),
            bind=_string(server, "bind", "simulation config.server"),
            port=server.get("port"),
            inference_backend=_optional_string(
                inference.get("backend"),
                "simulation config.inference.backend",
            )
            or "embodiinfer",
            inference_transport=_string(inference, "transport", "simulation config.inference"),
            inference_endpoint=_string(inference, "endpoint", "simulation config.inference"),
            inference_options=_mapping(inference.get("options", {}), "simulation config.inference.options"),
            simulator_id=_string(simulator, "id", "simulation config.simulator"),
            simulator_kind=_string(simulator, "type", "simulation config.simulator"),
            simulator_options=_mapping(simulator.get("options", {}), "simulation config.simulator.options"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    request_id: str
    runtime_id: str
    prompt: str | None
    task: str | None
    seed: int | None
    chunk_steps: int
    max_policy_steps: int
    inference_timeout_s: float

    def __post_init__(self) -> None:
        for name in ("request_id", "runtime_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SimulationContractError(f"{name} must not be empty")
        for name in ("prompt", "task"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SimulationContractError(f"{name} must be non-empty when provided")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise SimulationContractError("seed must be an integer when provided")
        for name in ("chunk_steps", "max_policy_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SimulationContractError(f"{name} must be a positive integer")
        if (
            isinstance(self.inference_timeout_s, bool)
            or not isinstance(self.inference_timeout_s, (int, float))
            or not math.isfinite(self.inference_timeout_s)
            or self.inference_timeout_s <= 0
        ):
            raise SimulationContractError("inference_timeout_s must be a finite positive number")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": EPISODE_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "runtime_id": self.runtime_id,
            "prompt": self.prompt,
            "task": self.task,
            "seed": self.seed,
            "chunk_steps": self.chunk_steps,
            "max_policy_steps": self.max_policy_steps,
            "inference_timeout_s": self.inference_timeout_s,
        }

    @classmethod
    def from_payload(cls, value: object) -> EpisodeRequest:
        payload = _mapping(value, "episode request")
        if payload.get("schema") != EPISODE_REQUEST_SCHEMA:
            raise SimulationContractError("unsupported episode request schema")
        return cls(
            request_id=_string(payload, "request_id", "episode request"),
            runtime_id=_string(payload, "runtime_id", "episode request"),
            prompt=_optional_string(payload.get("prompt"), "episode request.prompt"),
            task=_optional_string(payload.get("task"), "episode request.task"),
            seed=payload.get("seed"),
            chunk_steps=payload.get("chunk_steps"),
            max_policy_steps=payload.get("max_policy_steps"),
            inference_timeout_s=payload.get("inference_timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    request_id: str
    runtime_id: str
    policy_steps: int
    environment_steps: int
    total_reward: float
    terminated: bool
    truncated: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": EPISODE_RESULT_SCHEMA,
            "request_id": self.request_id,
            "runtime_id": self.runtime_id,
            "policy_steps": self.policy_steps,
            "environment_steps": self.environment_steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @classmethod
    def from_payload(cls, value: object) -> EpisodeResult:
        payload = _mapping(value, "episode result")
        if payload.get("schema") != EPISODE_RESULT_SCHEMA:
            raise SimulationContractError("unsupported episode result schema")
        result = cls(
            request_id=_string(payload, "request_id", "episode result"),
            runtime_id=_string(payload, "runtime_id", "episode result"),
            policy_steps=payload.get("policy_steps"),
            environment_steps=payload.get("environment_steps"),
            total_reward=payload.get("total_reward"),
            terminated=payload.get("terminated"),
            truncated=payload.get("truncated"),
        )
        for name in ("policy_steps", "environment_steps"):
            count = getattr(result, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SimulationContractError(f"{name} must be non-negative")
        if (
            isinstance(result.total_reward, bool)
            or not isinstance(result.total_reward, (int, float))
            or not math.isfinite(result.total_reward)
        ):
            raise SimulationContractError("total_reward must be finite")
        for name in ("terminated", "truncated"):
            if not isinstance(getattr(result, name), bool):
                raise SimulationContractError(f"{name} must be a boolean")
        return result


def error_payload(message: str) -> dict[str, str]:
    return {"schema": ERROR_SCHEMA, "error": message}


def error_message(value: object) -> str | None:
    if not isinstance(value, Mapping) or value.get("schema") != ERROR_SCHEMA:
        return None
    message = value.get("error")
    return message if isinstance(message, str) and message else None


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SimulationContractError(f"{context} must be an object")
    return dict(value)


def _string(value: Mapping[str, Any], name: str, context: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise SimulationContractError(f"{context}.{name} must be non-empty")
    return result


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SimulationContractError(f"{context} must be non-empty when provided")
    return value


__all__ = [
    "EpisodeRequest",
    "EpisodeResult",
    "SimulationContractError",
    "SimulationServiceConfig",
    "error_message",
    "error_payload",
]
