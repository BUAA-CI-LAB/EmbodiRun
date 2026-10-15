"""Canonical simulation contracts and episode runtime."""

from .contracts import (
    EpisodeRequest,
    EpisodeResult,
    SimulationContractError,
    SimulationServiceConfig,
)
from .runtime import EpisodeOutcome, SimulationRuntime

_LAZY_ASYNC = {"AsyncEpisodeOutcome", "AsyncSimulationConfig", "AsyncSimulationRuntime"}
_LAZY_SERVICE = {"SimulationService", "SimulationServiceError", "create_inference_client"}


def __getattr__(name: str):
    if name in _LAZY_ASYNC:
        from .async_runtime import AsyncEpisodeOutcome, AsyncSimulationConfig, AsyncSimulationRuntime

        return {
            "AsyncEpisodeOutcome": AsyncEpisodeOutcome,
            "AsyncSimulationConfig": AsyncSimulationConfig,
            "AsyncSimulationRuntime": AsyncSimulationRuntime,
        }[name]
    if name in _LAZY_SERVICE:
        from .service import SimulationService, SimulationServiceError, create_inference_client

        return {
            "SimulationService": SimulationService,
            "SimulationServiceError": SimulationServiceError,
            "create_inference_client": create_inference_client,
        }[name]
    raise AttributeError(name)


__all__ = [
    "EpisodeOutcome",
    "EpisodeRequest",
    "EpisodeResult",
    "SimulationContractError",
    "SimulationRuntime",
    "AsyncEpisodeOutcome",
    "AsyncSimulationConfig",
    "AsyncSimulationRuntime",
    "SimulationServiceConfig",
    "SimulationService",
    "SimulationServiceError",
    "create_inference_client",
]
