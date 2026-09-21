"""Host-side client for a loopback simulation service over its node executor."""

from __future__ import annotations

from urllib.parse import urlsplit

from embodirun.application.simulation.contracts import (
    EpisodeRequest,
    EpisodeResult,
    error_message,
)

from .executor import Executor


class SimulationClientError(RuntimeError):
    pass


class SimulationClient:
    def __init__(self, executor: Executor, endpoint: str) -> None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("simulation endpoint must be a credential-free loopback HTTP URL")
        try:
            if parsed.port is None:
                raise ValueError("simulation endpoint must include a port")
        except ValueError as error:
            raise ValueError("simulation endpoint must include a valid port") from error
        self.executor = executor
        self.endpoint = endpoint.rstrip("/")

    def run(self, request: EpisodeRequest, *, timeout_s: float) -> EpisodeResult:
        response = self.executor.request_json(
            "POST",
            f"{self.endpoint}/v1/episodes",
            request.to_payload(),
            timeout_s=timeout_s,
        )
        if response.status != 200:
            detail = error_message(response.payload) or "invalid error response"
            raise SimulationClientError(f"simulation service returned HTTP {response.status}: {detail}")
        try:
            result = EpisodeResult.from_payload(response.payload)
        except ValueError as error:
            raise SimulationClientError(f"simulation service response is invalid: {error}") from error
        if result.request_id != request.request_id:
            raise SimulationClientError("simulation response request_id does not match")
        if result.runtime_id != request.runtime_id:
            raise SimulationClientError("simulation response runtime_id does not match")
        return result


__all__ = ["SimulationClient", "SimulationClientError"]
