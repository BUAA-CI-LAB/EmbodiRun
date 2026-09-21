"""HTTP listener and CLI for the bounded simulator application service."""

from __future__ import annotations

import argparse
import json
import socket
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from embodirun.application.simulation.contracts import (
    EpisodeRequest,
    SimulationContractError,
    SimulationServiceConfig,
    error_payload,
)
from embodirun.application.simulation.service import (
    SimulationService,
    SimulationServiceError,
    create_inference_client,
)

_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 64 * 1024


class SimulationHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: SimulationService) -> None:
        self.simulation_service = service
        if ":" in service.config.bind:
            self.address_family = socket.AF_INET6
        super().__init__(
            (service.config.bind, service.config.port),
            SimulationRequestHandler,
        )


class SimulationRequestHandler(BaseHTTPRequestHandler):
    server: SimulationHttpServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
            return
        try:
            payload = self.server.simulation_service.health()
        except Exception as error:  # noqa: BLE001
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "starting", "error": str(error)},
            )
            return
        self._send(HTTPStatus.OK, payload)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/episodes":
            self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
            return
        try:
            request = EpisodeRequest.from_payload(self._read_json())
            result = self.server.simulation_service.execute(request)
        except (SimulationContractError, SimulationServiceError, ValueError) as error:
            status = HTTPStatus.CONFLICT if "already executing" in str(error) else HTTPStatus.BAD_REQUEST
            self._send(status, error_payload(str(error)))
            return
        except Exception as error:  # noqa: BLE001
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, error_payload(str(error)))
            return
        self._send(HTTPStatus.OK, result.to_payload())

    def _read_json(self) -> object:
        if self.headers.get_content_type() != "application/json":
            raise SimulationContractError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError as error:
            raise SimulationContractError("Content-Length must be an integer") from error
        if not 0 < length <= _MAX_REQUEST_BYTES:
            raise SimulationContractError("request body size is invalid")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SimulationContractError("request body must be valid JSON") from error

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodirun-simulation-serve")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        content = args.config.read_bytes()
        if len(content) > _MAX_CONFIG_BYTES:
            raise ValueError("simulation config is too large")
        config = SimulationServiceConfig.from_json(content.decode("utf-8"))
        server = SimulationHttpServer(SimulationService(config))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        parser.error(str(error))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.simulation_service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SimulationHttpServer",
    "SimulationService",
    "SimulationServiceError",
    "create_inference_client",
    "main",
]
