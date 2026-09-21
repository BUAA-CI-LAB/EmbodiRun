"""HTTP listener, request handler, and CLI assembly for Control.

The application service, authorization, job lifecycle, and device coordination
live in ``embodirun.application``. This module owns only HTTP transport and
startup/shutdown wiring while re-exporting the historical service symbols for
compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from embodirun.application import control_service as _control_service
from embodirun.application.api import ControlApplication
from embodirun.application.auth import (
    AuthenticationError,
    AuthorizationError,
    AuthPolicy,
    Role,
)
from embodirun.application.contracts import (
    ControlContractError,
    ControlServiceConfig,
    TaskRequest,
    error_payload,
)
from embodirun.application.control_service import (
    ControlService,
    ControlServiceError,
    ControlTaskRejected,
    _configured_recorder,
    _control_job_database,
    _json_safe,
    _load_auth_policy,
    _payload_mapping,
    create_inference_client,
)
from embodirun.application.jobs import JobRegistry
from embodirun.application.model_loop import ControlRuntimeCancelled
from embodirun.devices.execution.arbitration import (
    CommandCancelled,
    CommandRejected,
)

from .http_api import ControlHTTPAPI, _headers, _token

_sensor_identity = _control_service._sensor_identity

_MAX_CONFIG_BYTES = 1024 * 1024

_MAX_REQUEST_BYTES = 64 * 1024

_TRUSTED_MANUAL_OWNER = ("trusted", "loopback")


class ControlHttpServer(ThreadingHTTPServer):
    """HTTP listener carrying one control service on a loopback-only socket."""

    daemon_threads = True

    def __init__(
        self,
        service: ControlService,
        *,
        application: ControlApplication | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        auth_policy: AuthPolicy | None = None,
    ) -> None:
        self.control_service = service
        self.control_application = application
        self._owns_application = False
        if self.control_application is None and isinstance(service, ControlService):
            registry = JobRegistry(_control_job_database(service, state_dir))
            self.control_application = ControlApplication(
                service,
                registry=registry,
                observation_store=service.observation_store,
                auth_policy=auth_policy,
            )
            self._owns_application = True
        if self.control_application is not None:
            self.control_api = ControlHTTPAPI(self.control_application)
        if ":" in service.config.bind:
            self.address_family = socket.AF_INET6
        try:
            super().__init__(
                (service.config.bind, service.config.port),
                ControlRequestHandler,
            )
        except BaseException:
            if self._owns_application:
                self.close_application()
            raise

    def close_application(self) -> bool:
        application = self.control_application
        if application is None or not self._owns_application:
            return True
        closed = application.close()
        if closed:
            self._owns_application = False
        return closed

    def server_close(self) -> None:
        super().server_close()
        self.close_application()


class ControlRequestHandler(BaseHTTPRequestHandler):
    """Small JSON HTTP surface for health checks and task submission."""

    server: ControlHttpServer

    def do_GET(self) -> None:
        if self._dispatch_application("GET"):
            return
        if self.path == "/v1/describe":
            self._send(HTTPStatus.OK, self.server.control_service.describe())
            return
        if self.path == "/v1/observe":
            try:
                self._send(HTTPStatus.OK, self.server.control_service.observe())
            except (
                ControlTaskRejected,
                ControlServiceError,
                OSError,
                ValueError,
            ) as error:
                self._send(HTTPStatus.CONFLICT, error_payload(str(error)))
            return
        if self.path == "/v1/control":
            if not self._authorize_legacy(Role.OBSERVER):
                return
            self._send(HTTPStatus.OK, self.server.control_service.control_snapshot())
            return
        if self.path != "/healthz":
            self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
            return
        try:
            payload = self.server.control_service.health()
        except Exception as error:  # noqa: BLE001 - normalize service failures
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "starting", "error": str(error)},
            )
            return
        self._send(HTTPStatus.OK, payload)

    def do_POST(self) -> None:
        if self.server.control_application is not None and self._is_application_route(self.path):
            try:
                body = self._read_optional_json()
            except (OSError, ValueError) as error:
                self._send(HTTPStatus.BAD_REQUEST, error_payload(str(error)))
                return
            if self._dispatch_application("POST", body=body):
                return
        if self.path != "/v1/tasks":
            if self.path.startswith("/v1/control/"):
                self._handle_control_post()
                return
            self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
            return
        try:
            request = TaskRequest.from_payload(self._read_json())
        except (ControlContractError, OSError, ValueError) as error:
            self._send(HTTPStatus.BAD_REQUEST, error_payload(str(error)))
            return
        try:
            result = self.server.control_service.execute(request)
        except (CommandRejected, CommandCancelled, ControlRuntimeCancelled) as error:
            self._send(HTTPStatus.CONFLICT, error_payload(str(error)))
            return
        except ControlTaskRejected as error:
            status = HTTPStatus.CONFLICT if "already executing" in str(error) else HTTPStatus.BAD_REQUEST
            self._send(status, error_payload(str(error)))
            return
        except Exception as error:  # noqa: BLE001 - normalize service failures
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, error_payload(str(error)))
            return
        self._send(HTTPStatus.OK, result.to_payload())

    def _handle_control_post(self) -> None:
        if not self._authorize_legacy(Role.CONTROLLER):
            return
        manual_scope = getattr(self, "_legacy_scope", None)
        manual_kwargs = (
            {}
            if manual_scope is None
            else {
                "caller_id": manual_scope[0],
                "session_id": manual_scope[1],
            }
        )
        try:
            if self.path == "/v1/control/emergency-stop":
                payload = self.server.control_service.emergency_stop()
            elif self.path == "/v1/control/reset":
                payload = self.server.control_service.reset_emergency_stop()
            elif self.path == "/v1/control/manual/acquire":
                payload = self.server.control_service.acquire_manual(**manual_kwargs)
            elif self.path == "/v1/control/manual/release":
                payload = self.server.control_service.release_manual(**manual_kwargs)
            elif self.path == "/v1/control/manual/deadman":
                root = _payload_mapping(self._read_json(), "manual deadman")
                active = root.get("active")
                if not isinstance(active, bool):
                    raise ValueError("manual deadman active must be boolean")
                payload = self.server.control_service.set_manual_deadman(active, **manual_kwargs)
            elif self.path == "/v1/control/manual/action":
                payload = self.server.control_service.submit_manual_action(self._read_json(), **manual_kwargs)
            else:
                self._send(HTTPStatus.NOT_FOUND, error_payload("route not found"))
                return
        except (ControlTaskRejected, CommandRejected) as error:
            self._send(HTTPStatus.CONFLICT, error_payload(str(error)))
            return
        except (OSError, ValueError) as error:
            self._send(HTTPStatus.BAD_REQUEST, error_payload(str(error)))
            return
        except Exception as error:  # noqa: BLE001 - normalize service failures
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, error_payload(str(error)))
            return
        self._send(HTTPStatus.OK, payload)

    def _dispatch_application(
        self,
        method: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> bool:
        if self.server.control_application is None or not self._is_application_route(self.path):
            return False
        try:
            response = self.server.control_api.dispatch(
                method,
                self.path,
                body=body,
                headers=dict(self.headers.items()),
            )
        except Exception as error:  # noqa: BLE001 - normalize handler boundary
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(error), "code": "application_error"},
            )
            return True
        self._send(HTTPStatus(response.status), response.payload)
        return True

    @staticmethod
    def _is_application_route(path: str) -> bool:
        route = path.split("?", 1)[0].rstrip("/") or "/"
        if route.startswith("/v1/control"):
            return False
        return (
            route
            in {
                "/v1/describe",
                "/v1/observe",
                "/v1/propose",
                "/v1/execute",
                "/v1/tasks",
                "/v1/cancel",
                "/v1/stop",
                "/v1/recordings/start",
                "/v1/recordings/stop",
                "/v1/recordings/status",
            }
            or route.startswith("/v1/jobs/")
            or route.startswith("/v1/media/")
            or route.startswith("/v1/recordings/record/")
        )

    def _authorize_legacy(self, role: Role) -> bool:
        application = self.server.control_application
        self._legacy_scope = None
        if application is None:
            return True
        try:
            headers = _headers(dict(self.headers.items()))
            token = _token(headers)
            caller_id, session_id = self.server.control_api._identity(headers, token)
            application.auth.authorize_scope(
                token,
                role,
                caller_id=caller_id,
                session_id=session_id,
            )
        except AuthenticationError as error:
            self._send(
                HTTPStatus.UNAUTHORIZED,
                {"error": str(error), "code": "authentication_required"},
            )
            return False
        except AuthorizationError as error:
            self._send(
                HTTPStatus.FORBIDDEN,
                {"error": str(error), "code": "forbidden"},
            )
            return False
        except ValueError as error:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {"error": str(error), "code": "invalid_request"},
            )
            return False
        self._legacy_scope = (caller_id, session_id)
        return True

    def _read_json(self) -> object:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        length_value = self.headers.get("Content-Length")
        try:
            length = int(length_value or "")
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if not 0 < length <= _MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be between 1 and {_MAX_REQUEST_BYTES} bytes")
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid UTF-8 JSON") from error

    def _read_optional_json(self) -> Mapping[str, Any] | None:
        length_value = self.headers.get("Content-Length")
        if length_value is None or not length_value.strip():
            return None
        try:
            if int(length_value) == 0:
                return None
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        value = self._read_json()
        if value is not None and not isinstance(value, Mapping):
            raise ValueError("request body must be a JSON object")
        return value

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(
            _json_safe(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Load one generated runtime and serve it until the supervisor stops us."""

    parser = argparse.ArgumentParser(prog="embodirun-control-serve")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="persistent control job/recorder state directory (default: device state)",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="JSON token-principal mapping; omission keeps trusted loopback/SSH mode",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        help="root directory for optional bounded observation/action recordings",
    )
    parser.add_argument(
        "--recording-id",
        help="safe recording directory name (default: runtime ID)",
    )
    args = parser.parse_args(argv)
    try:
        content = args.config.read_bytes()
        if len(content) > _MAX_CONFIG_BYTES:
            raise ValueError("control config is too large")
        config = ControlServiceConfig.from_json(content.decode("utf-8"))
        service = ControlService(config)
        recorder = _configured_recorder(
            service.observation_store,
            config,
            args.recording_dir,
            args.recording_id,
        )
        if recorder is not None:
            service.attach_recorder(recorder)
        server = ControlHttpServer(
            service,
            state_dir=args.state_dir,
            auth_policy=_load_auth_policy(args.token_file),
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        parser.error(str(error))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()
    return 0


__all__ = [
    "ControlHttpServer",
    "ControlRequestHandler",
    "ControlService",
    "ControlServiceError",
    "ControlTaskRejected",
    "create_inference_client",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI test
    raise SystemExit(main())
