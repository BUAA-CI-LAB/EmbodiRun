"""Authenticated HTTP surface for the Go2 control executor."""

from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any, ClassVar

from .config import MAX_REQUEST_BYTES, MIN_EXTERNAL_TOKEN_CHARS
from .executor import ActionExecutor
from .types import ApiError


def valid_api_token_for_bind(host: str, token: str | None) -> bool:
    """Require a non-trivial bearer token on every non-loopback bind."""

    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    return (
        token is not None
        and len(token) >= MIN_EXTERNAL_TOKEN_CHARS
        and token.strip() == token
        and not any(character.isspace() for character in token)
    )


class Go2RequestHandler(BaseHTTPRequestHandler):
    """Request handler configured per server via :func:`make_request_handler`."""

    executor: ClassVar[ActionExecutor]
    api_token: ClassVar[str | None] = None
    server_version = "Go2ControlAPI/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _authorized(self) -> bool:
        if not self.api_token:
            return True
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.api_token}"
        return hmac.compare_digest(header, expected)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:
        if not self._check_auth():
            return
        if self.path == "/health":
            snapshot = self.executor.snapshot()
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "transport": snapshot["transport"],
                    "operator_motion_ready": snapshot["operator_motion_ready"],
                    "robot_state_available": snapshot["robot_state_available"],
                    "robot_state_fresh": snapshot["robot_state_fresh"],
                    "active_action": snapshot["active_action"],
                },
            )
        elif self.path == "/v1/state":
            self._send_json(HTTPStatus.OK, self.executor.snapshot())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._check_auth():
            return
        if self.path not in {"/v1/actions", "/v1/stop"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            if self.path == "/v1/stop":
                self._send_json(
                    HTTPStatus.OK,
                    self.executor.stop("emergency_api_stop"),
                )
                return
            length_text = self.headers.get("Content-Length")
            if length_text is None:
                raise ApiError(
                    HTTPStatus.LENGTH_REQUIRED,
                    "Content-Length is required",
                )
            length = int(length_text)
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ApiError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "invalid request size",
                )
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            status, response = self.executor.execute(payload)
            self._send_json(status, response)
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON request"})
        except Exception as exc:  # noqa: BLE001 - HTTP boundary converts failures to JSON
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def make_request_handler(
    executor: ActionExecutor,
    token: str | None,
) -> type[Go2RequestHandler]:
    """Bind executor/token without sharing mutable globals across servers."""

    class BoundGo2RequestHandler(Go2RequestHandler):
        pass

    BoundGo2RequestHandler.executor = executor
    BoundGo2RequestHandler.api_token = token
    return BoundGo2RequestHandler


__all__ = ["Go2RequestHandler", "make_request_handler", "valid_api_token_for_bind"]
