"""Transport-neutral HTTP dispatch for the control application.

``ControlHTTPAPI`` is intentionally not an HTTP server.  It converts one
decoded request into a :class:`ControlApplication` call and returns a small
status/payload pair that an existing handler can serialize.  Keeping this
boundary independent lets the service preserve its current server lifecycle
while the application API is tested without sockets or hardware.

Authentication and caller/session identity are taken from headers.  A
configured :class:`AuthPolicy` therefore controls every route, including
observation and media reads; a request body cannot select its own role.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

from embodirun.application.api import (
    ApplicationError,
    ApplicationInvalidRequest,
    ApplicationProposalFailed,
    ApplicationStaleObservation,
    ApplicationUnsupported,
    ControlApplication,
)
from embodirun.application.auth import AuthenticationError, AuthorizationError
from embodirun.application.contracts import TaskRequest, TaskResult, error_payload
from embodirun.application.jobs import (
    JobCancelledError,
    JobConflict,
    JobError,
    JobExecutionError,
    JobNotFound,
    JobOwnershipError,
    JobTimeout,
    JobUnknownError,
)


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    """The adapter result an existing HTTP handler can serialize."""

    status: int
    payload: Mapping[str, Any]

    def json_bytes(self) -> bytes:
        """Encode the detached payload for a conventional JSON response."""

        return json.dumps(self.payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


class HTTPDispatchError(ValueError):
    """Malformed method, path, query, or JSON body."""


class ControlHTTPAPI:
    """Dispatch versioned application routes without owning a socket."""

    def __init__(
        self,
        application: ControlApplication,
        *,
        trusted_caller_id: str = "http-trusted",
        trusted_session_id: str = "loopback",
    ) -> None:
        if not isinstance(application, ControlApplication):
            raise TypeError("application must be a ControlApplication")
        for name, value in (
            ("trusted_caller_id", trusted_caller_id),
            ("trusted_session_id", trusted_session_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        self.application = application
        self.trusted_caller_id = trusted_caller_id
        self.trusted_session_id = trusted_session_id

    def dispatch(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResponse:
        """Dispatch one request and turn expected failures into JSON errors."""

        try:
            return self._dispatch(method, path, body=body, headers=headers)
        except AuthenticationError as error:
            return self._error(401, error)
        except AuthorizationError as error:
            return self._error(403, error)
        except JobOwnershipError as error:
            return self._error(403, error)
        except JobNotFound as error:
            return self._error(404, error)
        except JobConflict as error:
            return self._error(409, error)
        except JobTimeout as error:
            return HTTPResponse(
                202,
                {
                    "code": "job_timeout",
                    "status": "pending",
                    "job": _record(error.record),
                },
            )
        except (JobUnknownError, JobCancelledError) as error:
            return HTTPResponse(
                409,
                {
                    "code": ("job_unknown" if isinstance(error, JobUnknownError) else "job_cancelled"),
                    "status": "uncertain",
                    "job": _record(error.record),
                },
            )
        except JobExecutionError as error:
            return HTTPResponse(
                502,
                {
                    "code": "job_failed",
                    "status": "failed",
                    "job": _record(error.record),
                },
            )
        except ApplicationProposalFailed as error:
            return self._error(502, error)
        except (ApplicationStaleObservation, ApplicationUnsupported) as error:
            return self._error(409, error)
        except (
            ApplicationInvalidRequest,
            HTTPDispatchError,
            ValueError,
            TypeError,
        ) as error:
            return self._error(400, error)
        except JobError as error:
            return self._error(409, error)
        except ApplicationError as error:
            return self._error(409, error)

    def _dispatch(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
    ) -> HTTPResponse:
        verb = _method(method)
        parsed = urlsplit(path)
        route = parsed.path.rstrip("/") or "/"
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        request_body = _body(body)
        header_values = _headers(headers)
        token = _token(header_values)
        caller_id, session_id = self._identity(header_values, token)

        if verb == "GET" and route == "/v1/describe":
            return HTTPResponse(
                200,
                self.application.describe(
                    caller_id=caller_id,
                    session_id=session_id,
                    token=token,
                ),
            )
        if verb == "GET" and route == "/v1/observe":
            return HTTPResponse(
                200,
                self.application.observe(
                    caller_id=caller_id,
                    session_id=session_id,
                    token=token,
                    runtime_id=_optional(query, "runtime_id"),
                    observation_id=_optional(query, "observation_id"),
                    include_robot=_bool(query.get("include_robot", "false")),
                    max_age_ns=_optional_int(query, "max_age_ns"),
                    max_skew_ns=_optional_int(query, "max_skew_ns"),
                ),
            )
        if verb == "GET" and route.startswith("/v1/media/"):
            observation_id = route.removeprefix("/v1/media/")
            if not observation_id:
                raise HTTPDispatchError("media route requires observation_id")
            return HTTPResponse(
                200,
                self.application.media(
                    caller_id=caller_id,
                    session_id=session_id,
                    observation_id=observation_id,
                    frame_name=_optional(query, "frame"),
                    runtime_id=_optional(query, "runtime_id"),
                    include_data=_bool(query.get("include_data", "false")),
                    token=token,
                ),
            )
        if verb == "GET" and route == "/v1/recordings/status":
            return HTTPResponse(
                200,
                self.application.recording_status(
                    caller_id=caller_id,
                    session_id=session_id,
                    token=token,
                ),
            )
        if verb == "GET" and route.startswith("/v1/recordings/record/"):
            observation_id = route.removeprefix("/v1/recordings/record/")
            if not observation_id:
                raise HTTPDispatchError("record route requires observation_id")
            value = self.application.recording_get(
                caller_id=caller_id,
                session_id=session_id,
                observation_id=observation_id,
                token=token,
            )
            if value is None:
                return HTTPResponse(404, error_payload("record was not found"))
            return HTTPResponse(200, value)
        if verb == "GET" and route.startswith("/v1/jobs/"):
            request_id = route.removeprefix("/v1/jobs/")
            if not request_id or "/" in request_id:
                raise HTTPDispatchError("job route requires one request_id")
            return HTTPResponse(
                200,
                self.application.inspect(
                    caller_id=caller_id,
                    session_id=session_id,
                    request_id=request_id,
                    token=token,
                ),
            )

        if verb == "POST" and route == "/v1/execute":
            execute_args = self._execute_args(request_body, caller_id, session_id, token)
            result = self.application.execute(**execute_args)
            return HTTPResponse(200 if execute_args["wait"] else 202, result)
        if verb == "POST" and route == "/v1/propose":
            propose_args = self._propose_args(request_body, caller_id, session_id, token)
            return HTTPResponse(200, self.application.propose(**propose_args))
        if verb == "POST" and route == "/v1/tasks":
            task_payload = {key: value for key, value in request_body.items() if key not in {"wait", "timeout_s"}}
            request = TaskRequest.from_payload(task_payload)
            result = self.application.legacy_task(
                request,
                caller_id=caller_id,
                session_id=session_id,
                token=token,
                wait=_bool(request_body.get("wait", "true")),
                timeout_s=_optional_float(request_body, "timeout_s"),
            )
            return HTTPResponse(200, _task_payload(result))
        if verb == "POST" and route == "/v1/recordings/start":
            return HTTPResponse(
                200,
                self.application.recording_start(
                    caller_id=caller_id,
                    session_id=session_id,
                    token=token,
                ),
            )
        if verb == "POST" and route == "/v1/recordings/stop":
            timeout_s = _optional_float(request_body, "timeout_s")
            return HTTPResponse(
                200,
                self.application.recording_stop(
                    caller_id=caller_id,
                    session_id=session_id,
                    timeout_s=1.0 if timeout_s is None else timeout_s,
                    token=token,
                ),
            )
        if verb == "POST" and route.startswith("/v1/jobs/"):
            suffix = route.removeprefix("/v1/jobs/").split("/")
            if len(suffix) != 2 or not suffix[0] or suffix[1] not in {"cancel", "stop"}:
                raise HTTPDispatchError("job mutation route must end in /cancel or /stop")
            request_id, operation = suffix
            method = self.application.stop if operation == "stop" else self.application.cancel
            return HTTPResponse(
                202,
                method(
                    caller_id=caller_id,
                    session_id=session_id,
                    request_id=request_id,
                    token=token,
                ),
            )
        if verb == "POST" and route in {"/v1/cancel", "/v1/stop"}:
            request_id = request_body.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise HTTPDispatchError("request_id is required")
            method = self.application.stop if route.endswith("/stop") else self.application.cancel
            return HTTPResponse(
                202,
                method(
                    caller_id=caller_id,
                    session_id=session_id,
                    request_id=request_id,
                    token=token,
                ),
            )
        raise HTTPDispatchError(f"unsupported route: {verb} {route}")

    def _execute_args(
        self,
        payload: Mapping[str, Any],
        caller_id: str,
        session_id: str,
        token: str | None,
    ) -> dict[str, Any]:
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise HTTPDispatchError("request_id is required")
        has_action = "action" in payload
        has_actions = "actions" in payload
        if has_action == has_actions:
            raise HTTPDispatchError("exactly one of action or actions is required")
        return {
            "caller_id": caller_id,
            "session_id": session_id,
            "request_id": request_id,
            "action": payload["action"] if has_action else payload["actions"],
            "source": payload.get("source", "agent"),
            "observation_id": payload.get("observation_id"),
            "runtime_id": payload.get("runtime_id"),
            "steps": payload.get("steps", 1),
            "control_hz": payload.get("control_hz"),
            "wait": _bool(payload.get("wait", "false")),
            "timeout_s": _optional_float(payload, "timeout_s"),
            "token": token,
            "max_age_ns": _optional_int(payload, "max_age_ns"),
            "max_skew_ns": _optional_int(payload, "max_skew_ns"),
            "parameters": payload.get("parameters"),
        }

    def _propose_args(
        self,
        payload: Mapping[str, Any],
        caller_id: str,
        session_id: str,
        token: str | None,
    ) -> dict[str, Any]:
        request_id = payload.get("request_id")
        observation_id = payload.get("observation_id")
        instruction = payload.get("instruction")
        if not isinstance(request_id, str) or not request_id.strip():
            raise HTTPDispatchError("request_id is required")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise HTTPDispatchError("observation_id is required")
        if not isinstance(instruction, str) or not instruction.strip():
            raise HTTPDispatchError("instruction is required")
        return {
            "caller_id": caller_id,
            "session_id": session_id,
            "request_id": request_id,
            "observation_id": observation_id,
            "instruction": instruction,
            "runtime_id": payload.get("runtime_id"),
            "timeout_s": payload.get("timeout_s", 10.0),
            "token": token,
            "max_age_ns": _optional_int(payload, "max_age_ns"),
            "max_skew_ns": _optional_int(payload, "max_skew_ns"),
        }

    def _identity(self, headers: Mapping[str, str], token: str | None) -> tuple[str, str]:
        # ``x-embodirun-*`` are the canonical identity headers; the legacy
        caller = _first_header(headers, "x-embodirun-caller-id", "x-caller-id")
        session = _first_header(headers, "x-embodirun-session-id", "x-session-id")
        if self.application.auth.token_authentication_enabled:
            # Let the application report the normal 401 for a missing token;
            # scope headers are required once a token is actually present.
            if token is None:
                return (
                    caller or self.trusted_caller_id,
                    session or self.trusted_session_id,
                )
            if not caller or not session:
                raise ApplicationInvalidRequest("token-authenticated routes require caller and session headers")
            return caller, session
        return caller or self.trusted_caller_id, session or self.trusted_session_id

    @staticmethod
    def _error(status: int, error: BaseException) -> HTTPResponse:
        payload = error_payload(str(error) or error.__class__.__name__)
        payload["code"] = _error_code(error)
        return HTTPResponse(status, payload)


def _method(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPDispatchError("method must be a non-empty string")
    return value.upper()


def _body(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HTTPDispatchError("body must be a JSON object")
    return dict(value)


def _headers(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HTTPDispatchError("headers must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise HTTPDispatchError("headers must contain string names and values")
        result[key.lower()] = item
    return result


def _first_header(headers: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = headers.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _token(headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("authorization")
    if authorization is not None:
        parts = authorization.strip().split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise AuthenticationError("authorization must use a Bearer token")
        return parts[1].strip()
    return _first_header(headers, "x-embodirun-token", "x-control-token")


def _optional(mapping: Mapping[str, Any], name: str) -> str | None:
    value = mapping.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise HTTPDispatchError(f"{name} must be a string")
    return value


def _optional_int(mapping: Mapping[str, Any], name: str) -> int | None:
    value = mapping.get(name)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise HTTPDispatchError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise HTTPDispatchError(f"{name} must be an integer") from error
    if isinstance(value, float) and parsed != value:
        raise HTTPDispatchError(f"{name} must be an integer")
    return parsed


def _optional_float(mapping: Mapping[str, Any], name: str) -> float | None:
    value = mapping.get(name)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise HTTPDispatchError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise HTTPDispatchError(f"{name} must be a number") from error


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise HTTPDispatchError("boolean values must be true or false")


def _task_payload(value: TaskResult | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, TaskResult):
        return value.to_payload()
    if isinstance(value, Mapping):
        return dict(value)
    raise ApplicationError("legacy task did not return a TaskResult or mapping")


def _error_code(error: BaseException) -> str:
    """Small machine-readable categories without freezing a new contract."""

    code_by_type = (
        (AuthenticationError, "authentication_required"),
        (AuthorizationError, "forbidden"),
        (JobOwnershipError, "job_owner_mismatch"),
        (JobNotFound, "job_not_found"),
        (JobConflict, "job_conflict"),
        (ApplicationStaleObservation, "observation_stale"),
        (ApplicationProposalFailed, "proposal_failed"),
        (ApplicationUnsupported, "unsupported"),
        (ApplicationInvalidRequest, "invalid_request"),
        (HTTPDispatchError, "invalid_request"),
    )
    for error_type, code in code_by_type:
        if isinstance(error, error_type):
            return code
    if isinstance(error, JobError):
        return "job_error"
    if isinstance(error, ApplicationError):
        return "application_error"
    return "invalid_request"


def _record(record: Any) -> dict[str, Any]:
    # This is deliberately duck-typed to avoid importing application private
    # response helpers into the transport module.
    from embodirun.application.api import _record_payload

    return _record_payload(record)


__all__ = ["ControlHTTPAPI", "HTTPDispatchError", "HTTPResponse"]
