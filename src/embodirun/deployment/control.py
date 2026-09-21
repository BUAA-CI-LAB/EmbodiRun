"""Host-side clients for the loopback Control application.

``ControlClient`` remains the compatibility wrapper for the original
``/v1/tasks`` request.  ``HostControlClient`` is the small JSON client used by
the Host CLI for the application routes.  Both clients use the executor's
loopback HTTP channel, so a caller never needs to open a robot connection or
create a second SSH session for individual action steps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from embodirun.application.contracts import (
    TaskRequest,
    TaskResult,
    error_message,
)
from embodirun.client.transport import ControlClientError, request_json

from .executor import Executor


class ControlClient:
    """Submit versioned tasks through the node executor's authenticated channel."""

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
            raise ValueError("control endpoint must be a credential-free loopback HTTP URL")
        try:
            if parsed.port is None:
                raise ValueError("control endpoint must include a port")
        except ValueError as error:
            raise ValueError("control endpoint must include a valid port") from error
        self.executor = executor
        self.endpoint = endpoint.rstrip("/")

    def run(self, request: TaskRequest, *, timeout_s: float) -> TaskResult:
        """Wait for one control task and validate its correlated result."""

        response = self.executor.request_json(
            "POST",
            f"{self.endpoint}/v1/tasks",
            request.to_payload(),
            timeout_s=timeout_s,
        )
        if response.status != 200:
            detail = error_message(response.payload) or "invalid error response"
            raise ControlClientError(f"control service returned HTTP {response.status}: {detail}")
        try:
            result = TaskResult.from_payload(response.payload)
        except ValueError as error:
            raise ControlClientError(f"control service response is invalid: {error}") from error
        if result.request_id != request.request_id:
            raise ControlClientError("control response request_id does not match")
        if result.runtime_id != request.runtime_id:
            raise ControlClientError("control response runtime_id does not match")
        return result


class HostControlClient(ControlClient):
    """Call application routes through one configured loopback endpoint.

    Caller and session IDs are intentionally required.  They are supplied by
    the operator or a stable wrapper configuration and are sent on every
    request, which lets the service scope observation and job ownership across
    separate CLI invocations.  The transport timeout is deliberately separate
    from the application ``wait`` timeout: losing an HTTP reply does not
    authorize a retry.
    """

    DEFAULT_TIMEOUT_S = 10.0

    def __init__(
        self,
        executor: Executor,
        endpoint: str,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
    ) -> None:
        super().__init__(executor, endpoint)
        self.caller_id = _required_id(caller_id, "caller_id")
        self.session_id = _required_id(session_id, "session_id")
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("token must be a non-empty string when provided")
        self.token = token.strip() if token is not None else None

    def describe(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
        """Return the service capability and ownership description."""

        return self._request("GET", "/v1/describe", None, timeout_s=timeout_s)

    def observe(
        self,
        *,
        runtime_id: str | None = None,
        observation_id: str | None = None,
        include_robot: bool = False,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Read the shared observation store or service observation endpoint."""

        query = _query(
            runtime_id=runtime_id,
            observation_id=observation_id,
            include_robot=_boolean(include_robot),
            max_age_ns=max_age_ns,
            max_skew_ns=max_skew_ns,
        )
        return self._request(
            "GET",
            f"/v1/observe{query}",
            None,
            timeout_s=timeout_s,
        )

    def media(
        self,
        observation_id: str,
        *,
        frame: str | None = None,
        runtime_id: str | None = None,
        include_data: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Read bounded media references for one shared observation.

        ``include_data`` is opt-in because a media response can be much larger
        than an observation reference.  The service still applies its own
        byte limit and stale/permission checks.
        """

        observation_id = _required_id(observation_id, "observation_id")
        query = _query(
            frame=frame,
            runtime_id=runtime_id,
            include_data=_boolean(include_data),
        )
        return self._request(
            "GET",
            f"/v1/media/{quote(observation_id, safe=':')}{query}",
            None,
            timeout_s=timeout_s,
        )

    def get_media(
        self,
        observation_id: str,
        *,
        frame: str | None = None,
        runtime_id: str | None = None,
        include_data: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Compatibility spelling for callers that prefer a verb-style name."""

        return self.media(
            observation_id,
            frame=frame,
            runtime_id=runtime_id,
            include_data=include_data,
            timeout_s=timeout_s,
        )

    def recording_status(
        self,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Read recorder state without starting or stopping a recording."""

        return self._request(
            "GET",
            "/v1/recordings/status",
            None,
            timeout_s=timeout_s,
        )

    def recording_start(
        self,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Request recording start through the service-owned recorder."""

        return self._request(
            "POST",
            "/v1/recordings/start",
            {},
            timeout_s=timeout_s,
            accepted_statuses={200, 202},
        )

    def recording_stop(
        self,
        *,
        recording_timeout_s: float | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Request recorder stop, keeping recorder and HTTP timeouts distinct."""

        if recording_timeout_s is not None and (
            isinstance(recording_timeout_s, bool)
            or not isinstance(recording_timeout_s, (int, float))
            or not math.isfinite(recording_timeout_s)
            or recording_timeout_s < 0
        ):
            raise ValueError("recording_timeout_s must be finite and non-negative")
        payload = {"timeout_s": recording_timeout_s} if recording_timeout_s is not None else {}
        return self._request(
            "POST",
            "/v1/recordings/stop",
            payload,
            timeout_s=timeout_s,
            accepted_statuses={200, 202},
        )

    def recording_get(
        self,
        observation_id: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Read one recorded observation by its stable observation ID."""

        observation_id = _required_id(observation_id, "observation_id")
        return self._request(
            "GET",
            f"/v1/recordings/record/{quote(observation_id, safe=':')}",
            None,
            timeout_s=timeout_s,
        )

    def execute(
        self,
        *,
        request_id: str,
        action: Mapping[str, Any] | None = None,
        actions: Sequence[Mapping[str, Any]] | None = None,
        source: str = "agent",
        observation_id: str | None = None,
        steps: int = 1,
        control_hz: float | None = None,
        wait: bool = False,
        job_timeout_s: float | None = None,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
        parameters: Mapping[str, Any] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Submit one bounded action chunk without retrying on transport loss."""

        request_id = _required_id(request_id, "request_id")
        if (action is None) == (actions is None):
            raise ValueError("exactly one of action or actions is required")
        if action is not None and not isinstance(action, Mapping):
            raise ValueError("action must be a JSON object")
        if actions is not None:
            if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
                raise ValueError("actions must be a JSON array")
            if not actions or any(not isinstance(item, Mapping) for item in actions):
                raise ValueError("actions must contain JSON objects")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if control_hz is not None and (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not math.isfinite(control_hz)
            or control_hz <= 0
        ):
            raise ValueError("control_hz must be finite and positive")
        if job_timeout_s is not None and (
            isinstance(job_timeout_s, bool)
            or not isinstance(job_timeout_s, (int, float))
            or not math.isfinite(job_timeout_s)
            or job_timeout_s <= 0
        ):
            raise ValueError("job_timeout_s must be finite and positive")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("parameters must be a JSON object")
        payload: dict[str, Any] = {
            "request_id": request_id,
            "source": source,
            "steps": steps,
            "wait": bool(wait),
        }
        if action is not None:
            payload["action"] = dict(action)
        else:
            payload["actions"] = [dict(item) for item in actions or ()]
        _put_optional(
            payload,
            observation_id=observation_id,
            control_hz=control_hz,
            timeout_s=job_timeout_s,
            max_age_ns=max_age_ns,
            max_skew_ns=max_skew_ns,
            parameters=dict(parameters) if parameters is not None else None,
        )
        return self._request(
            "POST",
            "/v1/execute",
            payload,
            timeout_s=timeout_s,
            request_id=request_id,
            accepted_statuses={200, 202},
        )

    def inspect(
        self,
        request_id: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Inspect the original request after acceptance or an unknown reply."""

        request_id = _required_id(request_id, "request_id")
        return self._request(
            "GET",
            f"/v1/jobs/{quote(request_id, safe=':')}",
            None,
            timeout_s=timeout_s,
            request_id=request_id,
        )

    def cancel(
        self,
        request_id: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Request cancellation scoped to this caller/session and request ID."""

        return self._job_mutation("cancel", request_id, timeout_s=timeout_s)

    def stop(
        self,
        request_id: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Request a stop for exactly one request ID."""

        return self._job_mutation("stop", request_id, timeout_s=timeout_s)

    def _job_mutation(
        self,
        operation: str,
        request_id: str,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        request_id = _required_id(request_id, "request_id")
        if operation not in {"cancel", "stop"}:
            raise ValueError("unsupported job operation")
        return self._request(
            "POST",
            f"/v1/jobs/{quote(request_id, safe=':')}/{operation}",
            {},
            timeout_s=timeout_s,
            request_id=request_id,
            accepted_statuses={200, 202},
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
        request_id: str | None = None,
        accepted_statuses: set[int] | frozenset[int] = frozenset({200}),
    ) -> dict[str, Any]:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        return request_json(
            self.executor,
            method,
            f"{self.endpoint}{path}",
            payload,
            timeout_s=float(timeout_s),
            headers=self._headers(),
            request_id=request_id,
            accepted_statuses=accepted_statuses,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "X-EmbodiRun-Caller-Id": self.caller_id,
            "X-EmbodiRun-Session-Id": self.session_id,
        }
        if self.token is not None:
            headers["X-EmbodiRun-Token"] = self.token
        return headers


def _required_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _boolean(value: bool) -> str:
    if not isinstance(value, bool):
        raise ValueError("boolean query values must be bool")
    return "true" if value else "false"


def _query(**values: Any) -> str:
    pairs = [(name, value) for name, value in values.items() if value is not None]
    return f"?{urlencode(pairs)}" if pairs else ""


def _put_optional(target: dict[str, Any], **values: Any) -> None:
    for name, value in values.items():
        if value is not None:
            target[name] = value


__all__ = ["ControlClient", "ControlClientError", "HostControlClient"]
