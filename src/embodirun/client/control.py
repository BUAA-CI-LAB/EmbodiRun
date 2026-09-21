"""A dependency-free client for the public Control HTTP API.

This is deliberately a thin transport wrapper.  It keeps caller/session
identity on every request, exposes the existing observe/execute/jobs/media
routes, and returns detached JSON values.  It does not import a Control
service implementation, inspect private drivers, or infer a robot action
space from an arbitrary vector.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError as UrllibHTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from .transport import ControlClientError
from .transport import request_json as shared_request_json


class AgentClientError(ControlClientError):
    """Base error raised by :class:`ControlClient`."""


class ControlTransportError(AgentClientError):
    """The Control endpoint could not be reached or returned invalid JSON.

    A transport failure after an execute request is an unknown outcome.  The
    caller must inspect the original request ID before considering a retry.
    """

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        self.request_id = request_id
        self.unknown = request_id is not None
        super().__init__(message, request_id=request_id, unknown=self.unknown)


class ControlHTTPError(AgentClientError):
    """Control returned a non-success status with its structured error."""

    def __init__(
        self,
        status: int,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        self.status = int(status)
        self.payload = dict(payload or {})
        self.request_id = request_id
        self.unknown = self.payload.get("status") in {"unknown", "uncertain"}
        code = self.payload.get("code")
        message = self.payload.get("error") or self.payload.get("message") or code
        suffix = f" ({code})" if code else ""
        super().__init__(
            f"Control HTTP {self.status}{suffix}: {message or 'request failed'}",
            status=self.status,
            payload=self.payload,
            request_id=request_id,
            unknown=self.unknown,
        )


class UnsupportedOperation(ControlHTTPError):
    """The configured owner cannot provide the requested operation."""


@dataclass(frozen=True, slots=True)
class Observation:
    """Detached shared observation returned by ``GET /v1/observe``.

    ``observation_id`` is the consistency token to send with a subsequent
    action.  ``fresh`` is the server's freshness result; the client does not
    turn it into physical execution evidence.
    """

    observation_id: str
    payload: Mapping[str, Any]

    @property
    def fresh(self) -> bool | None:
        value = self.payload.get("fresh")
        return value if isinstance(value, bool) else None

    @property
    def robot(self) -> Any:
        # Application responses use ``state`` for the retained robot snapshot;
        # keep the convenience ``robot`` alias used by lightweight callers
        # and older test doubles.
        value = self.payload.get("robot", self.payload.get("state"))
        if isinstance(value, Mapping) and isinstance(value.get("values"), Mapping):
            return value["values"]
        return value


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _query_path(path: str, query: Mapping[str, Any] | None) -> str:
    if not query:
        return path
    values = {key: value for key, value in query.items() if value is not None}
    return f"{path}?{urlencode(values)}" if values else path


class ControlClient:
    """Call the existing versioned Control API over JSON HTTP.

    ``endpoint`` may include a path prefix but must identify the Control
    listener.  ``caller_id`` and ``session_id`` are sent as headers on every
    request.  A token is optional for trusted loopback deployments and is
    sent in the Control service's ``X-EmbodiRun-Token`` header when configured.

    ``describe``, ``observe``, ``media`` and ``inspect`` are read-only queries.
    ``propose`` computes against a retained snapshot but does not prepare or
    move a device.  ``execute``, ``cancel``, ``stop`` and the ``manual_*``
    methods are control requests and their response is service/job evidence;
    a successful HTTP response is not proof of a physical all-stop.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        caller_id: str,
        session_id: str,
        token: str | None = None,
        timeout_s: float = 10.0,
        opener: Any = urlopen,
    ) -> None:
        endpoint = _non_empty(endpoint, "endpoint").rstrip("/")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("control endpoint must be an HTTP(S) URL without credentials, query, or fragment")
        try:
            _ = parsed.port
        except ValueError as error:
            raise ValueError("control endpoint must include a valid port") from error
        _non_empty(caller_id, "caller_id")
        _non_empty(session_id, "session_id")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be finite and positive")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if token is not None:
            _non_empty(token, "token")
        if not callable(opener):
            raise TypeError("opener must be callable")
        self.endpoint = endpoint
        self.caller_id = caller_id
        self.session_id = session_id
        self.token = token
        self.timeout_s = float(timeout_s)
        self._transport = _URLTransport(_safe_urlopen if opener is urlopen else opener)

    def describe(self) -> dict[str, Any]:
        """Return capabilities and route metadata without touching a device."""

        return self._request("GET", "/v1/describe")

    def observe(
        self,
        *,
        runtime_id: str | None = None,
        observation_id: str | None = None,
        include_robot: bool = True,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
    ) -> Observation:
        """Read one retained observation using the caller/session identity.

        ``max_age_ns`` and ``max_skew_ns`` are nanosecond limits.  Supplying
        ``observation_id`` asks the server to read that exact snapshot; it
        never authorizes motion.  ``include_robot`` only controls whether the
        read includes the robot state in the response.
        """

        payload = self._request(
            "GET",
            "/v1/observe",
            query={
                "runtime_id": runtime_id,
                "observation_id": observation_id,
                "include_robot": str(bool(include_robot)).lower(),
                "max_age_ns": max_age_ns,
                "max_skew_ns": max_skew_ns,
            },
        )
        value = payload.get("observation_id")
        if not isinstance(value, str) or not value:
            raise ControlTransportError("observe response has no observation_id")
        return Observation(value, payload)

    def media(
        self,
        observation_id: str,
        *,
        frame: str | None = None,
        include_data: bool = False,
        runtime_id: str | None = None,
    ) -> dict[str, Any]:
        """Read camera media attached to an observation.

        ``observation_id`` identifies the retained snapshot and ``frame``
        selects a named camera frame.  ``include_data=True`` may return frame
        bytes in the JSON response.  This is a read operation and does not
        capture, prepare, or move a device.
        """

        return self._request(
            "GET",
            f"/v1/media/{quote(_non_empty(observation_id, 'observation_id'), safe=':')}",
            query={
                "frame": frame,
                "include_data": str(bool(include_data)).lower(),
                "runtime_id": runtime_id,
            },
        )

    def propose(
        self,
        *,
        request_id: str,
        observation_id: str,
        instruction: str,
        runtime_id: str | None = None,
        timeout_s: float | None = None,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
    ) -> dict[str, Any]:
        """Generate a non-executing proposal from one retained observation.

        The model receives the server's snapshot identified by
        ``observation_id``; this method never calls ``prepare`` or an action
        endpoint.  ``timeout_s`` is in seconds, while ``max_age_ns`` and
        ``max_skew_ns`` are nanoseconds.  A proposal response is data to
        review before a separate :meth:`execute` call.
        """

        return self._request(
            "POST",
            "/v1/propose",
            body={
                "request_id": _non_empty(request_id, "request_id"),
                "observation_id": _non_empty(observation_id, "observation_id"),
                "instruction": _non_empty(instruction, "instruction"),
                "runtime_id": runtime_id,
                "max_age_ns": max_age_ns,
                "max_skew_ns": max_skew_ns,
                **({"timeout_s": timeout_s} if timeout_s is not None else {}),
            },
            request_id=request_id,
        )

    def execute(
        self,
        actions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        observation_id: str | None = None,
        runtime_id: str | None = None,
        steps: int | None = None,
        control_hz: float | None = None,
        wait: bool = True,
        timeout_s: float | None = None,
        source: str = "agent",
        parameters: Mapping[str, Any] | None = None,
        max_age_ns: int | None = None,
        max_skew_ns: int | None = None,
    ) -> dict[str, Any]:
        """Submit one bounded action segment through ``/v1/execute``.

        The action values must already use the configured binding's public
        action names and units.  This method intentionally does not map a
        6D waypoint or a bare vector into robot joints.

        ``timeout_s`` is in seconds; ``max_age_ns`` and ``max_skew_ns`` are
        nanoseconds.  With ``wait=True`` the response waits for the service's
        job result, but ``accepted``/``running`` and an unknown transport
        outcome remain non-terminal evidence.  The client never retries an
        execution and never turns this response into proof of a physical
        stop.

        ``runtime_id`` optionally names the configured runtime that should
        serve this segment, matching ``observe`` and ``propose``.
        """

        if not isinstance(actions, Mapping) and (
            isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence)
        ):
            raise TypeError("actions must be an action object or sequence of objects")
        body: dict[str, Any] = {
            "request_id": _non_empty(request_id, "request_id"),
            "actions": actions,
            "source": source,
            "observation_id": observation_id,
            "runtime_id": runtime_id,
            "steps": steps if steps is not None else (1 if isinstance(actions, Mapping) else len(actions)),
            "control_hz": control_hz,
            "wait": bool(wait),
            "timeout_s": timeout_s,
            "parameters": parameters,
            "max_age_ns": max_age_ns,
            "max_skew_ns": max_skew_ns,
        }
        return self._request("POST", "/v1/execute", body=body, request_id=request_id)

    def inspect(self, request_id: str) -> dict[str, Any]:
        """Read job status for ``request_id`` without issuing another action.

        The returned status describes Control's job state.  ``cancelled`` or
        ``stopped`` is not the same as verified physical zero motion; callers
        still need the device's feedback path for that claim.
        """

        return self._request("GET", f"/v1/jobs/{quote(_non_empty(request_id, 'request_id'), safe=':')}")

    def cancel(self, request_id: str) -> dict[str, Any]:
        """Request cancellation of a Control job identified by ``request_id``.

        Cancellation is a service/job transition and can race with execution.
        Inspect the same request ID after the response; cancellation does not
        by itself prove that a device has physically stopped.
        """

        request_id = _non_empty(request_id, "request_id")
        return self._request(
            "POST",
            f"/v1/jobs/{quote(request_id, safe=':')}/cancel",
            body={},
            request_id=request_id,
        )

    def stop(self, request_id: str) -> dict[str, Any]:
        """Request the existing job-scoped stop operation.

        This is a control-plane request, not a physical all-stop guarantee.
        Use the returned request identity and subsequent feedback/inspection
        to establish the actual terminal state.
        """

        request_id = _non_empty(request_id, "request_id")
        return self._request(
            "POST",
            f"/v1/jobs/{quote(request_id, safe=':')}/stop",
            body={},
            request_id=request_id,
        )

    def manual_acquire(self) -> dict[str, Any]:
        """Acquire the service's manual-control lease for this identity.

        Lease ownership is control-plane state; acquiring it does not move a
        device or prove that an existing motion has stopped.
        """

        return self._request("POST", "/v1/control/manual/acquire", body={})

    def manual_release(self) -> dict[str, Any]:
        """Release this caller's manual-control lease without moving a device."""

        return self._request("POST", "/v1/control/manual/release", body={})

    def manual_deadman(self, active: bool) -> dict[str, Any]:
        """Set this identity's manual deadman state.

        ``active`` is a boolean control-plane flag.  The response reports the
        service's lease/deadman state and is not physical motion evidence.
        """

        if not isinstance(active, bool):
            raise TypeError("active must be a boolean")
        return self._request("POST", "/v1/control/manual/deadman", body={"active": active})

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return shared_request_json(
                self._transport,
                method,
                self.endpoint + _query_path(path, query),
                body,
                timeout_s=self.timeout_s,
                headers={
                    "Accept": "application/json",
                    "X-EmbodiRun-Caller-ID": self.caller_id,
                    "X-EmbodiRun-Session-ID": self.session_id,
                    **({"X-EmbodiRun-Token": self.token} if self.token is not None else {}),
                },
                request_id=request_id,
                accepted_statuses=frozenset(range(200, 300)),
            )
        except ControlClientError as error:
            if error.status is None:
                raise ControlTransportError(str(error), request_id=request_id) from error
            error_type = (
                UnsupportedOperation
                if isinstance(error.payload, Mapping) and error.payload.get("code") == "unsupported"
                else ControlHTTPError
            )
            raise error_type(
                error.status,
                error.payload if isinstance(error.payload, Mapping) else {},
                request_id=request_id,
            ) from error
        except (ValueError, UnicodeError) as error:
            raise ControlTransportError(str(error), request_id=request_id) from error


__all__ = [
    "AgentClientError",
    "ControlClient",
    "ControlHTTPError",
    "ControlTransportError",
    "Observation",
    "UnsupportedOperation",
]


class _URLTransport:
    """stdlib URL transport implementing the shared JSONTransport protocol."""

    def __init__(self, opener: Any) -> None:
        self.opener = opener

    def request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        data = (
            json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8") if payload is not None else None
        )
        request_headers = {"Accept": "application/json", **dict(headers or {})}
        if data is not None:
            request_headers.setdefault("Content-Type", "application/json")
        request = Request(url, data=data, headers=request_headers, method=method)
        try:
            response = self.opener(request, timeout=timeout_s)
        except UrllibHTTPError as error:
            response = error
        try:
            raw = response.read()
            status = int(response.status)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        decoded = json.loads(raw.decode("utf-8")) if raw else {}
        return SimpleNamespace(status=status, payload=decoded)


class _RejectRedirects(HTTPRedirectHandler):
    """Never forward caller identity or tokens through an HTTP redirect."""

    def redirect_request(self, request, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ControlTransportError("Control endpoint returned an unexpected redirect")


_SAFE_OPENER = build_opener(_RejectRedirects)


def _safe_urlopen(request: Request, *, timeout: float):
    return _SAFE_OPENER.open(request, timeout=timeout)
