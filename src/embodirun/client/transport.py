"""Shared JSON request, status, and identity semantics for Control transports.

The public URL client and the Host executor client use different transport
implementations, but both pass through the same response-object validation,
accepted-status handling, request-ID matching, and unknown-outcome rule.
Keeping these route semantics here avoids making the public SDK import Host or
its optional SSH dependencies. This module does not add retries: after an
execution transport failure, callers inspect the original request ID.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class JSONTransport(Protocol):
    """Minimal URL or executor transport used by shared route semantics.

    Implementations provide transport only; :func:`request_json` owns common
    HTTP status, JSON-object, and request-ID checks.
    """

    def request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> Any: ...


class ControlClientError(RuntimeError):
    """A Control request failed or returned an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        payload: Any = None,
        request_id: str | None = None,
        unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.status_code = status
        self.payload = payload
        self.request_id = request_id
        self.unknown = unknown


def response_detail(payload: Any) -> str:
    if isinstance(payload, Mapping):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "invalid error response"


def response_unknown(payload: Any) -> bool:
    return isinstance(payload, Mapping) and payload.get("status") in {
        "unknown",
        "uncertain",
    }


def request_json(
    transport: JSONTransport,
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    *,
    timeout_s: float,
    headers: Mapping[str, str] | None = None,
    request_id: str | None = None,
    accepted_statuses: set[int] | frozenset[int] = frozenset({200}),
) -> dict[str, Any]:
    """Exchange one JSON request and apply shared status/identity semantics.

    Network failures are marked ``unknown`` when a request ID is available.
    The function accepts only the caller-specified status set and rejects a
    response whose request ID conflicts with the request being inspected.
    """

    try:
        response = transport.request_json(
            method,
            url,
            payload,
            timeout_s=float(timeout_s),
            headers=headers,
        )
    except (OSError, RuntimeError, TimeoutError) as error:
        raise ControlClientError(
            f"control request outcome is unknown: {error}",
            request_id=request_id,
            unknown=True,
        ) from error
    status = getattr(response, "status", None)
    body = getattr(response, "payload", None)
    if not isinstance(status, int):
        raise ControlClientError(
            "control response has no HTTP status",
            payload=body,
            request_id=request_id,
        )
    if status not in accepted_statuses:
        raise ControlClientError(
            f"control service returned HTTP {status}: {response_detail(body)}",
            status=status,
            payload=body,
            request_id=request_id,
            unknown=response_unknown(body),
        )
    if not isinstance(body, Mapping):
        raise ControlClientError(
            "control service response must be a JSON object",
            status=status,
            payload=body,
            request_id=request_id,
        )
    result = dict(body)
    returned_request_id = result.get("request_id")
    if request_id is not None and returned_request_id is not None and returned_request_id != request_id:
        raise ControlClientError(
            "control response request_id does not match",
            status=status,
            payload=result,
            request_id=request_id,
        )
    return result


__all__ = [
    "ControlClientError",
    "JSONTransport",
    "request_json",
    "response_detail",
    "response_unknown",
]
