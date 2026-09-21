"""Bounded standard-library HTTP client for small JSON control planes."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class HttpClientError(RuntimeError):
    """A bounded HTTP request or response failed validation."""


@dataclass(frozen=True, slots=True)
class Response:
    body: bytes
    headers: Mapping[str, str]
    status: int


class JsonHttpClient:
    """Minimal synchronous HTTP client with auth, timeouts, and size limits."""

    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 2.0) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = float(timeout_s)

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        maximum_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        accept: str = "application/json",
    ) -> Response:
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        if maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive")
        data = None
        headers = {"Accept": accept}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            try:
                data = json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise HttpClientError("request payload is not valid JSON") from error
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read(maximum_bytes + 1)
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                status = int(response.status)
        except urllib.error.HTTPError as error:
            try:
                detail = error.read(min(maximum_bytes, 16 * 1024)).decode("utf-8", errors="replace")
            finally:
                error.close()
            raise HttpClientError(f"{method.upper()} {path} returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise HttpClientError(f"{method.upper()} {path} failed: {error}") from error
        if len(body) > maximum_bytes:
            raise HttpClientError(f"{method.upper()} {path} response exceeds {maximum_bytes} bytes")
        return Response(body=body, headers=response_headers, status=status)

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        maximum_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> dict[str, Any]:
        response = self.request(method, path, payload, maximum_bytes=maximum_bytes)
        try:
            value = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HttpClientError(f"{method.upper()} {path} returned invalid JSON") from error
        if not isinstance(value, dict):
            raise HttpClientError(f"{method.upper()} {path} must return a JSON object")
        return value


__all__ = [
    "DEFAULT_MAX_RESPONSE_BYTES",
    "HttpClientError",
    "JsonHttpClient",
    "Response",
]
