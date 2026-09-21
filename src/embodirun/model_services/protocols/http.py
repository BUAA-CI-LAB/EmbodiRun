"""Bounded HTTP request and response protocol implementation."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class HttpTransportError(RuntimeError):
    """An HTTP exchange failed before backend-specific decoding."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        maximum_bytes: int,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        maximum_bytes: int,
    ) -> HttpResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = response.read(maximum_bytes + 1)
                result = HttpResponse(
                    status=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=payload,
                )
        except urllib.error.HTTPError as error:
            detail = error.read(min(maximum_bytes, 16 * 1024))
            error.close()
            raise HttpTransportError(
                f"{method} {url} returned HTTP {error.code}: {detail.decode('utf-8', errors='replace')}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise HttpTransportError(f"{method} {url} failed: {error}") from error
        if len(result.body) > maximum_bytes:
            raise HttpTransportError(f"{method} {url} response exceeded {maximum_bytes} bytes")
        if not 200 <= result.status < 300:
            raise HttpTransportError(f"{method} {url} returned HTTP {result.status}")
        return result


__all__ = [
    "HttpResponse",
    "HttpTransport",
    "HttpTransportError",
    "UrllibHttpTransport",
]
