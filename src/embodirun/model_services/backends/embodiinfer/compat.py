"""Use the current HTTP client with the schemas advertised by frozen policy servers."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ...protocols.http import HttpResponse, HttpTransport, UrllibHttpTransport
from .http import EmbodiInferHttpClient


def _legacy_metadata(raw: bytes, kind: str) -> bytes:
    metadata = json.loads(raw)
    if metadata.get("schema") != f"embodiinfer.policy.{kind}.v1":
        raise ValueError("unexpected outgoing policy schema")
    metadata["schema"] = f"vvla.policy.{kind}.v1"
    return json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")


class _SchemaTransport:
    def __init__(self, transport: HttpTransport) -> None:
        self.transport = transport
        self.legacy = False

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
        if self.legacy and method == "POST" and body is not None:
            if url.endswith("/v1/sessions"):
                body = _legacy_metadata(body, "session")
            elif url.endswith("/steps"):
                # The client puts JSON metadata first. Preserve all following
                # multipart bytes, including images and their headers, exactly.
                header, separator, remainder = body.partition(b"\r\n\r\n")
                metadata, boundary, images = remainder.partition(b"\r\n--")
                if not separator or not boundary or b'name="metadata"' not in header:
                    raise ValueError("unexpected outgoing policy multipart layout")
                body = header + separator + _legacy_metadata(metadata, "step") + boundary + images
        return self.transport.request(
            method, url, headers=headers, body=body, timeout_s=timeout_s, maximum_bytes=maximum_bytes
        )


def connect_client(
    endpoint: str, token: str | None, *, timeout_s: float = 5.0, transport: HttpTransport | None = None
) -> EmbodiInferHttpClient:
    """Check health and select the advertised v1 schema before opening any session.

    Frozen pre-rename inference revisions advertise ``vvla.policy.capabilities.v1``.
    Only their outgoing schema fields are translated; observations, image bytes,
    response validation and the current runtime's HTTP client stay unchanged.
    """
    wire = _SchemaTransport(transport if transport is not None else UrllibHttpTransport())
    client = EmbodiInferHttpClient(endpoint, token=token, timeout_s=timeout_s, transport=wire)
    client.health()
    schema = client.capabilities().get("schema")
    if schema not in ("embodiinfer.policy.capabilities.v1", "vvla.policy.capabilities.v1"):
        raise ValueError(f"unsupported policy service capabilities schema: {schema!r}")
    wire.legacy = schema == "vvla.policy.capabilities.v1"
    return client
