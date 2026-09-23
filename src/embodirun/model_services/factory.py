"""Create inference clients through the centralized provider registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .client import InferenceClient
from .providers import provider


def build_inference_client(
    protocol: str,
    endpoint: str,
    options: Mapping[str, Any],
    *,
    backend: str = "embodiinfer",
    timeout_s: float,
) -> InferenceClient:
    """Build a registered provider client without exposing its implementation."""

    selected = provider(backend)
    if not selected.action_capable:
        raise ValueError(f"inference provider {backend!r} has no action capability")
    if not selected.supports(protocol):
        raise ValueError(f"inference provider {backend!r} does not support {protocol!r} transport")
    merged = dict(options)
    merged.setdefault("transport", protocol)
    return selected.client_builder(endpoint, merged, timeout_s)


__all__ = ["build_inference_client"]
