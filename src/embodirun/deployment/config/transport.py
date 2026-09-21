"""Transport listener configuration, independent of the inference backend."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .server import ServerConfig, parse_server
from .validation import ConfigError, mapping


@dataclass(frozen=True, slots=True)
class InferenceClientConfig:
    server: ServerConfig
    transport_options: dict[str, Any] = field(default_factory=dict)


def parse_inference_client(value: Any, context: str) -> InferenceClientConfig:
    raw = mapping(value, context)
    return InferenceClientConfig(
        server=parse_server(
            {key: item for key, item in raw.items() if key != "transport_options"},
            context,
        ),
        transport_options=parse_transport_options(raw.get("transport_options", {}), f"{context}.transport_options"),
    )


def parse_transport_options(value: Any, context: str) -> dict[str, Any]:
    """Preserve native CommConfig options; the endpoint validates their schema."""

    options = mapping(value, context)
    try:
        json.dumps(options, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{context} must contain JSON-compatible values") from error
    return dict(options)
