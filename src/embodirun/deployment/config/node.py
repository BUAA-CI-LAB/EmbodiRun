"""Deployment node configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .connection import ConnectionConfig, parse_connection
from .validation import mapping, optional_string, string


@dataclass(frozen=True, slots=True)
class NodeConfig:
    node_id: str
    kind: str
    connection: ConnectionConfig
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    address: str | None = None


def parse_node(node_id: str, value: dict[str, Any]) -> NodeConfig:
    connection = parse_connection(mapping(value.get("connection"), f"nodes.{node_id}.connection"))
    return NodeConfig(
        node_id=node_id,
        kind=string(value, "type", f"nodes.{node_id}"),
        connection=connection,
        options=dict(value),
        address=optional_string(value, "address", f"nodes.{node_id}"),
    )


__all__ = ["NodeConfig"]
