"""Shared TCP listener configuration for model and control services."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import ConfigError, integer, mapping, reject_unknown, string


@dataclass(frozen=True, slots=True)
class ServerConfig:
    bind: str
    port: int


def parse_server(value: object, context: str) -> ServerConfig:
    """Parse one bind address and validated TCP port."""

    server = mapping(value, context)
    reject_unknown(server, {"bind", "port"}, context)
    result = ServerConfig(
        bind=string(server, "bind", context),
        port=integer(server.get("port"), f"{context}.port"),
    )
    if not 1 <= result.port <= 65535:
        raise ConfigError(f"{context}.port must be between 1 and 65535")
    return result


__all__ = ["ServerConfig", "parse_server"]
