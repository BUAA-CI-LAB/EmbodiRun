"""Node connection configuration and parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .validation import (
    ConfigError,
    boolean,
    integer,
    optional_string,
    positive_number,
    reject_unknown,
    string,
)

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class ConnectionConfig:
    kind: Literal["local", "ssh"]
    host: str | None = None
    port: int = 22
    username: str | None = None
    password_env: str | None = None
    identity_file: str | None = None
    accept_new_host_key: bool = False
    connect_timeout_s: float = 10.0
    command_timeout_s: float = 30.0
    proxy_command: str | None = None


def parse_connection(value: dict[str, Any]) -> ConnectionConfig:
    context = "node connection"
    kind = string(value, "type", context)
    if kind not in {"local", "ssh"}:
        raise ConfigError(f"{context}.type must be local or ssh")
    if kind == "local":
        reject_unknown(value, {"type"}, "local connection")
        return ConnectionConfig(kind="local")

    allowed = {
        "type",
        "host",
        "port",
        "username",
        "password_env",
        "identity_file",
        "accept_new_host_key",
        "connect_timeout_s",
        "command_timeout_s",
        "proxy_command",
    }
    reject_unknown(value, allowed, "SSH connection")
    password_env = optional_string(value, "password_env", context)
    if password_env is not None and not _ENVIRONMENT_NAME.fullmatch(password_env):
        raise ConfigError("node connection.password_env must be an environment variable name")
    port = integer(value.get("port", 22), f"{context}.port")
    if not 1 <= port <= 65535:
        raise ConfigError(f"{context}.port must be between 1 and 65535")
    return ConnectionConfig(
        kind="ssh",
        host=string(value, "host", context),
        port=port,
        username=string(value, "username", context),
        password_env=password_env,
        identity_file=optional_string(value, "identity_file", context),
        proxy_command=optional_string(value, "proxy_command", context),
        accept_new_host_key=boolean(
            value.get("accept_new_host_key", False),
            f"{context}.accept_new_host_key",
        ),
        connect_timeout_s=positive_number(
            value.get("connect_timeout_s", 10.0),
            f"{context}.connect_timeout_s",
        ),
        command_timeout_s=positive_number(
            value.get("command_timeout_s", 30.0),
            f"{context}.command_timeout_s",
        ),
    )


__all__ = ["ConnectionConfig"]
