"""Strict JSON configuration shared by the owner's control and camera components."""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .camera.settings import CameraServiceConfig
from .control.config import ControlConfig


def is_loopback_bind(host: str) -> bool:
    candidate = host.strip().lower()
    if candidate == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def validate_bind_token(host: str, token: Optional[str]) -> None:  # noqa: UP045
    """Require a strong bearer token when exposing the service off-host."""

    if token is not None:
        try:
            token.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("owner token must contain only ASCII characters") from error
        if len(token) < 32:
            raise ValueError("owner token must contain at least 32 characters")
        if token.strip() != token or any(character.isspace() for character in token):
            raise ValueError("owner token must not contain whitespace")
    if not is_loopback_bind(host) and token is None:
        raise ValueError("owner token is required when binding beyond localhost")


@dataclass(frozen=True)
class OwnerConfig:
    """One endpoint and credential; operator readiness is a process flag only."""

    host: str
    port: int
    token: str | None
    control: ControlConfig
    camera: CameraServiceConfig | None

    @classmethod
    def load(cls, path: Path, *, operator_ready: bool = False) -> OwnerConfig:
        """Resolve configuration without opening SDKs, sockets or camera devices."""
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) - {"server", "control", "camera"}:
            raise ValueError("owner configuration accepts only server, control and camera")
        if not isinstance(value.get("server", {}), dict) or not isinstance(value.get("control", {}), dict):
            raise ValueError("server and control must be objects")
        server = dict(value.get("server", {}))
        if set(server) - {"host", "port", "token_env"}:
            raise ValueError("unknown owner server option")
        host, port = server.get("host", "127.0.0.1"), server.get("port", 8080)
        if not isinstance(host, str) or not host.strip():
            raise ValueError("server.host must be a non-empty string")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("server.port must be an integer between 1 and 65535")
        token_name = server.get("token_env", "GO2_API_TOKEN")
        if not isinstance(token_name, str) or not token_name:
            raise ValueError("server.token_env must name an environment variable")
        token = os.environ.get(token_name)
        validate_bind_token(host, token)
        control = dict(value.get("control", {}))
        allowed = {"mode", "interface", "state_topic", "cyclonedds_lib_dir", "rpc_timeout_s"}
        if set(control) - allowed:
            raise ValueError("unknown control option; readiness must be supplied as --operator-ready")
        control_config = ControlConfig(
            mode=control.get("mode", "dry-run"),
            interface=control.get("interface"),
            state_topic=control.get("state_topic", "rt/lf/sportmodestate"),
            cyclonedds_lib_dir=control.get("cyclonedds_lib_dir"),
            rpc_timeout_s=control.get("rpc_timeout_s", 2.0),
            operator_ready=operator_ready,
        )
        camera = value.get("camera")
        camera_config = None
        if camera is not None:
            camera_config = CameraServiceConfig.from_mapping(camera)
        return cls(host, port, token, control_config, camera_config)
