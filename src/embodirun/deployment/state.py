"""Persistent, secret-free state for initialized deployment services."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

_STATE_VERSION = 1


class StateError(ValueError):
    """A deployment state file is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    environment_id: str
    node: str
    project: Literal["deploy", "inference"]
    group: str
    path: str
    status: Literal["pending", "ready", "failed"] = "pending"


@dataclass(frozen=True, slots=True)
class NodeState:
    node_id: str
    home: str
    root: str
    deploy_project: str
    inference_project: str
    platform: str
    machine: str
    python: str
    python_version: str


@dataclass(frozen=True, slots=True)
class ServiceState:
    service_id: str
    node: str
    status: Literal["stopped", "running", "failed"] = "stopped"
    pid: int | None = None
    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class DeploymentState:
    name: str
    config_digest: str
    deploy_commit: str
    # Resolved during init; absent if every node failed to initialize.
    inference_commit: str | None
    nodes: dict[str, NodeState] = field(default_factory=dict)
    environments: dict[str, EnvironmentState] = field(default_factory=dict)
    services: dict[str, ServiceState] = field(default_factory=dict)


class StateStore:
    """Read and atomically replace one local deployment state file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> DeploymentState | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StateError(f"cannot read state {self.path}: {error}") from error
        return _parse_state(payload)

    def save(self, state: DeploymentState) -> None:
        payload = {
            "version": _STATE_VERSION,
            "name": state.name,
            "config_digest": state.config_digest,
            "deploy_commit": state.deploy_commit,
            "inference_commit": state.inference_commit,
            "nodes": {name: asdict(value) for name, value in sorted(state.nodes.items())},
            "environments": {name: asdict(value) for name, value in sorted(state.environments.items())},
            "services": {name: asdict(value) for name, value in sorted(state.services.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise StateError(f"cannot write state {self.path}: {error}") from error


def _parse_state(value: Any) -> DeploymentState:
    root = _mapping(value, "state")
    version = root.get("version")
    if version != _STATE_VERSION:
        raise StateError(f"unsupported state version: {version!r}")
    nodes = {name: _node_state(name, item) for name, item in _mapping(root.get("nodes", {}), "nodes").items()}
    environments = {
        name: _environment_state(name, item)
        for name, item in _mapping(root.get("environments", {}), "environments").items()
    }
    services = {
        name: _service_state(name, item) for name, item in _mapping(root.get("services", {}), "services").items()
    }
    return DeploymentState(
        name=_string(root, "name", "state"),
        config_digest=_string(root, "config_digest", "state"),
        deploy_commit=_string(root, "deploy_commit", "state"),
        inference_commit=(
            _string(root, "inference_commit", "state") if root.get("inference_commit") is not None else None
        ),
        nodes=nodes,
        environments=environments,
        services=services,
    )


def _environment_state(name: str, value: Any) -> EnvironmentState:
    item = _mapping(value, f"environments.{name}")
    status = item.get("status", "pending")
    if status not in {"pending", "ready", "failed"}:
        raise StateError(f"environments.{name}.status is invalid")
    state = EnvironmentState(
        environment_id=_string(item, "environment_id", f"environments.{name}"),
        node=_string(item, "node", f"environments.{name}"),
        project=_choice(
            item,
            "project",
            {"deploy", "inference"},
            f"environments.{name}",
        ),
        group=_string(item, "group", f"environments.{name}"),
        path=_string(item, "path", f"environments.{name}"),
        status=status,
    )
    if name != state.environment_id:
        raise StateError(f"environment key {name!r} does not match environment_id")
    return state


def _node_state(name: str, value: Any) -> NodeState:
    item = _mapping(value, f"nodes.{name}")
    state = NodeState(
        node_id=_string(item, "node_id", f"nodes.{name}"),
        home=_string(item, "home", f"nodes.{name}"),
        root=_string(item, "root", f"nodes.{name}"),
        deploy_project=_string(item, "deploy_project", f"nodes.{name}"),
        inference_project=_string(item, "inference_project", f"nodes.{name}"),
        platform=_string(item, "platform", f"nodes.{name}"),
        machine=_string(item, "machine", f"nodes.{name}"),
        python=_string(item, "python", f"nodes.{name}"),
        python_version=_string(item, "python_version", f"nodes.{name}"),
    )
    if name != state.node_id:
        raise StateError(f"node key {name!r} does not match node_id")
    return state


def _service_state(name: str, value: Any) -> ServiceState:
    item = _mapping(value, f"services.{name}")
    status = item.get("status", "stopped")
    if status not in {"stopped", "running", "failed"}:
        raise StateError(f"services.{name}.status is invalid")
    pid = item.get("pid")
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0):
        raise StateError(f"services.{name}.pid must be a positive integer")
    endpoint = item.get("endpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint):
        raise StateError(f"services.{name}.endpoint must be a non-empty string")
    state = ServiceState(
        service_id=_string(item, "service_id", f"services.{name}"),
        node=_string(item, "node", f"services.{name}"),
        status=status,
        pid=pid,
        endpoint=endpoint,
    )
    if name != state.service_id:
        raise StateError(f"service key {name!r} does not match service_id")
    return state


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StateError(f"{context} must be an object with string keys")
    return value


def _string(value: dict[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise StateError(f"{context}.{key} must be a non-empty string")
    return result


def _choice(value: dict[str, Any], key: str, choices: set[str], context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or result not in choices:
        raise StateError(f"{context}.{key} must be one of {', '.join(sorted(choices))}")
    return result


__all__ = [
    "DeploymentState",
    "EnvironmentState",
    "NodeState",
    "ServiceState",
    "StateError",
    "StateStore",
]
