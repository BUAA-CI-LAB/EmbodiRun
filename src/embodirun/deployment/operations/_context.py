"""Protocols accepted by deployment operations, independent of the CLI."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from embodirun.deployment.config import DeploymentConfig
from embodirun.deployment.executor import Executor
from embodirun.deployment.plan import DeploymentPlan


class Progress(Protocol):
    """Thread-safe progress callbacks used by per-node worker threads.

    Operations invoke node callbacks concurrently, so implementations must
    serialize renderer/state updates and tolerate callbacks arriving in any
    completion order.
    """

    def begin(self, operation: str, deployment: str) -> None: ...
    def add_node(self, node_id: str, *, total: int) -> None: ...
    def update(self, node_id: str, message: str, *, detail: str | None = None) -> None: ...
    def advance(self, node_id: str) -> None: ...
    def fail(self, node_id: str, error: Exception) -> None: ...
    def succeed(self, node_id: str, *, detail: str | None = None) -> None: ...
    def finish(self, *, success: bool) -> None: ...
    def message(self, text: str) -> None: ...


class DeploymentContext(Protocol):
    """Shared immutable configuration and per-node executor factory.

    An operation may request one executor context per node; the implementation
    owns closing each context after its node work finishes.
    """

    config: DeploymentConfig
    deployment: DeploymentPlan
    state_path: Path
    progress: Progress

    def executor(self, node_id: str) -> AbstractContextManager[Executor]: ...


__all__ = ["DeploymentContext", "Progress"]
