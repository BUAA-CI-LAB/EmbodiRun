"""Shared dependencies supplied to deployment subcommands."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from embodirun.deployment.config import DeploymentConfig, NodeConfig
from embodirun.deployment.executor import Executor
from embodirun.deployment.plan import DeploymentPlan

from .progress import ProgressReporter


@dataclass(frozen=True, slots=True)
class CommandContext:
    config: DeploymentConfig
    deployment: DeploymentPlan
    state_path: Path
    executor_factory: Callable[[NodeConfig], Executor]
    progress: ProgressReporter

    @contextmanager
    def executor(self, node_id: str) -> Iterator[Executor]:
        """Give one operation exclusive ownership of one node executor."""

        executor = self.executor_factory(self.config.nodes[node_id])
        try:
            yield executor
        finally:
            executor.close()


def state_path(directory: Path | None, deployment_name: str) -> Path:
    root = Path.home() / ".local" / "state" / "embodirun" if directory is None else directory.expanduser()
    return root / f"{deployment_name}.json"


__all__ = [
    "CommandContext",
    "state_path",
]
