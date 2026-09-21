"""Thread-safe, human-readable progress for deployment commands."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


@dataclass(slots=True)
class _NodeProgress:
    total: int
    completed: int = 0
    status: str = "pending"
    stage: str = "Pending"


class ProgressReporter(Protocol):
    """Report node-level progress without coupling commands to a renderer."""

    def begin(self, command: str, deployment: str) -> None: ...

    def add_node(self, node_id: str, *, total: int) -> None: ...

    def update(self, node_id: str, stage: str, *, detail: str | None = None) -> None: ...

    def advance(self, node_id: str) -> None: ...

    def succeed(self, node_id: str, *, detail: str = "Ready") -> None: ...

    def fail(self, node_id: str, error: object) -> None: ...

    def finish(self, *, success: bool) -> None: ...

    def message(self, value: str) -> None: ...


class _StatusColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        status = task.fields.get("status", "pending")
        if status == "succeeded":
            return Text("✓", style="bold green")
        if status == "failed":
            return Text("✗", style="bold red")
        if status == "running":
            return Text("●", style="bold cyan")
        return Text("○", style="dim")


class ConsoleProgressReporter:
    """Render live progress on a terminal and readable lines elsewhere."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._interactive = bool(getattr(self._stream, "isatty", lambda: False)())
        self._console = Console(
            file=self._stream,
            force_terminal=self._interactive,
            no_color=not self._interactive,
        )
        self._lock = RLock()
        self._command: str | None = None
        self._nodes: dict[str, _NodeProgress] = {}
        self._progress: Progress | None = None
        self._rich_tasks: dict[str, int] = {}

    def begin(self, command: str, deployment: str) -> None:
        with self._lock:
            if self._command is not None:
                raise RuntimeError("a progress operation is already active")
            self._command = command
            self._nodes.clear()
            self._rich_tasks.clear()
            title = f"{command.capitalize()} deployment {deployment}"
            self._console.print(Text(title, style="bold"))
            if self._interactive:
                self._progress = Progress(
                    _StatusColumn(),
                    TextColumn(
                        "{task.fields[node]}",
                        justify="left",
                        style="bold",
                        markup=False,
                    ),
                    BarColumn(bar_width=24),
                    MofNCompleteColumn(),
                    TextColumn(
                        "{task.fields[stage]}",
                        justify="left",
                        markup=False,
                    ),
                    TimeElapsedColumn(),
                    console=self._console,
                    refresh_per_second=8,
                )
                self._progress.start()

    def add_node(self, node_id: str, *, total: int) -> None:
        if total <= 0:
            raise ValueError("node progress total must be positive")
        with self._lock:
            self._require_active()
            if node_id in self._nodes:
                raise ValueError(f"progress already contains node {node_id!r}")
            self._nodes[node_id] = _NodeProgress(total=total)
            if self._progress is not None:
                self._rich_tasks[node_id] = self._progress.add_task(
                    "",
                    total=total,
                    node=node_id,
                    stage="Pending",
                    status="pending",
                )

    def update(
        self,
        node_id: str,
        stage: str,
        *,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            node = self._node(node_id)
            node.status = "running"
            node.stage = _stage_text(stage, detail)
            if self._progress is not None:
                self._progress.update(
                    self._rich_tasks[node_id],
                    stage=node.stage,
                    status=node.status,
                )
            else:
                self._console.print(f"  ● {node_id}: {node.stage}", markup=False)

    def advance(self, node_id: str) -> None:
        with self._lock:
            node = self._node(node_id)
            node.completed = min(node.completed + 1, node.total)
            if self._progress is not None:
                self._progress.update(
                    self._rich_tasks[node_id],
                    completed=node.completed,
                )

    def succeed(self, node_id: str, *, detail: str = "Ready") -> None:
        with self._lock:
            node = self._node(node_id)
            node.completed = node.total
            node.status = "succeeded"
            node.stage = detail
            if self._progress is not None:
                self._progress.update(
                    self._rich_tasks[node_id],
                    completed=node.total,
                    stage=detail,
                    status=node.status,
                )
            else:
                self._console.print(f"  ✓ {node_id}: {detail}", markup=False)

    def fail(self, node_id: str, error: object) -> None:
        with self._lock:
            node = self._node(node_id)
            node.status = "failed"
            node.stage = f"Failed: {error}"
            if self._progress is not None:
                self._progress.update(
                    self._rich_tasks[node_id],
                    stage=node.stage,
                    status=node.status,
                )
            else:
                self._console.print(f"  ✗ {node_id}: {node.stage}", markup=False)

    def finish(self, *, success: bool) -> None:
        with self._lock:
            if self._command is None:
                return
            if self._progress is not None:
                self._progress.stop()
                self._progress = None
            ready = sum(node.status == "succeeded" for node in self._nodes.values())
            total = len(self._nodes)
            marker = "✓" if success else "✗"
            outcome = "completed" if success else "finished with errors"
            self._console.print(
                f"{marker} {self._command} {outcome}: {ready}/{total} nodes ready",
                markup=False,
            )
            self._command = None

    def message(self, value: str) -> None:
        with self._lock:
            self._console.print(value, markup=False)

    def _require_active(self) -> None:
        if self._command is None:
            raise RuntimeError("no progress operation is active")

    def _node(self, node_id: str) -> _NodeProgress:
        self._require_active()
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown progress node {node_id!r}") from None


def _stage_text(stage: str, detail: str | None) -> str:
    return stage if detail is None else f"{stage} · {detail}"


__all__ = ["ConsoleProgressReporter", "ProgressReporter"]
