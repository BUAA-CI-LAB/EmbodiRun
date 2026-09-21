"""Executor interface and default implementation selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from ..config import NodeConfig
from .command import Command, CommandResult
from .local import LocalExecutor
from .response import JsonHttpResponse
from .ssh import SshExecutor


class Executor(Protocol):
    def run(self, command: Command, *, check: bool = True) -> CommandResult: ...

    def get_json(self, url: str, *, timeout_s: float) -> JsonHttpResponse: ...

    def request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> JsonHttpResponse: ...

    def read_bytes(self, path: str) -> bytes: ...

    def replace_symlink(self, path: str, target: str) -> None: ...

    def write_bytes(self, path: str, content: bytes, *, mode: int = 0o600) -> None: ...

    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None: ...

    def close(self) -> None: ...


def executor_for(node: NodeConfig) -> Executor:
    if node.connection.kind == "local":
        return LocalExecutor()
    return SshExecutor(node.connection)


__all__ = ["Executor", "executor_for"]
