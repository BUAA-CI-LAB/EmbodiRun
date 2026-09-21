"""Canonical local and SSH command execution primitives for deployment."""

from .command import Command, CommandError, CommandResult
from .executor import Executor, executor_for
from .local import LocalExecutor
from .response import JsonHttpResponse
from .ssh import SshExecutor, render_posix

__all__ = [
    "Command",
    "CommandError",
    "CommandResult",
    "Executor",
    "JsonHttpResponse",
    "LocalExecutor",
    "SshExecutor",
    "executor_for",
    "render_posix",
]
