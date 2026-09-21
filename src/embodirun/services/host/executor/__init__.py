"""Compatibility exports for :mod:`embodirun.deployment.executor`."""

from embodirun.deployment.executor import (
    Command,
    CommandError,
    CommandResult,
    Executor,
    JsonHttpResponse,
    LocalExecutor,
    SshExecutor,
    executor_for,
    render_posix,
)

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
