"""Transport-independent command values and execution errors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class Command:
    """One argv-based command with explicit working directory and environment."""

    argv: tuple[str, ...]
    cwd: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    stdin: str | None = field(default=None, repr=False)
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not self.argv or any(not argument or "\x00" in argument for argument in self.argv):
            raise ValueError("command arguments must be non-empty and cannot contain NUL")
        if self.cwd is not None and (not self.cwd or "\x00" in self.cwd):
            raise ValueError("command cwd cannot be empty or contain NUL")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("command timeout must be positive")
        for name, value in self.environment.items():
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            if "\x00" in value:
                raise ValueError(f"environment variable {name!r} cannot contain NUL")


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class CommandError(RuntimeError):
    """A local or remote command returned a non-zero exit code."""

    def __init__(self, exit_code: int, stderr: str = "") -> None:
        detail = stderr.strip()
        message = f"command failed with exit code {exit_code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.exit_code = exit_code


__all__ = ["Command", "CommandError", "CommandResult"]
