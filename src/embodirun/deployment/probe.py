"""Probe base capabilities on a local or remote deployment node."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from .environment import EnvironmentError
from .executor import Command, Executor


@dataclass(frozen=True, slots=True)
class NodeProbe:
    """Base capabilities discovered before initializing a node."""

    home: str
    platform: str
    machine: str
    python: str | None
    python_version: str | None
    git: str | None
    uv: str | None


def probe_node(executor: Executor, *, require_init_tools: bool = False) -> NodeProbe:
    """Probe connectivity and report optional tools without changing the node."""

    home = _required_output(executor, ("printenv", "HOME"), "home directory")
    platform = _required_output(executor, ("uname", "-s"), "platform")
    machine = _required_output(executor, ("uname", "-m"), "machine")
    if not home.startswith("/") or not platform or not machine:
        raise EnvironmentError("node probe omitted required machine information")
    path = _optional_output(executor, ("printenv", "PATH"))
    python = _find_executable(executor, "python3", home=home, path=path)
    git = _find_executable(executor, "git", home=home, path=path)
    uv = _find_executable(executor, "uv", home=home, path=path)
    python_version = _python_version(executor, python) if python is not None else None
    probe = NodeProbe(
        home=home,
        platform=platform,
        machine=machine,
        python=python,
        python_version=python_version,
        git=git,
        uv=uv,
    )
    if require_init_tools:
        missing = [name for name in ("python", "git", "uv") if getattr(probe, name) is None]
        if missing:
            raise EnvironmentError("node is missing tools required by init: " + ", ".join(missing))
        if probe.python_version is None:
            raise EnvironmentError("node Python could not report its version")
    return probe


def _required_output(
    executor: Executor,
    argv: tuple[str, ...],
    description: str,
) -> str:
    result = executor.run(Command(argv, timeout_s=20.0), check=False)
    value = result.stdout.strip()
    if result.exit_code != 0 or not value:
        raise EnvironmentError(f"node did not report its {description}")
    return value


def _optional_output(executor: Executor, argv: tuple[str, ...]) -> str:
    result = executor.run(Command(argv, timeout_s=20.0), check=False)
    return result.stdout.strip() if result.exit_code == 0 else ""


def _find_executable(
    executor: Executor,
    name: str,
    *,
    home: str,
    path: str,
) -> str | None:
    directories = [item for item in path.split(":") if item.startswith("/")]
    directories.extend(
        (
            posixpath.join(home, ".local", "bin"),
            posixpath.join(home, ".cargo", "bin"),
        )
    )
    for directory in dict.fromkeys(directories):
        candidate = posixpath.join(directory, name)
        result = executor.run(
            Command(("test", "-x", candidate), timeout_s=20.0),
            check=False,
        )
        if result.exit_code == 0:
            return candidate
    return None


def _python_version(executor: Executor, python: str) -> str | None:
    result = executor.run(
        Command((python, "--version"), timeout_s=20.0),
        check=False,
    )
    if result.exit_code != 0:
        return None
    output = (result.stdout or result.stderr).strip()
    prefix = "Python "
    return output[len(prefix) :] if output.startswith(prefix) else None


__all__ = ["NodeProbe", "probe_node"]
