"""Supervise node processes through a host facade and this file's node entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import posixpath
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

# The node supervisor is also uploaded and invoked as a bare script by older
# ProjectManager releases.  Make that form independent of the caller's cwd and
# PYTHONPATH while keeping normal package imports canonical.
if __package__ in {None, ""}:  # pragma: no cover - exercised by subprocess tests
    _SOURCE_ROOT = Path(__file__).resolve().parents[2]
    if str(_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_ROOT))
    __package__ = "embodirun.deployment"

if TYPE_CHECKING:
    from .executor import Command, Executor
    from .plan import ServiceSpec

_SERVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_IDENTITY_ENVIRONMENT = "RLINF_DEPLOY_SERVICE_ID"
_REQUEST_LIMIT_BYTES = 1024 * 1024


class SupervisorError(ValueError):
    """A process request is unsafe or the managed process state is invalid."""


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    """Observed state of one supervised service process."""

    state: Literal["stopped", "running", "stale", "pid-reused"]
    pid: int | None = None
    log: str | None = None


class ServiceSupervisor:
    """Invoke the checked process lifecycle entrypoint on one managed node."""

    def __init__(
        self,
        executor: Executor,
        *,
        python: str,
        agent_path: str,
        run_root: str,
        log_root: str,
    ) -> None:
        if not python or "\x00" in python:
            raise ValueError("python must be a valid executable path")
        for name, value in (
            ("agent_path", agent_path),
            ("run_root", run_root),
            ("log_root", log_root),
        ):
            if not value or value == "/" or "\x00" in value:
                raise ValueError(f"{name} must be a non-root POSIX path")
        self.executor = executor
        self.python = python
        self.agent_path = agent_path
        self.run_root = run_root
        self.log_root = log_root

    def start(self, service: ServiceSpec) -> ProcessStatus:
        """Start a service idempotently and verify its process identity."""

        _validate_service_id(service.service_id)
        environment = dict(service.command.environment)
        existing = environment.get(_IDENTITY_ENVIRONMENT)
        if existing is not None and existing != service.service_id:
            raise SupervisorError(f"service command overrides reserved variable {_IDENTITY_ENVIRONMENT}")
        environment[_IDENTITY_ENVIRONMENT] = service.service_id
        request = json.dumps(
            {
                "argv": service.command.argv,
                "cwd": service.command.cwd,
                "environment": environment,
            },
            allow_nan=False,
            separators=(",", ":"),
        )
        result = self.executor.run(self._command("start", service.service_id, stdin=f"{request}\n"))
        return _parse_status(result.stdout, log=self._log_path(service.service_id))

    def status(self, service_id: str) -> ProcessStatus:
        """Inspect the PID file without changing the managed process."""

        result = self.executor.run(self._command("status", service_id))
        return _parse_status(result.stdout, log=self._log_path(service_id))

    def stop(self, service_id: str) -> ProcessStatus:
        """Stop only the process carrying this service's identity marker."""

        result = self.executor.run(self._command("stop", service_id, timeout_s=10.0))
        return _parse_status(result.stdout, log=self._log_path(service_id))

    def _command(
        self,
        action: str,
        service_id: str,
        *,
        stdin: str | None = None,
        timeout_s: float | None = None,
    ) -> Command:
        from .executor import Command

        _validate_service_id(service_id)
        return Command(
            (
                self.python,
                self.agent_path,
                action,
                "--service-id",
                service_id,
                "--run-root",
                self.run_root,
                "--log-root",
                self.log_root,
            ),
            stdin=stdin,
            timeout_s=timeout_s,
        )

    def _log_path(self, service_id: str) -> str:
        _validate_service_id(service_id)
        return posixpath.join(self.log_root, f"{service_id}.log")


def _parse_status(output: str, *, log: str) -> ProcessStatus:
    lines = [line for line in output.splitlines() if line]
    if not lines:
        raise RuntimeError("service supervisor returned no status")
    fields = lines[-1].split("\t")
    state = fields[0]
    if state not in {"stopped", "running", "stale", "pid-reused"}:
        raise RuntimeError(f"service supervisor returned invalid status {state!r}")
    pid: int | None = None
    if len(fields) > 1:
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise RuntimeError("service supervisor returned an invalid PID") from error
        if pid <= 0:
            raise RuntimeError("service supervisor returned an invalid PID")
    return ProcessStatus(state=state, pid=pid, log=log)


def _validate_service_id(service_id: str) -> None:
    if not _SERVICE_ID.fullmatch(service_id):
        raise SupervisorError(
            "service IDs must start with an alphanumeric character and contain "
            "only alphanumerics, dots, underscores, or hyphens"
        )


def _agent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodirun-supervisor")
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-root", required=True)
    return parser


def _agent_main(argv: list[str] | None = None) -> int:
    args = _agent_parser().parse_args(argv)
    try:
        _validate_service_id(args.service_id)
        run_root = Path(args.run_root)
        log_root = Path(args.log_root)
        pid_file = run_root / f"{args.service_id}.pid"
        log_file = log_root / f"{args.service_id}.log"
        if args.action == "start":
            status = _agent_start(
                args.service_id,
                pid_file=pid_file,
                log_file=log_file,
            )
        elif args.action == "status":
            status = _agent_status(args.service_id, pid_file=pid_file)
        else:
            status = _agent_stop(args.service_id, pid_file=pid_file)
    except (OSError, SupervisorError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    fields = [status.state]
    if status.pid is not None:
        fields.append(str(status.pid))
    print("\t".join(fields))
    return 0


def _agent_start(
    service_id: str,
    *,
    pid_file: Path,
    log_file: Path,
) -> ProcessStatus:
    identity_file = _identity_path(pid_file)
    current = _agent_status(service_id, pid_file=pid_file)
    if current.state == "running":
        return current
    if current.state == "pid-reused":
        raise SupervisorError("PID belongs to another process")
    pid_file.unlink(missing_ok=True)
    identity_file.unlink(missing_ok=True)

    argv, cwd, environment = _read_start_request(service_id)
    pid_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_descriptor = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_descriptor)
    identity = _process_identity(process.pid)
    if identity is None:
        _terminate_process_group(process.pid)
        raise SupervisorError("started process identity could not be read")
    _write_private_text(identity_file, f"{identity}\n")
    _write_private_text(pid_file, f"{process.pid}\n")
    time.sleep(0.2)
    status = _agent_status(service_id, pid_file=pid_file)
    if status.state != "running":
        _terminate_process_group(process.pid)
        pid_file.unlink(missing_ok=True)
        identity_file.unlink(missing_ok=True)
        raise SupervisorError("service exited immediately or its process identity could not be verified")
    return status


def _read_start_request(
    service_id: str,
) -> tuple[list[str], str | None, dict[str, str]]:
    content = sys.stdin.buffer.read(_REQUEST_LIMIT_BYTES + 1)
    if len(content) > _REQUEST_LIMIT_BYTES:
        raise SupervisorError("service start request is too large")
    try:
        request = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupervisorError("service start request is not valid JSON") from error
    if not isinstance(request, dict):
        raise SupervisorError("service start request must be an object")
    argv = request.get("argv")
    cwd = request.get("cwd")
    environment = request.get("environment")
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value for value in argv):
        raise SupervisorError("service argv must be a non-empty string array")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        raise SupervisorError("service cwd must be null or a non-empty string")
    if not isinstance(environment, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in environment.items()
    ):
        raise SupervisorError("service environment must contain string pairs")
    if environment.get(_IDENTITY_ENVIRONMENT) != service_id:
        raise SupervisorError("service start request has an invalid identity")
    process_environment = os.environ.copy()
    process_environment.update(environment)
    virtual_environment = environment.get("VIRTUAL_ENV")
    if virtual_environment:
        # Resolve PATH on the service node, not on the host running the CLI.
        process_environment["PATH"] = os.pathsep.join(
            (
                posixpath.join(virtual_environment, "bin"),
                process_environment.get("PATH", os.defpath),
            )
        )
        process_environment.pop("PYTHONHOME", None)
    return argv, cwd, process_environment


def _agent_status(service_id: str, *, pid_file: Path) -> ProcessStatus:
    if not pid_file.is_file():
        return ProcessStatus("stopped")
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError):
        return ProcessStatus("stale")
    if pid <= 0 or not _process_exists(pid):
        return ProcessStatus("stale", pid if pid > 0 else None)
    recorded_identity = _read_identity(_identity_path(pid_file))
    current_identity = _process_identity(pid)
    if recorded_identity is not None and recorded_identity != current_identity:
        return ProcessStatus("pid-reused", pid)
    if recorded_identity is None and not _process_has_environment_identity(pid, service_id):
        return ProcessStatus("pid-reused", pid)
    return ProcessStatus("running", pid)


def _agent_stop(service_id: str, *, pid_file: Path) -> ProcessStatus:
    status = _agent_status(service_id, pid_file=pid_file)
    if status.state == "stopped":
        return status
    if status.state == "stale":
        pid_file.unlink(missing_ok=True)
        _identity_path(pid_file).unlink(missing_ok=True)
        return status
    if status.state == "pid-reused" or status.pid is None:
        raise SupervisorError("PID belongs to another process; refusing to signal it")
    _terminate_process_group(status.pid)
    pid_file.unlink(missing_ok=True)
    _identity_path(pid_file).unlink(missing_ok=True)
    return ProcessStatus("stopped")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_has_environment_identity(pid: int, service_id: str) -> bool:
    expected = f"{_IDENTITY_ENVIRONMENT}={service_id}".encode()
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return expected in environment.split(b"\0")


def _process_identity(pid: int) -> str | None:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="ascii").rsplit(")", 1)[1].split()
        return f"proc:{fields[19]}"
    except (IndexError, OSError, UnicodeError):
        pass
    completed = subprocess.run(
        ("ps", "-p", str(pid), "-o", "lstart="),
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return f"ps:{value}" if completed.returncode == 0 and value else None


def _identity_path(pid_file: Path) -> Path:
    return pid_file.with_suffix(".identity")


def _read_identity(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while _process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_exists(pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def _write_private_text(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


__all__ = ["ProcessStatus", "ServiceSupervisor", "SupervisorError"]


if __name__ == "__main__":
    raise SystemExit(_agent_main())
