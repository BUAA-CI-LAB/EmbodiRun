"""SSH/SFTP transport boundary used by the Go2 deployment commands."""

from __future__ import annotations

import contextlib
import os
import posixpath
import shlex
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from embodirun.deployment.executor import Command


@dataclass(frozen=True)
class SshConnection:
    """Connection settings with secrets deliberately excluded from ``repr``."""

    host: str
    username: str
    port: int = 22
    password: str | None = field(default=None, repr=False)
    identity_file: Path | None = None
    connect_timeout_s: float = 10.0
    command_timeout_s: float = 30.0
    accept_new_host_key: bool = False

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("SSH host cannot be empty")
        if not self.username.strip():
            raise ValueError("SSH username cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        if self.connect_timeout_s <= 0 or self.command_timeout_s <= 0:
            raise ValueError("SSH timeouts must be positive")


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one remote command."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""


class RemoteCommandError(RuntimeError):
    """A remote command failed, without retaining its potentially secret command."""

    def __init__(self, exit_code: int, stderr: str = "") -> None:
        detail = stderr.strip()
        message = f"remote command failed with exit code {exit_code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.exit_code = exit_code


class RemoteTransport(Protocol):
    """Minimal transport surface; tests can provide an in-memory fake."""

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        """Execute one argv-based command on the remote host."""

    def resolve_path(self, path: str) -> str:
        """Resolve a remote path against the SSH user's home directory."""

    def make_dirs(self, path: str, *, mode: int = 0o755) -> None:
        """Create a remote directory tree."""

    def put_file(self, local_path: Path, remote_path: str, *, mode: int = 0o644) -> None:
        """Atomically upload a local file."""

    def put_bytes(self, payload: bytes, remote_path: str, *, mode: int = 0o600) -> None:
        """Atomically upload in-memory bytes."""

    def close(self) -> None:
        """Release the underlying connection."""


class ParamikoTransport:
    """Production SSH transport with Paramiko imported only when connected."""

    def __init__(
        self,
        connection: SshConnection,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.connection = connection
        self._client_factory = client_factory
        self._client: Any | None = None
        self._sftp: Any | None = None

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client

        if self._client_factory is None:
            try:
                import paramiko
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Paramiko is required for remote deployment; run `uv sync --frozen --no-dev --group host`"
                ) from exc
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            policy = paramiko.AutoAddPolicy() if self.connection.accept_new_host_key else None
            if policy is not None:
                client.set_missing_host_key_policy(policy)
        else:
            client = self._client_factory()

        kwargs: dict[str, Any] = {
            "hostname": self.connection.host,
            "port": self.connection.port,
            "username": self.connection.username,
            "timeout": self.connection.connect_timeout_s,
            "banner_timeout": self.connection.connect_timeout_s,
            "auth_timeout": self.connection.connect_timeout_s,
            "allow_agent": True,
            "look_for_keys": self.connection.identity_file is None,
        }
        if self.connection.password is not None:
            kwargs["password"] = self.connection.password
        if self.connection.identity_file is not None:
            kwargs["key_filename"] = os.fspath(self.connection.identity_file)

        try:
            client.connect(**kwargs)
        except Exception as exc:  # noqa: BLE001 - sanitize third-party transport failures
            # Paramiko does not normally echo credentials, but sanitize anyway in
            # case a custom client or future dependency does.
            message = _redact(str(exc), self.connection.password)
            raise RuntimeError(f"SSH connection failed: {message}") from None
        self._client = client
        return client

    def _open_sftp(self) -> Any:
        if self._sftp is None:
            self._sftp = self._connect().open_sftp()
        return self._sftp

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        client = self._connect()
        invocation = shlex.join(command.argv)
        if command.environment:
            assignments = " ".join(
                f"{name}={shlex.quote(value)}" for name, value in sorted(command.environment.items())
            )
            invocation = f"env {assignments} {invocation}"
        if command.cwd is not None:
            invocation = f"cd {shlex.quote(command.cwd)} && {invocation}"
        try:
            stdin, stdout, stderr = client.exec_command(
                invocation,
                timeout=command.timeout_s or self.connection.command_timeout_s,
            )
            if command.stdin is not None:
                stdin.write(command.stdin)
                stdin.flush()
            stdin.channel.shutdown_write()
            output = stdout.read().decode("utf-8", errors="replace")
            error = stderr.read().decode("utf-8", errors="replace")
            exit_code = int(stdout.channel.recv_exit_status())
        except Exception as exc:  # noqa: BLE001 - normalize third-party channel failures
            message = _redact(str(exc), self.connection.password)
            raise RuntimeError(f"remote command transport failed: {message}") from None

        output = _redact(output, self.connection.password)
        error = _redact(error, self.connection.password)
        result = CommandResult(exit_code=exit_code, stdout=output, stderr=error)
        if check and exit_code != 0:
            raise RemoteCommandError(exit_code, error)
        return result

    def resolve_path(self, path: str) -> str:
        if not path or "\x00" in path:
            raise ValueError("remote path cannot be empty or contain NUL")
        sftp = self._open_sftp()
        home = sftp.normalize(".")
        if path == "~":
            return home
        if path.startswith("~/"):
            path = posixpath.join(home, path[2:])
        elif not path.startswith("/"):
            path = posixpath.join(home, path)
        return posixpath.normpath(path)

    def make_dirs(self, path: str, *, mode: int = 0o755) -> None:
        sftp = self._open_sftp()
        normalized = posixpath.normpath(path)
        if not normalized.startswith("/"):
            raise ValueError("make_dirs requires an absolute remote path")
        current = "/"
        for component in normalized.strip("/").split("/"):
            if not component:
                continue
            current = posixpath.join(current, component)
            try:
                attributes = sftp.stat(current)
                if not stat.S_ISDIR(attributes.st_mode):
                    raise RuntimeError(f"remote path exists but is not a directory: {current}")
            except OSError:
                sftp.mkdir(current, mode=mode)

    def put_file(self, local_path: Path, remote_path: str, *, mode: int = 0o644) -> None:
        self.make_dirs(posixpath.dirname(remote_path))
        temporary = _temporary_remote_path(remote_path)
        sftp = self._open_sftp()
        try:
            sftp.put(os.fspath(local_path), temporary)
            sftp.chmod(temporary, mode)
            self._replace(sftp, temporary, remote_path)
        except Exception:
            _best_effort_remove(sftp, temporary)
            raise

    def put_bytes(self, payload: bytes, remote_path: str, *, mode: int = 0o600) -> None:
        self.make_dirs(posixpath.dirname(remote_path))
        temporary = _temporary_remote_path(remote_path)
        sftp = self._open_sftp()
        try:
            with sftp.file(temporary, "wb") as remote_file:
                remote_file.write(payload)
                remote_file.flush()
            sftp.chmod(temporary, mode)
            self._replace(sftp, temporary, remote_path)
        except Exception:
            _best_effort_remove(sftp, temporary)
            raise

    @staticmethod
    def _replace(sftp: Any, source: str, destination: str) -> None:
        try:
            sftp.posix_rename(source, destination)
        except (AttributeError, OSError):
            _best_effort_remove(sftp, destination)
            sftp.rename(source, destination)

    def close(self) -> None:
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self._client is not None:
            self._client.close()
            self._client = None


def _temporary_remote_path(path: str) -> str:
    return f"{path}.tmp-{uuid.uuid4().hex}"


def _best_effort_remove(sftp: Any, path: str) -> None:
    with contextlib.suppress(OSError):
        sftp.remove(path)


def _redact(text: str, secret: str | None) -> str:
    if secret:
        return text.replace(secret, "<redacted>")
    return text
