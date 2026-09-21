"""SSH deployment command execution."""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import posixpath
import shlex
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import ConnectionConfig
from .command import Command, CommandError, CommandResult
from .response import JsonHttpResponse

_MAX_JSON_RESPONSE_BYTES = 64 * 1024
_PARAMIKO_LOG_CHANNEL = "embodirun.paramiko"
_PARAMIKO_LOGGER = logging.getLogger(_PARAMIKO_LOG_CHANNEL)
_PARAMIKO_LOGGER.addHandler(logging.NullHandler())
_PARAMIKO_LOGGER.propagate = False


class SshExecutor:
    """Execute POSIX commands over one lazily-created Paramiko connection."""

    def __init__(self, connection: ConnectionConfig) -> None:
        if connection.kind != "ssh":
            raise ValueError("SshExecutor requires an SSH connection")
        self.connection = connection
        self._client: Any | None = None
        self._password = _password_from_environment(connection)

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import paramiko
        except ImportError as error:  # pragma: no cover - depends on installed group
            raise RuntimeError(
                "SSH execution requires the host environment; run `uv sync --frozen --no-dev --group host`"
            ) from error

        client = paramiko.SSHClient()
        client.set_log_channel(_PARAMIKO_LOG_CHANNEL)
        client.load_system_host_keys()
        if self.connection.accept_new_host_key:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
        if self._password is not None:
            kwargs["password"] = self._password
        if self.connection.identity_file is not None:
            kwargs["key_filename"] = os.fspath(Path(self.connection.identity_file).expanduser())
        proxy = None
        try:
            if self.connection.proxy_command is not None:
                command = shlex.join(
                    part.replace("%h", self.connection.host).replace("%p", str(self.connection.port))
                    for part in shlex.split(self.connection.proxy_command)
                )
                proxy = paramiko.ProxyCommand(command)
                kwargs["sock"] = proxy
            client.connect(**kwargs)
        except Exception as error:  # noqa: BLE001 - normalize third-party errors
            client.close()
            if proxy is not None:
                proxy.close()
            detail = _redact(str(error), self._password)
            raise RuntimeError(f"SSH connection failed: {detail}") from None
        self._client = client
        return client

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        invocation = render_posix(command)
        timeout = command.timeout_s or self.connection.command_timeout_s
        try:
            stdin, stdout, stderr = self._connect().exec_command(
                invocation,
                timeout=timeout,
            )
            if command.stdin is not None:
                stdin.write(command.stdin)
                stdin.flush()
            stdin.channel.shutdown_write()
            output = stdout.read().decode("utf-8", errors="replace")
            error = stderr.read().decode("utf-8", errors="replace")
            exit_code = int(stdout.channel.recv_exit_status())
        except Exception as exc:  # noqa: BLE001 - normalize third-party errors
            detail = _redact(str(exc), self._password)
            raise RuntimeError(f"SSH command failed: {detail}") from None
        result = CommandResult(
            exit_code=exit_code,
            stdout=_redact(output, self._password),
            stderr=_redact(error, self._password),
        )
        if check and result.exit_code != 0:
            raise CommandError(result.exit_code, result.stderr)
        return result

    def get_json(
        self,
        url: str,
        *,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> JsonHttpResponse:
        """Read remote-node HTTP JSON through an SSH direct TCP channel."""

        return self.request_json("GET", url, None, timeout_s=timeout_s, headers=headers)

    def request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> JsonHttpResponse:
        """Exchange bounded JSON through an SSH direct TCP channel."""

        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("remote service URL must be HTTP without credentials")
        try:
            port = parsed.port or 80
        except ValueError as error:
            raise ValueError("remote service URL has an invalid port") from error
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        transport = self._connect().get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH transport is not active")
        try:
            channel = transport.open_channel(
                "direct-tcpip",
                (parsed.hostname, port),
                ("127.0.0.1", 0),
                timeout=timeout_s,
            )
        except Exception as error:  # noqa: BLE001 - normalize Paramiko errors
            detail = _redact(str(error), self._password)
            raise RuntimeError(f"SSH HTTP channel failed: {detail}") from None
        channel.settimeout(timeout_s)
        connection = http.client.HTTPConnection(
            parsed.hostname,
            port,
            timeout=timeout_s,
        )
        connection.sock = channel
        body = _request_body(payload)
        request_headers = {"Accept": "application/json", "Connection": "close"}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        _merge_request_headers(request_headers, headers)
        try:
            connection.request(
                method,
                target,
                body=body,
                headers=request_headers,
            )
            response = connection.getresponse()
            body = response.read(_MAX_JSON_RESPONSE_BYTES + 1)
            status = response.status
        finally:
            connection.close()
        if len(body) > _MAX_JSON_RESPONSE_BYTES:
            raise ValueError("HTTP JSON response is too large")
        return JsonHttpResponse(status, json.loads(body))

    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None:
        """Atomically upload a UTF-8 text file through SFTP."""

        self.write_bytes(path, content.encode("utf-8"), mode=mode)

    def read_bytes(self, path: str) -> bytes:
        """Download one file through SFTP."""

        _validate_remote_path(path)
        sftp = None
        try:
            sftp = self._connect().open_sftp()
            with sftp.open(path, "rb") as stream:
                return stream.read()
        except Exception as error:
            detail = _redact(str(error), self._password)
            raise RuntimeError(f"SFTP file download failed: {detail}") from None
        finally:
            if sftp is not None:
                sftp.close()

    def replace_symlink(self, path: str, target: str) -> None:
        """Atomically point one remote symlink at a new target through SFTP."""

        _validate_remote_path(path)
        _validate_remote_path(target)
        parent = posixpath.dirname(path)
        self.run(Command(("mkdir", "-p", parent)))
        temporary = f"{path}.{uuid.uuid4().hex}.tmp"
        sftp = None
        try:
            sftp = self._connect().open_sftp()
            sftp.symlink(target, temporary)
            sftp.posix_rename(temporary, path)
        except Exception as error:
            detail = _redact(str(error), self._password)
            raise RuntimeError(f"SFTP symlink replacement failed: {detail}") from None
        finally:
            if sftp is not None:
                with contextlib.suppress(OSError):
                    sftp.remove(temporary)
                sftp.close()

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        """Atomically upload bytes through SFTP."""

        _validate_remote_path(path)
        parent = posixpath.dirname(path)
        self.run(Command(("mkdir", "-p", parent)))
        temporary = f"{path}.{uuid.uuid4().hex}.tmp"
        sftp = None
        try:
            sftp = self._connect().open_sftp()
            with sftp.open(temporary, "wb") as stream:
                stream.write(content)
                stream.flush()
            sftp.chmod(temporary, mode)
            sftp.posix_rename(temporary, path)
        except Exception as error:
            detail = _redact(str(error), self._password)
            raise RuntimeError(f"SFTP file upload failed: {detail}") from None
        finally:
            if sftp is not None:
                with contextlib.suppress(OSError):
                    sftp.remove(temporary)
                sftp.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def render_posix(command: Command) -> str:
    """Render an argv-based command for a POSIX remote shell."""

    invocation = shlex.join(command.argv)
    if command.environment:
        assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(command.environment.items()))
        invocation = f"env {assignments} {invocation}"
    if command.cwd is not None:
        invocation = f"cd {shlex.quote(command.cwd)} && {invocation}"
    return invocation


def _password_from_environment(connection: ConnectionConfig) -> str | None:
    if connection.password_env is None:
        return None
    password = os.environ.get(connection.password_env)
    if password is None:
        raise RuntimeError(f"SSH password environment variable {connection.password_env!r} is not set")
    return password


def _redact(value: str, secret: str | None) -> str:
    if secret:
        return value.replace(secret, "<redacted>")
    return value


def _validate_remote_path(path: str) -> None:
    if not path.startswith("/") or "\x00" in path:
        raise ValueError("remote file path must be an absolute POSIX path")


def _request_body(payload: Mapping[str, Any] | None) -> bytes | None:
    if payload is None:
        return None
    try:
        return json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("HTTP request payload is not valid JSON") from error


def _merge_request_headers(target: dict[str, str], extra: Mapping[str, str] | None) -> None:
    if extra is None:
        return
    for name, value in extra.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("HTTP header names must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError("HTTP header values must be strings")
        target[name] = value


__all__ = ["SshExecutor", "render_posix"]
