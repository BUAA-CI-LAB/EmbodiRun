"""Local deployment command execution."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .command import Command, CommandError, CommandResult
from .response import JsonHttpResponse

_MAX_JSON_RESPONSE_BYTES = 64 * 1024


class LocalExecutor:
    """Execute commands directly without a shell."""

    def run(self, command: Command, *, check: bool = True) -> CommandResult:
        environment = os.environ.copy()
        environment.update(command.environment)
        try:
            completed = subprocess.run(
                command.argv,
                cwd=command.cwd,
                env=environment,
                input=command.stdin,
                capture_output=True,
                text=True,
                timeout=command.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("local command timed out") from error
        except OSError as error:
            raise RuntimeError(f"local command could not start: {error}") from error
        result = CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
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
        """Read JSON from an HTTP endpoint reachable by the local node."""

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
        """Exchange bounded JSON with a service reachable by the local node."""

        body = _request_body(payload)
        request_headers = {"Accept": "application/json"}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        _merge_request_headers(request_headers, headers)
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout_s)
        except urllib.error.HTTPError as error:
            response = error
        try:
            response_body = response.read(_MAX_JSON_RESPONSE_BYTES + 1)
            status = int(response.status)
        finally:
            response.close()
        if len(response_body) > _MAX_JSON_RESPONSE_BYTES:
            raise ValueError("HTTP JSON response is too large")
        return JsonHttpResponse(status, json.loads(response_body))

    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None:
        """Atomically write a UTF-8 text file for a local deployment node."""

        self.write_bytes(path, content.encode("utf-8"), mode=mode)

    def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def replace_symlink(self, path: str, target: str) -> None:
        """Atomically point one local symlink at a new target."""

        link = Path(path)
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.symlink_to(target)
            os.replace(temporary, link)
        finally:
            temporary.unlink(missing_ok=True)

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        """Atomically write bytes for a local deployment node."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    def close(self) -> None:
        return None


__all__ = ["LocalExecutor"]


def _merge_request_headers(target: dict[str, str], extra: Mapping[str, str] | None) -> None:
    if extra is None:
        return
    for name, value in extra.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("HTTP header names must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError("HTTP header values must be strings")
        target[name] = value


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
