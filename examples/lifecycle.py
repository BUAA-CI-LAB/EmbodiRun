"""Stop a foreground recipe through its own private socket, never through a saved PID."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import tempfile
import threading
from pathlib import Path
from types import TracebackType
from typing import Any


class Supervisor:
    """Hold one action's lock while its children and their cleanup are alive."""

    def __init__(self, output: Path, action: str) -> None:
        self.output = output
        self.action = action
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.pointer = output / f".launcher-{action}.json"
        self.lock: Any = None
        self.server: socket.socket | None = None
        self.directory: tempfile.TemporaryDirectory | None = None
        self.thread: threading.Thread | None = None
        self.previous_signal: Any = None

    def __enter__(self) -> Supervisor:
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.output / f".launcher-{self.action}.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise RuntimeError(f"{self.action} is already running for {self.output}; use down first") from error
        try:
            # A short temporary path also works with macOS's small Unix-socket path limit.
            self.directory = tempfile.TemporaryDirectory(prefix="embodirun-", dir="/tmp")
            endpoint = Path(self.directory.name) / "control.sock"
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(str(endpoint))
            os.chmod(endpoint, 0o600)
            self.server.listen(1)
            self.server.settimeout(0.2)
            fd, temporary = tempfile.mkstemp(prefix=".launcher-", dir=self.output)
            with os.fdopen(fd, "w") as stream:
                json.dump({"socket": str(endpoint)}, stream)
            os.replace(temporary, self.pointer)
            self.previous_signal = signal.signal(signal.SIGTERM, self._terminate)
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _terminate(self, signum: int, frame: Any) -> None:
        self.cancelled.set()

    def _listen(self) -> None:
        assert self.server is not None
        while not self.finished.is_set():
            try:
                connection, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                connection.settimeout(2)
                try:
                    if connection.recv(16) != b"stop\n":
                        continue
                    self.cancelled.set()
                    # Acknowledgement follows cleanup, not just receipt of the request.
                    if self.finished.wait(300):
                        connection.sendall(b"exited\n")
                    else:
                        connection.sendall(b"cleanup pending\n")
                except OSError:
                    pass

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.previous_signal is not None:
            signal.signal(signal.SIGTERM, self.previous_signal)
        if self.server is not None:
            self.server.close()
        self.pointer.unlink(missing_ok=True)
        if self.directory is not None:
            self.directory.cleanup()
        if self.lock is not None:
            self.lock.close()


def stop_owned(output: Path) -> None:
    """Stop the task before its service owner; refuse stale/unknown process state."""
    stopped = False
    errors: list[str] = []
    for action in ("run", "up", "check", "dry-run"):
        pointer = output / f".launcher-{action}.json"
        if not pointer.exists():
            continue
        try:
            endpoint = json.loads(pointer.read_text())["socket"]
            if not isinstance(endpoint, str):
                raise ValueError("launcher socket path must be a string")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(305)
                client.connect(endpoint)
                client.sendall(b"stop\n")
                if client.recv(32) != b"exited\n":
                    raise RuntimeError("launcher did not confirm process cleanup")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(f"Cannot confirm {action} cleanup using {pointer}: {error}")
            continue
        print(f"{action}: launcher finished handling the stop request")
        stopped = True
    if errors:
        raise RuntimeError("; ".join(errors))
    if not stopped:
        print("No active local launcher was found. Inspect service logs and physical stop feedback at the owner.")
