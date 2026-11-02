"""Single authenticated HTTP endpoint for the Go2 owner's components."""

from __future__ import annotations

import signal
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from .camera.http_handler import CameraRoutes
from .control.http import Go2RequestHandler


class OwnerRequestHandler(Go2RequestHandler, CameraRoutes):
    """Compose existing wire protocols behind one endpoint and credential."""

    @property
    def executor(self):
        return self.server.owner.executor

    @property
    def api_token(self):
        return self.server.owner.config.token

    @property
    def camera_server(self):
        return self.server.owner.camera

    def do_GET(self) -> None:
        if not self._check_auth():
            return
        path = urlsplit(self.path).path
        if path == "/health":
            payload = self.server.owner.health()
            self._send_json(200 if payload["ok"] else 503, payload)
        elif path == "/v1/state":
            self._send_json(200, self.executor.snapshot())
        elif self.camera_server is None:
            self._send_json(404, {"error": "camera is not configured"})
        else:
            self.serve_camera_get()

    def log_message(self, fmt, *args) -> None:
        """Do not log every heartbeat or encoded frame request."""


class OwnerHTTPServer(ThreadingHTTPServer):
    """Keep camera response work separate from bounded control requests."""

    daemon_threads = True

    def __init__(self, owner):
        self.owner = owner
        super().__init__((owner.config.host, owner.config.port), OwnerRequestHandler)


def serve(owner, *, install_signal_handlers: bool = True) -> None:
    """Bind first, start capture, and stop motors before closing camera workers."""
    server = None
    stopping = threading.Event()
    previous = {}

    def shutdown(signum, frame):
        if stopping.is_set():
            return
        stopping.set()

        def stop_and_shutdown():
            try:
                owner.executor.stop("signal")
            finally:
                server.shutdown()

        threading.Thread(target=stop_and_shutdown, daemon=True, name="go2-stop").start()

    try:
        server = OwnerHTTPServer(owner)
        owner.start()
        if install_signal_handlers:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, shutdown)
        print(f"Go2 owner listening on http://{owner.config.host}:{owner.config.port}", flush=True)
        server.serve_forever(poll_interval=0.2)
    finally:
        try:
            owner.close()
        finally:
            if server is not None:
                server.server_close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
