"""HTTP server lifecycle and signal-safe final StopMove handling."""

from __future__ import annotations

import signal
import threading
from http.server import ThreadingHTTPServer
from types import FrameType

from .config import ControlServerConfig
from .executor import ActionExecutor
from .http import make_request_handler
from .sdk2 import loaded_dds_library_path


def create_http_server(
    config: ControlServerConfig,
    executor: ActionExecutor,
) -> ThreadingHTTPServer:
    handler = make_request_handler(executor, config.token)
    return ThreadingHTTPServer((config.host, config.port), handler)


def serve_control_api(
    config: ControlServerConfig,
    executor: ActionExecutor,
    *,
    install_signal_handlers: bool = True,
) -> None:
    """Serve until shutdown and always close the executor/transport."""

    try:
        server = create_http_server(config, executor)
    except BaseException:
        executor.close()
        raise

    stopping = threading.Event()

    def shutdown(signum: int, _frame: FrameType | None) -> None:
        if stopping.is_set():
            return
        stopping.set()
        print(f"received signal {signum}; stopping robot and server", flush=True)

        def stop_and_shutdown() -> None:
            try:
                executor.stop("signal")
            except Exception as exc:  # noqa: BLE001 - signal shutdown must continue
                print(f"signal StopMove failed: {exc}", flush=True)
            finally:
                server.shutdown()

        threading.Thread(
            target=stop_and_shutdown,
            daemon=True,
            name="go2-signal-shutdown",
        ).start()

    if install_signal_handlers:
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

    details = f"Go2 API listening on http://{config.host}:{config.port} mode={config.mode} interface={config.interface}"
    if config.mode == "live":
        details += f" dds_library={loaded_dds_library_path()}"
    print(details, flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        executor.close(send_stop=not stopping.is_set())


__all__ = ["create_http_server", "serve_control_api"]
