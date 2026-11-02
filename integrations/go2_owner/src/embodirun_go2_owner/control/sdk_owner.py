"""Single-thread ownership and FIFO dispatch for Unitree native SDK objects."""

from __future__ import annotations

import queue
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from .config import (
    SDK_OWNER_CALL_GRACE_S,
    SDK_OWNER_CLOSE_TIMEOUT_S,
    SDK_OWNER_INIT_TIMEOUT_S,
)
from .sdk_state import start_state_subscription


@dataclass
class SdkCall:
    method: str
    args: tuple[Any, ...]
    future: Future[int]


class SdkOwner:
    """Own native SDK objects and serialize every RPC on one thread."""

    def __init__(
        self,
        interface: str,
        state_topic: str,
        rpc_timeout_s: float,
        components: tuple[Any, Any, Any, Any],
        state_callback: Any,
    ) -> None:
        self.submit_lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.queue: queue.Queue[SdkCall | None] = queue.Queue()
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.init_error: BaseException | None = None
        self.closing = False
        self.closed = False
        self.faulted: str | None = None
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.subscriber: Any = None
        self.client: Any = None
        self.thread = threading.Thread(
            target=self._owner_loop,
            args=(interface, state_topic, components, state_callback),
            daemon=True,
            name="go2-sdk-owner",
        )
        self.thread.start()
        if not self.ready.wait(timeout=SDK_OWNER_INIT_TIMEOUT_S):
            with self.lifecycle_lock:
                self.closed = True
                self.queue.put(None)
            raise RuntimeError("timed out while initializing Unitree SDK owner thread")
        if self.init_error is not None:
            raise RuntimeError(f"failed to initialize Unitree SDK: {self.init_error}") from self.init_error

    def _owner_loop(
        self,
        interface: str,
        state_topic: str,
        components: tuple[Any, Any, Any, Any],
        state_callback: Any,
    ) -> None:
        try:
            (
                channel_factory_initialize,
                channel_subscriber,
                sport_client,
                sport_mode_state,
            ) = components
            channel_factory_initialize(0, interface)
            subscriber = start_state_subscription(
                channel_subscriber,
                state_topic,
                sport_mode_state,
                state_callback,
            )
            client = sport_client()
            client.SetTimeout(self.rpc_timeout_s)
            client.Init()
            self.subscriber = subscriber
            self.client = client
        except BaseException as exc:  # noqa: BLE001 - always release init waiter
            self.init_error = exc
            self.ready.set()
            self.stopped.set()
            return

        self.ready.set()
        try:
            while True:
                call = self.queue.get()
                if call is None:
                    break
                if not call.future.set_running_or_notify_cancel():
                    continue
                with self.lifecycle_lock:
                    faulted = self.faulted
                if faulted is not None and call.method != "StopMove":
                    call.future.set_exception(RuntimeError(f"Unitree SDK transport is faulted: {faulted}"))
                    continue
                call_started = time.monotonic()
                try:
                    method = getattr(client, call.method)
                    call.future.set_result(int(method(*call.args)))
                except BaseException as exc:  # noqa: BLE001 - propagate through Future
                    call.future.set_exception(exc)
                finally:
                    elapsed = time.monotonic() - call_started
                    timeout_s = max(0.1, self.rpc_timeout_s + SDK_OWNER_CALL_GRACE_S)
                    if elapsed > timeout_s and call.method != "StopMove":
                        with self.lifecycle_lock:
                            self.faulted = f"in-flight {call.method} exceeded {timeout_s:.1f}s"
        finally:
            with self.lifecycle_lock:
                self.closing = True
                self.closed = True
            owner_error = RuntimeError("Unitree SDK owner thread exited")
            while True:
                try:
                    pending = self.queue.get_nowait()
                except queue.Empty:
                    break
                if pending is not None:
                    try:
                        if pending.future.set_running_or_notify_cancel():
                            pending.future.set_exception(owner_error)
                    except RuntimeError:
                        pass
            # Destruction of native-backed objects remains on their owner
            # thread. ChannelFactory is process-wide and is not closed here.
            self.client = None
            self.subscriber = None
            del client
            del subscriber
            self.stopped.set()

    def enqueue(
        self,
        method: str,
        args: tuple[Any, ...],
        *,
        allow_closing: bool = False,
    ) -> SdkCall:
        call = SdkCall(method=method, args=args, future=Future())
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError("Unitree SDK transport is closed")
            if self.closing and not allow_closing:
                raise RuntimeError("Unitree SDK transport is closing")
            if self.stopped.is_set() or not self.thread.is_alive():
                raise RuntimeError("Unitree SDK owner thread is unavailable")
            if self.faulted is not None and method != "StopMove":
                raise RuntimeError(f"Unitree SDK transport is faulted: {self.faulted}")
            self.queue.put(call)
        return call

    @staticmethod
    def _completed_result(call: SdkCall) -> int:
        try:
            return int(call.future.result())
        except BaseException as exc:
            raise RuntimeError(f"Unitree SDK {call.method} failed: {exc}") from exc

    def wait(self, call: SdkCall, *, cancel_if_pending: bool = True) -> int:
        timeout_s = max(0.1, self.rpc_timeout_s + SDK_OWNER_CALL_GRACE_S)
        try:
            return int(call.future.result(timeout=timeout_s))
        except FutureTimeoutError as exc:
            if call.future.done():
                return self._completed_result(call)
            if cancel_if_pending and call.future.cancel():
                raise TimeoutError(f"Unitree SDK {call.method} timed out before execution; call cancelled") from exc
            if call.future.done():
                return self._completed_result(call)
            state = "in-flight" if call.future.running() else "queued"
            fault = f"{state} {call.method} exceeded {timeout_s:.1f}s"
            with self.lifecycle_lock:
                self.faulted = fault
            disposition = "transport faulted" if cancel_if_pending else "transport faulted; safety call remains queued"
            raise TimeoutError(f"Unitree SDK {call.method} timed out {state}; {disposition}") from exc
        except BaseException as exc:
            raise RuntimeError(f"Unitree SDK {call.method} failed: {exc}") from exc

    def call(self, method: str, *args: Any) -> int:
        with self.submit_lock:
            return self.wait(self.enqueue(method, args))

    def stop(self) -> int:
        with self.submit_lock:
            call = self.enqueue("StopMove", ())
            # Never cancel a safety stop merely because it was queued behind a
            # slow native RPC. It remains in the owner-thread FIFO.
            return self.wait(call, cancel_if_pending=False)

    def close(self) -> None:
        with self.submit_lock:
            with self.lifecycle_lock:
                if self.closed:
                    already_closed = True
                    final_stop = None
                else:
                    already_closed = False
                    self.closing = True
                    if self.stopped.is_set() or not self.thread.is_alive():
                        final_stop = None
                    else:
                        final_stop = SdkCall("StopMove", (), Future())
                        self.queue.put(final_stop)

            if not already_closed:
                if final_stop is not None:
                    try:
                        self.wait(final_stop, cancel_if_pending=False)
                    except Exception as exc:  # noqa: BLE001 - best-effort safety stop
                        print(
                            f"final Unitree StopMove failed: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                with self.lifecycle_lock:
                    self.closed = True
                    self.queue.put(None)

            self.thread.join(timeout=SDK_OWNER_CLOSE_TIMEOUT_S)
        if self.thread.is_alive():
            print(
                "Unitree SDK owner thread did not exit before close timeout",
                file=sys.stderr,
                flush=True,
            )


__all__ = ["SdkCall", "SdkOwner"]
