"""Versioned stream envelopes shared by trainer-selected transports.

No policy, tensor framework, robot driver, or trainer runtime is imported here.
Stream failures are explicit: an interrupted training trajectory must be reset by
its owner, never silently continued after dropping an observation or action.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import threading
from collections.abc import Callable
from typing import Any

SCHEMA = "embodirun.rollout.v1"


class StreamError(RuntimeError):
    """A message cannot safely be applied to the current stream."""


class StreamSession:
    """Validate session, route and contiguous sequence before exposing payloads."""

    def __init__(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.session_id = session_id
        self._sent: dict[tuple, int] = {}
        self._received: dict[tuple, int] = {}

    def pack(self, route: tuple, payload: Any) -> dict:
        sequence = self._sent.get(route, 0)
        self._sent[route] = sequence + 1
        return {
            "schema": SCHEMA,
            "session": self.session_id,
            "route": route,
            "sequence": sequence,
            "payload": payload,
        }

    def unpack(self, route: tuple, message: dict) -> Any:
        if (
            not isinstance(message, dict)
            or message.get("schema") != SCHEMA
            or message.get("session") != self.session_id
            or message.get("route") != route
        ):
            raise StreamError("foreign session, schema or route; reset the trajectory")
        expected = self._received.get(route, 0)
        if type(message.get("sequence")) is not int or message["sequence"] != expected:
            raise StreamError(f"expected sequence {expected}; reset the trajectory")
        self._received[route] = expected + 1
        return message["payload"]


def route_tag(route: tuple) -> int:
    """Derive a stable transport tag; the full route is also checked on receive."""
    value = json.dumps(route, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


class WirelessEndpoint:
    """Own a peer's event loop and return cancellable concurrent futures.

    The optional codec registrar belongs to the caller. Deploy does not need to
    know the caller's tensor or transition types. A send completes after local
    transmission, not after remote inference or action consumption.
    """

    def __init__(
        self,
        local: dict,
        peers: list[dict],
        *,
        timeout_s: float = 30,
        register: Callable | None = None,
        bind_host: str = "0.0.0.0",
        retry_disconnected_receive: bool = False,
    ):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.retry_disconnected_receive = retry_disconnected_receive
        self.receive_disconnect_retries = 0
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._closed = False
        try:
            self._submit(self._start(local, peers, bind_host, register)).result(timeout_s)
        except BaseException:
            self.close()
            raise

    async def _start(self, local, peers, bind_host, register):
        from wireless_comm import Comm, Peer

        self._comm = await Comm.create(
            local=Peer(**local),
            peers=tuple(Peer(**p) for p in peers),
            bind_host=bind_host,
        )
        if register is not None:
            register(self._comm)

    def _submit(self, coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def send(self, peer_id: str, route: tuple, payload: Any):
        return self._submit(self._send(peer_id, route, payload))

    async def _send(self, peer_id, route, payload):
        from wireless_comm import CommOptions

        return await self._comm.send(
            payload,
            self._comm.peer(peer_id),
            options=CommOptions(tag=route_tag(route), timeout=self.timeout_s),
        )

    def recv(self, peer_id: str, route: tuple):
        return self._submit(self._recv(peer_id, route))

    async def _recv(self, peer_id, route):
        from wireless_comm import (
            CommOptions,
            ConnectionClosedError,
            OperationTimeoutError,
        )

        deadline = self._loop.time() + self.timeout_s
        while True:
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise OperationTimeoutError(f"receive from {peer_id!r} timed out")
            try:
                payload, _ = await self._comm.recv(
                    self._comm.peer(peer_id),
                    options=CommOptions(tag=route_tag(route), timeout=remaining),
                )
                return payload
            except ConnectionClosedError:
                if not self.retry_disconnected_receive or self._closed:
                    raise
                # Re-arm the same receive while the peer reconnects. No send is
                # retried and no stream sequence is consumed or skipped here.
                self.receive_disconnect_retries += 1
                await asyncio.sleep(min(0.01, max(0, deadline - self._loop.time())))

    async def _shutdown(self):
        if hasattr(self, "_comm"):
            await self._comm.close()
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._submit(self._shutdown()).result(self.timeout_s)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=self.timeout_s)
            if not self._thread.is_alive():
                self._loop.close()
