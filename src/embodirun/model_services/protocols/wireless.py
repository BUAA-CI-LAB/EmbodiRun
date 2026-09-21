"""Synchronous RPC protocol over an asynchronous WirelessComm runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from wireless_comm import Comm, Peer, RuntimeConfig


class WirelessProtocolError(RuntimeError):
    """A WirelessComm RPC exchange failed locally or remotely."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class WirelessTransport(Protocol):
    """Blocking request boundary implemented by a wireless protocol runtime."""

    def request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


class WirelessRpcTransport:
    """Own a WirelessComm event loop and expose blocking RPC calls."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        server_node_id: str,
        rpc_schema: str,
        request_tag: int,
        response_tag: int,
        token: str | None = None,
    ) -> None:
        if not isinstance(server_node_id, str) or not server_node_id:
            raise ValueError("server_node_id must be non-empty")
        if not isinstance(rpc_schema, str) or not rpc_schema:
            raise ValueError("rpc_schema must be non-empty")
        tags = (("request_tag", request_tag), ("response_tag", response_tag))
        for name, value in tags:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._runtime_config = runtime_config
        self._server_node_id = server_node_id
        self._rpc_schema = rpc_schema
        self._request_tag = request_tag
        self._response_tag = response_tag
        self._token = token
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="wireless-rpc-client",
            daemon=True,
        )
        self._comm: Comm | None = None
        self._server: Peer | None = None
        self._responses: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._failure: BaseException | None = None
        self._closed = False
        self._thread.start()
        try:
            self._submit(self._start()).result()
        except Exception:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()
            raise

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        server_node_id: str,
        rpc_schema: str,
        request_tag: int,
        response_tag: int,
        token: str | None = None,
    ) -> WirelessRpcTransport:
        from wireless_comm import load_runtime_config

        return cls(
            load_runtime_config(config_path),
            server_node_id=server_node_id,
            rpc_schema=rpc_schema,
            request_tag=request_tag,
            response_tag=response_tag,
            token=token,
        )

    def request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        if self._closed:
            raise WirelessProtocolError("wireless transport is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        try:
            return self._submit(self._request(method, payload, timeout_s)).result()
        except WirelessProtocolError:
            raise
        except Exception as error:
            raise WirelessProtocolError(f"wireless request failed: {error}") from error

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._submit(self._shutdown()).result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()

    async def _start(self) -> None:
        from wireless_comm import Comm

        config = self._runtime_config
        self._comm = await Comm.create(
            local=config.local,
            peers=config.peers,
            bind_host=config.bind_host,
            config=config.comm,
        )
        try:
            self._server = self._comm.peer(self._server_node_id)
        except (KeyError, ValueError):
            await self._comm.close()
            raise
        self._responses = asyncio.create_task(
            self._response_loop(),
            name="wireless-rpc-responses",
        )

    async def _request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        from wireless_comm import CommOptions, OperationTimeoutError

        if self._failure is not None:
            raise WirelessProtocolError(f"wireless response loop failed: {self._failure}")
        deadline = self._loop.time() + timeout_s
        rpc_id = uuid.uuid4().hex
        future = self._loop.create_future()
        self._pending[rpc_id] = future
        metadata = {
            "schema": self._rpc_schema,
            "kind": "request",
            "rpc_id": rpc_id,
            "method": method,
        }
        if self._token is not None:
            metadata["token"] = self._token
        try:
            await self._comm.send(
                dict(payload or {}),
                self._server,
                piggypayload=metadata,
                options=CommOptions(tag=self._request_tag, timeout=timeout_s),
            )
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(asyncio.shield(future), remaining)
        except (asyncio.TimeoutError, OperationTimeoutError) as error:
            raise WirelessProtocolError(
                f"{method} timed out after {timeout_s:g} seconds",
                code="timeout",
            ) from error
        finally:
            self._pending.pop(rpc_id, None)

    async def _response_loop(self) -> None:
        from wireless_comm import CommError, CommOptions

        try:
            while True:
                payload, metadata = await self._comm.recv(
                    self._server,
                    CommOptions(tag=self._response_tag),
                )
                if not isinstance(metadata, Mapping):
                    raise WirelessProtocolError("wireless response is missing RPC metadata")
                if metadata.get("schema") != self._rpc_schema or metadata.get("kind") != "response":
                    raise WirelessProtocolError("wireless response has an unsupported RPC envelope")
                rpc_id = metadata.get("rpc_id")
                if not isinstance(rpc_id, str):
                    raise WirelessProtocolError("wireless response is missing rpc_id")
                future = self._pending.get(rpc_id)
                if future is None:
                    continue
                status = metadata.get("status")
                if isinstance(status, bool) or not isinstance(status, int):
                    future.set_exception(WirelessProtocolError("wireless response status is invalid"))
                elif not 200 <= status < 300:
                    message = payload.get("message") if isinstance(payload, Mapping) else None
                    future.set_exception(
                        WirelessProtocolError(
                            str(message or "wireless RPC request failed"),
                            status=status,
                            code=str(metadata.get("code") or "remote_error"),
                        )
                    )
                elif not isinstance(payload, Mapping):
                    future.set_exception(WirelessProtocolError("wireless response payload must be an object"))
                else:
                    future.set_result(dict(payload))
        except asyncio.CancelledError:
            raise
        except (CommError, WirelessProtocolError, ConnectionError, OSError) as error:
            self._failure = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(WirelessProtocolError(f"wireless response loop failed: {error}"))

    async def _shutdown(self) -> None:
        if self._responses is not None:
            self._responses.cancel()
            await asyncio.gather(self._responses, return_exceptions=True)
        if self._comm is not None:
            await self._comm.close()

    def _submit(self, coroutine: Any) -> concurrent.futures.Future[Any]:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


__all__ = [
    "WirelessProtocolError",
    "WirelessRpcTransport",
    "WirelessTransport",
]
