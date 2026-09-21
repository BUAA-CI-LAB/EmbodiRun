"""Bounded registered-memory tensor transfers with caller-owned control routing.

Only tensor buffers use UCX. Callers provide ordered asynchronous metadata send
and receive functions, and serialize operations on this half-duplex connection.
Destination tensors own their memory; a chunk credit is sent only after the
copy out of the reusable registered arena has completed. No model or robot API.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import time
import uuid

_PROTOCOL = "embodirun.nixl-tensors.v1"
# A timed-out receive may still be the target of a remote WRITE. Retain the
# registration and its storage until worker-process exit; never reuse it.
_FAILED_OWNERS = []


class NixlTensorTransport:
    """One persistent UCX arena; CPU or CUDA tensors, no silent device fallback."""

    def __init__(
        self,
        device="cpu",
        *,
        chunk_bytes=16 * 1024**2,
        max_payload_bytes=16 * 1024**3,
        timeout_s=30,
    ):
        import torch
        from nixl._api import nixl_agent, nixl_agent_config

        if not (type(chunk_bytes) is int and 1 <= chunk_bytes <= 256 * 1024**2):
            raise ValueError("NIXL chunk_bytes must be between 1 byte and 256 MiB")
        if type(max_payload_bytes) is not int or max_payload_bytes < chunk_bytes:
            raise ValueError("NIXL payload limit must be at least the chunk capacity")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("NIXL timeout must be finite and positive")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("NIXL transport supports CPU and CUDA only")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.chunk_bytes, self.max_payload_bytes, self.timeout_s = (
            chunk_bytes,
            max_payload_bytes,
            timeout_s,
        )
        self.name = f"embodirun-{os.getpid()}-{uuid.uuid4().hex}"
        self.peer = None
        self.sequence = 0
        self.closed = self.failed = False
        self._busy = False
        self._send = self._recv = None
        self._handles = []
        self.last_metrics = {}
        self.dtypes = {
            str(dtype): dtype
            for dtype in (
                torch.bool,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            )
        }
        self.arena = torch.empty(chunk_bytes, dtype=torch.uint8, device=self.device)
        self.agent = nixl_agent(self.name, nixl_agent_config(backends=["UCX"]))
        started = time.perf_counter()
        self.registration = self.agent.register_memory(self.arena, backends=["UCX"])
        self.setup_metrics = {
            "registration_s": time.perf_counter() - started,
            "arena_bytes": chunk_bytes,
            "nixl_version": importlib.metadata.version("nixl"),
            "device": str(self.device),
        }

    def _check(self):
        if self.failed:
            raise RuntimeError("NIXL connection failed; restart the worker before resynchronizing")
        if self.closed or self._busy:
            raise RuntimeError("NIXL connection is closed or has another active operation")

    async def _guard(self, operation):
        self._check()
        self._busy = True
        try:
            return await asyncio.wait_for(operation(), timeout=self.timeout_s)
        except BaseException:
            self.invalidate()
            raise
        finally:
            self._busy = False

    def invalidate(self):
        """Fail this connection after transport or caller schema validation errors."""
        if not self.failed:
            self.failed = True
            _FAILED_OWNERS.append(self)

    async def _device_ready(self):
        if self.device.type == "cuda":
            event = self.torch.cuda.Event()
            event.record(self.torch.cuda.current_stream(self.device))
            while not event.query():
                await asyncio.sleep(0.0001)

    @property
    def mem_type(self):
        return "DRAM" if self.device.type == "cpu" else "VRAM"

    @property
    def device_id(self):
        return self.device.index if self.device.type == "cuda" else 0

    async def connect(self, send_control, recv_control, *, initiator):
        """Exchange only arena metadata, using the caller's existing control path."""

        async def operation():
            if self.peer is not None:
                raise RuntimeError("NIXL connection already initialized")
            self._send, self._recv = send_control, recv_control
            hello = {
                "protocol": _PROTOCOL,
                "name": self.name,
                "metadata": base64.b64encode(self.agent.get_agent_metadata()).decode(),
                "address": self.arena.data_ptr(),
                "capacity": self.chunk_bytes,
                "mem_type": self.mem_type,
                "device_id": self.device_id,
            }
            if initiator:
                await self._send(hello)
                peer = await self._recv()
            else:
                peer = await self._recv()
                await self._send(hello)
            if (
                peer.get("protocol") != _PROTOCOL
                or peer.get("mem_type") not in {"DRAM", "VRAM"}
                or type(peer.get("capacity")) is not int
                or not 1 <= peer["capacity"] <= 256 * 1024**2
                or type(peer.get("address")) is not int
                or peer["address"] <= 0
                or type(peer.get("device_id")) is not int
                or peer["device_id"] < 0
                or not isinstance(peer.get("name"), str)
                or peer["name"] == self.name
            ):
                raise ValueError("invalid NIXL peer metadata")
            loaded = self.agent.add_remote_agent(base64.b64decode(peer["metadata"], validate=True))
            if isinstance(loaded, bytes):
                loaded = loaded.decode()
            if loaded != peer["name"]:
                raise ValueError("NIXL metadata identity differs from control peer")
            self.peer = peer
            self.capacity = min(self.chunk_bytes, peer["capacity"])
            self.session = hashlib.sha256("\0".join(sorted((self.name, peer["name"]))).encode()).hexdigest()

        await self._guard(operation)

    def _stamp(self, kind, chunk=None):
        value = {"kind": kind, "session": self.session, "sequence": self.sequence}
        if chunk is not None:
            value["chunk"] = chunk
        return value

    def _token(self, chunk):
        return json.dumps(self._stamp("chunk", chunk), sort_keys=True).encode()

    def _describe(self, tensors):
        specs, views = [], []
        for tensor in tensors:
            if (
                not isinstance(tensor, self.torch.Tensor)
                or tensor.device != self.device
                or str(tensor.dtype) not in self.dtypes
            ):
                raise ValueError("tensor dtype/device differs from configured NIXL transport")
            value = tensor.detach().contiguous()
            view = value.reshape(-1).view(self.torch.uint8)
            views.append(view)
            specs.append(
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": view.numel(),
                }
            )
        self._validate_specs(specs)
        return specs, views

    def _validate_specs(self, specs):
        if not isinstance(specs, list) or not 1 <= len(specs) <= 4096:
            raise ValueError("invalid NIXL tensor count")
        total = 0
        for item in specs:
            shape = item.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) > 16
                or any(type(n) is not int or n < 0 for n in shape)
                or item.get("dtype") not in self.dtypes
            ):
                raise ValueError("invalid NIXL tensor shape or dtype")
            size = math.prod(shape) * self.torch.empty(0, dtype=self.dtypes[item["dtype"]]).element_size()
            if type(item.get("bytes")) is not int or size != item["bytes"]:
                raise ValueError("NIXL tensor byte length differs from shape/dtype")
            total += size
        if total > self.max_payload_bytes:
            raise ValueError("NIXL tensor payload exceeds configured limit")
        return total

    def _copy_chunk(self, views, start, size, *, into_arena):
        position = 0
        for view in views:
            end = position + view.numel()
            first, last = max(start, position), min(start + size, end)
            if first < last:
                tensor_slice = view[first - position : last - position]
                arena_slice = self.arena[first - start : last - start]
                if into_arena:
                    arena_slice.copy_(tensor_slice, non_blocking=True)
                else:
                    tensor_slice.copy_(arena_slice, non_blocking=True)
            position = end

    async def _write_chunk(self, size, token):
        local = self.agent.get_xfer_descs([(self.arena.data_ptr(), size, self.device_id)], self.mem_type)
        remote = self.agent.get_xfer_descs(
            [(self.peer["address"], size, self.peer["device_id"])],
            self.peer["mem_type"],
        )
        handle = self.agent.initialize_xfer("WRITE", local, remote, self.peer["name"], backends=["UCX"])
        self._handles.append(handle)
        # On cancellation/timeout, retain the handle and arena for process exit.
        # NIXL can refuse to release a transfer that has not finished aborting.
        state = self.agent.transfer(handle, token)
        while state == "PROC":
            await asyncio.sleep(0.0001)
            state = self.agent.check_xfer_state(handle)
        if state != "DONE":
            raise RuntimeError(f"NIXL WRITE failed: {state}")
        self.agent.release_xfer_handle(handle)
        self._handles.remove(handle)

    async def _notification(self, token):
        while True:
            notifications = self.agent.get_new_notifs(backends=["UCX"])
            found = False
            for name, messages in notifications.items():
                if isinstance(name, bytes):
                    name = name.decode()
                for message in messages:
                    if name != self.peer["name"] or message != token or found:
                        raise ValueError("unexpected or duplicate NIXL completion notification")
                    found = True
            if found:
                return
            await asyncio.sleep(0.0001)

    async def send(self, tensors, metadata):
        """Transfer a tensor list, returning after the receiver owns every byte."""

        async def operation():
            if self.peer is None:
                raise RuntimeError("NIXL connection is not initialized")
            started = time.perf_counter()
            specs, views = self._describe(tensors)
            total = sum(item["bytes"] for item in specs)
            await self._send({**self._stamp("offer"), "tensors": specs, "metadata": metadata})
            metrics = {
                "pack_wait_s": 0.0,
                "transfer_wait_s": 0.0,
                "credit_wait_s": 0.0,
                "wire_bytes": total,
                "chunks": 0,
            }
            for chunk, offset in enumerate(range(0, total, self.capacity)):
                size = min(self.capacity, total - offset)
                before = time.perf_counter()
                self._copy_chunk(views, offset, size, into_arena=True)
                await self._device_ready()
                metrics["pack_wait_s"] += time.perf_counter() - before
                before = time.perf_counter()
                if await self._recv() != self._stamp("ready", chunk):
                    raise ValueError("NIXL receiver credit does not match chunk")
                metrics["credit_wait_s"] += time.perf_counter() - before
                before = time.perf_counter()
                await self._write_chunk(size, self._token(chunk))
                metrics["transfer_wait_s"] += time.perf_counter() - before
                before = time.perf_counter()
                if await self._recv() != self._stamp("copied", chunk):
                    raise ValueError("NIXL receiver did not acknowledge copied chunk")
                metrics["credit_wait_s"] += time.perf_counter() - before
                metrics["chunks"] += 1
            if await self._recv() != self._stamp("owned"):
                raise ValueError("NIXL receiver did not acknowledge payload ownership")
            self.sequence += 1
            self.last_metrics = dict(metrics, total_s=time.perf_counter() - started)

        await self._guard(operation)

    async def recv(self):
        """Receive fresh owned tensors; arena reuse cannot mutate returned tensors."""

        async def operation():
            if self.peer is None:
                raise RuntimeError("NIXL connection is not initialized")
            started = time.perf_counter()
            offer = await self._recv()
            if {key: offer.get(key) for key in ("kind", "session", "sequence")} != self._stamp("offer"):
                raise ValueError("NIXL offer session or sequence differs")
            total = self._validate_specs(offer["tensors"])
            tensors = [
                self.torch.empty(item["shape"], dtype=self.dtypes[item["dtype"]], device=self.device)
                for item in offer["tensors"]
            ]
            views = [tensor.reshape(-1).view(self.torch.uint8) for tensor in tensors]
            metrics = {
                "notification_wait_s": 0.0,
                "unpack_wait_s": 0.0,
                "wire_bytes": total,
                "chunks": 0,
            }
            for chunk, offset in enumerate(range(0, total, self.capacity)):
                size = min(self.capacity, total - offset)
                await self._send(self._stamp("ready", chunk))
                before = time.perf_counter()
                await self._notification(self._token(chunk))
                metrics["notification_wait_s"] += time.perf_counter() - before
                before = time.perf_counter()
                self._copy_chunk(views, offset, size, into_arena=False)
                await self._device_ready()
                metrics["unpack_wait_s"] += time.perf_counter() - before
                await self._send(self._stamp("copied", chunk))
                metrics["chunks"] += 1
            await self._send(self._stamp("owned"))
            self.sequence += 1
            self.last_metrics = dict(metrics, total_s=time.perf_counter() - started)
            return tensors, offer["metadata"]

        return await self._guard(operation)

    def close(self):
        """Deregister an idle successful connection; failed owners require exit."""
        if self.closed:
            return
        if self.failed:
            return  # Retained in _FAILED_OWNERS until this process exits.
        self._check()
        if self.peer is not None:
            self.agent.remove_remote_agent(self.peer["name"])
        self.agent.deregister_memory(self.registration, backends=["UCX"])
        self._send = self._recv = None
        self.arena = None
        self.closed = True
