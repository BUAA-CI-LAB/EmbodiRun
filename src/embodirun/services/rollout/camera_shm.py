"""Same-host immutable camera snapshots in an explicitly owned mmap file.

Use a path under /dev/shm for shared RAM. A reader never owns the file lifetime,
avoiding multiprocessing resource-tracker unlinking on Python 3.10–3.12.
Snapshots use a process lock and copy bytes to give consumers stable ownership.
This module has no camera, robot or action execution dependencies.
"""

from __future__ import annotations

import base64
import fcntl
import json
import mmap
import os
import stat
import struct
import threading
from pathlib import Path

_HEADER_SIZE = 8192
_HEADER = struct.Struct("<8sIIII")
_SLOT = struct.Struct("<II")
_MAGIC = b"RLCAM01\0"


class SharedCameraStore:
    """Publish/read latest and fixed recorded generations without JSON image data."""

    @staticmethod
    def clock_identity() -> dict:
        """Identify the host boot and Linux time namespace for local timestamps."""
        return {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "time_namespace": os.stat("/proc/self/ns/time").st_ino,
        }

    def __init__(
        self,
        path: str | Path,
        *,
        create: bool = False,
        record_capacity: int = 120,
        slot_bytes: int = 2 * 1024**2,
    ) -> None:
        self.path = Path(path)
        self.writable = create
        self._clock_identity = self.clock_identity()
        self._lock = threading.Lock()
        self._closed = False
        if create and not (0 < record_capacity <= 4096 and 1024 <= slot_bytes <= 64 * 1024**2):
            raise ValueError("Invalid camera shared-memory capacity")
        if not create and not stat.S_ISREG(self.path.lstat().st_mode):
            raise ValueError("Camera shared-memory readers require a regular file")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY
        self._fd = os.open(self.path, flags | os.O_NOFOLLOW, 0o600)
        try:
            if create:
                os.ftruncate(self._fd, _HEADER_SIZE + (record_capacity + 1) * slot_bytes)
            self._map = mmap.mmap(self._fd, 0, access=mmap.ACCESS_WRITE if create else mmap.ACCESS_READ)
            if create:
                self.capacity, self.slot_bytes = record_capacity, slot_bytes
                self._write_header({"state": "starting", "motor_access": False}, 0)
            else:
                magic, self.capacity, self.slot_bytes, _, _ = _HEADER.unpack_from(self._map)
                if magic != _MAGIC or len(self._map) != _HEADER_SIZE + (self.capacity + 1) * self.slot_bytes:
                    raise ValueError("Invalid camera shared-memory layout")
        except BaseException:
            if hasattr(self, "_map"):
                self._map.close()
            os.close(self._fd)
            raise

    @property
    def allocated_bytes(self) -> int:
        """Mapped capacity, including fixed recorded slots and the latest slot."""
        return len(self._map)

    def _encode_header(self, status: dict, count: int) -> bytes:
        status = dict(
            status,
            recorded=count,
            record_count_target=self.capacity,
            clock_identity=self._clock_identity,
        )
        metadata = json.dumps(status, allow_nan=False, separators=(",", ":")).encode()
        if len(metadata) + _HEADER.size > _HEADER_SIZE:
            raise ValueError("Camera status exceeds shared-memory header capacity")
        return _HEADER.pack(_MAGIC, self.capacity, self.slot_bytes, count, len(metadata)) + metadata

    def _write_header(self, status: dict, count: int) -> None:
        encoded = self._encode_header(status, count)
        self._map[: len(encoded)] = encoded

    def _encode_slot(self, packet: dict) -> tuple[bytes, list[bytes], int]:
        metadata = {k: v for k, v in packet.items() if k != "images"}
        metadata["images"] = {}
        buffers = []
        total = 0
        for name, image in packet["images"].items():
            value = image.get("data")
            if value is None:
                value = base64.b64decode(image["base64"], validate=True)
            if not isinstance(value, bytes):
                raise TypeError("Camera shared-memory payloads must be bytes")
            metadata["images"][name] = {
                **{k: v for k, v in image.items() if k not in {"data", "base64", "offset", "length"}},
                "offset": total,
                "length": len(value),
            }
            buffers.append(value)
            total += len(value)
        description = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
        if _SLOT.size + len(description) + total > self.slot_bytes:
            raise ValueError("Camera packet exceeds shared-memory slot capacity")
        return description, buffers, total

    def _write_slot(self, index: int, encoded: tuple[bytes, list[bytes], int]) -> None:
        description, buffers, total = encoded
        offset = _HEADER_SIZE + index * self.slot_bytes
        cursor = offset + _SLOT.size
        self._map[cursor : cursor + len(description)] = description
        cursor += len(description)
        for value in buffers:
            self._map[cursor : cursor + len(value)] = value
            cursor += len(value)
        self._map[offset : offset + _SLOT.size] = _SLOT.pack(len(description), total)

    def publish(self, packet: dict, status: dict, *, record: bool = False) -> None:
        """Commit one complete generation and optionally preserve it for replay."""
        if not self.writable or self._closed:
            raise RuntimeError("Camera store is not open for publishing")
        with self._lock:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                _, _, _, count, _ = _HEADER.unpack_from(self._map)
                if record and count >= self.capacity:
                    raise ValueError("Camera recording capacity exhausted")
                next_count = count + int(record)
                # Validate both descriptions before mutating any visible slot.
                # The same encoded metadata is reused for latest and recording.
                header = self._encode_header(status, next_count)
                encoded = self._encode_slot(packet)
                self._write_slot(0, encoded)
                if record:
                    self._write_slot(count + 1, encoded)
                self._map[: len(header)] = header
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def get(self, path: str) -> dict:
        """Read a stable snapshot; clients cannot publish, execute or unlink."""
        if self._closed:
            raise RuntimeError("Camera store is closed")
        with self._lock:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
            try:
                _, _, _, count, size = _HEADER.unpack_from(self._map)
                if not 0 < size <= _HEADER_SIZE - _HEADER.size:
                    raise ValueError("Invalid camera status length")
                status = json.loads(self._map[_HEADER.size : _HEADER.size + size])
                if path == "/health":
                    return status
                if path == "/latest":
                    if status["state"] != "running":
                        raise LookupError("Camera publisher is not running")
                    index = 0
                elif path.startswith("/frame/") and path[7:].isdecimal():
                    if not count:
                        raise LookupError("Camera recording is not ready")
                    index = 1 + int(path[7:]) % count
                else:
                    raise KeyError(path)
                offset = _HEADER_SIZE + index * self.slot_bytes
                metadata_size, data_size = _SLOT.unpack_from(self._map, offset)
                if not metadata_size or _SLOT.size + metadata_size + data_size > self.slot_bytes:
                    raise ValueError("Invalid camera snapshot lengths")
                start = offset + _SLOT.size
                packet = json.loads(self._map[start : start + metadata_size])
                start += metadata_size
                for image in packet["images"].values():
                    position, length = image.pop("offset"), image.pop("length")
                    if position < 0 or length < 0 or position + length > data_size:
                        raise ValueError("Invalid camera image buffer bounds")
                    image["data"] = self._map[start + position : start + position + length]
                return packet
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        """Release this mapping, marking an owned publisher as stopped."""
        if self._closed:
            return
        if self.writable:
            status = self.get("/health")
            with self._lock:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
                try:
                    self._write_header(dict(status, state="stopped"), status["recorded"])
                finally:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._map.close()
        os.close(self._fd)
        self._closed = True
