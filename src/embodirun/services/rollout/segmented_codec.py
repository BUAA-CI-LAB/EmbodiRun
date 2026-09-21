"""Frame the pinned Wireless payload codec for transport comparison.

Zenoh receives one contiguous message, so joining segments is an explicit copy.
The caller registers its own dataclasses; Deploy knows no trainer/model types.
Requires the optional ``wireless`` extra. Unsafe pickle remains disabled.
"""

from __future__ import annotations

import struct

_HEADER = struct.Struct("!8sII")
_MAGIC = b"RLSEG01\0"


class SegmentedCodec:
    def __init__(self, register=None, *, max_message_bytes=256 * 1024**2):
        from wireless_comm.codec import CodecRegistry, PayloadCodec
        from wireless_comm.types import CommConfig

        self.config = CommConfig(max_message_bytes=max_message_bytes)
        registry = CodecRegistry()
        if register is not None:
            register(registry)
        self.codec = PayloadCodec(registry, self.config)

    def encode(self, value):
        encoded = self.codec.encode(value, None)
        lengths = [part.nbytes for part in encoded.segments]
        header = _HEADER.pack(_MAGIC, len(encoded.manifest), len(lengths))
        table = struct.pack(f"!{len(lengths)}Q", *lengths)
        size = len(header) + len(table) + len(encoded.manifest) + sum(lengths)
        if size > self.config.max_message_bytes:
            raise ValueError("segmented message exceeds configured byte limit")
        return b"".join([header, table, encoded.manifest, *(part.view for part in encoded.segments)])

    def decode(self, raw):
        from wireless_comm.codec import BufferSegment, EncodedPayload

        if not _HEADER.size <= len(raw) <= self.config.max_message_bytes:
            raise ValueError("invalid segmented message size")
        magic, manifest_size, count = _HEADER.unpack_from(raw)
        table_end = _HEADER.size + count * 8
        manifest_end = table_end + manifest_size
        if (
            magic != _MAGIC
            or count > self.config.max_segments
            or manifest_size > self.config.max_manifest_bytes
            or manifest_end > len(raw)
        ):
            raise ValueError("invalid segmented message header")
        lengths = struct.unpack_from(f"!{count}Q", raw, _HEADER.size)
        if manifest_end + sum(lengths) != len(raw):
            raise ValueError("invalid segmented message lengths")
        # Writable owned storage matches Wireless's receive buffers for tensors.
        owner = bytearray(raw)
        view = memoryview(owner)
        segments = []
        offset = manifest_end
        for size in lengths:
            segments.append(BufferSegment(view[offset : offset + size], owner))
            offset += size
        payload = EncodedPayload(bytes(view[table_end:manifest_end]), tuple(segments))
        return self.codec.decode(payload).object
