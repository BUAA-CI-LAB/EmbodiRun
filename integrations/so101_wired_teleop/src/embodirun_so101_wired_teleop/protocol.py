"""Fixed-size UDP framing for wired SO-101 teleoperation.

The packet is deliberately minimal: a magic word, a sequence number and the six
joint targets. It is sent at the control rate, so anything larger would cost
bandwidth for no benefit. The sequence number lets a follower drop duplicates
and detect a stalled stream.
"""

from __future__ import annotations

import math
import struct

MAGIC = b"S101"
PROBE_MAGIC = b"S10?"
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
PACKET = struct.Struct("!4sI6f")


class ProtocolError(ValueError):
    """A datagram is not a well-formed teleoperation packet."""


def encode(sequence: int, action: dict[str, float], *, probe: bool = False) -> bytes:
    """Serialise ``action`` for the given sequence number."""

    try:
        values = [float(action[name]) for name in JOINTS]
    except KeyError as error:
        raise ProtocolError(f"action is missing joint {error}") from error
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("joint targets must be finite")
    return PACKET.pack(PROBE_MAGIC if probe else MAGIC, sequence & 0xFFFFFFFF, *values)


def decode(data: bytes, *, probe: bool = False) -> tuple[int, dict[str, float]]:
    """Parse a datagram into ``(sequence, action)``."""

    if len(data) != PACKET.size:
        raise ProtocolError(f"wrong packet size: {len(data)} != {PACKET.size}")
    magic, sequence, *values = PACKET.unpack(data)
    if magic != (PROBE_MAGIC if probe else MAGIC):
        raise ProtocolError(f"wrong packet magic: {magic!r}")
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("joint targets must be finite")
    return sequence, dict(zip(JOINTS, values, strict=True))


def is_newer(sequence: int, previous: int) -> bool:
    """Compare wrapping uint32 sequence numbers, rejecting duplicates and reordering."""

    return previous < 0 or 0 < (sequence - previous) % (1 << 32) < (1 << 31)
