"""Public-observation watchdogs for bounded, locally closed-loop navigation."""

from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EventPolicy:
    max_chunks: int = 32
    stalled_chunks: int = 4
    collision_limit: int = 4
    loop_history: int = 12
    loop_min_gap: int = 4
    similarity_threshold: float = 2.0

    def __post_init__(self):
        for name in ("max_chunks", "stalled_chunks", "collision_limit", "loop_history", "loop_min_gap"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.loop_min_gap >= self.loop_history or not 0 < self.similarity_threshold <= 255:
            raise ValueError("invalid visual watchdog configuration")


def thumbnail(image):
    from PIL import Image

    with Image.open(io.BytesIO(image)) as frame:
        return frame.convert("L").resize((32, 24), Image.Resampling.BOX).tobytes()


def difference(left, right):
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


class EventMonitor:
    """Detect stalls from RGB and collisions, without map, pose or goal metrics."""

    def __init__(self, initial_image, policy: EventPolicy):
        self.policy = policy
        self.frames = deque([thumbnail(initial_image)], maxlen=policy.loop_history)
        self.stalled = self.collisions = 0

    def update(self, feedback):
        forwards = feedback.actions.count(1)
        if feedback.collisions and forwards and feedback.collisions / forwards >= 0.5:
            self.collisions += feedback.collisions
        else:
            self.collisions = 0
        if self.collisions >= self.policy.collision_limit:
            return "blocked"
        frame = thumbnail(feedback.image)
        threshold = self.policy.similarity_threshold
        self.stalled = self.stalled + 1 if difference(frame, self.frames[-1]) <= threshold else 0
        if self.stalled >= self.policy.stalled_chunks:
            return "stalled"
        previous = list(self.frames)
        for index in range(len(previous) - self.policy.loop_min_gap + 1):
            if difference(frame, previous[index]) <= threshold and any(
                difference(frame, other) > threshold * 4 for other in previous[index + 1 :]
            ):
                return "loop"
        self.frames.append(frame)
        return None
