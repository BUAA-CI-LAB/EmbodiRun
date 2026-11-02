"""Episode-scoped navigation observation contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from embodirun.types import Metadata

from ._validation import EPISODE_ID_PATTERN, NavigationContractError, metadata, sequence
from .frames import EncodedDepthFrame, EncodedRGBFrame

MAX_RGB_CONTEXT = 64


@dataclass(frozen=True, slots=True)
class NavigationObservation:
    episode_id: str
    sequence: int
    reset: bool
    rgb_frames: Sequence[EncodedRGBFrame]
    depth: EncodedDepthFrame | None = None
    robot_state: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not EPISODE_ID_PATTERN.fullmatch(self.episode_id):
            raise NavigationContractError(
                "episode_id must be 1-128 characters using letters, digits, '.', '_', ':', or '-'"
            )
        observation_sequence = sequence(self.sequence, "observation.sequence")
        object.__setattr__(self, "sequence", observation_sequence)
        if not isinstance(self.reset, bool):
            raise NavigationContractError("reset must be a boolean")
        if isinstance(self.rgb_frames, (str, bytes)) or not isinstance(self.rgb_frames, Sequence):
            raise NavigationContractError("rgb_frames must be a sequence")
        frames = tuple(self.rgb_frames)
        if not frames or len(frames) > MAX_RGB_CONTEXT:
            raise NavigationContractError(f"rgb_frames must contain 1-{MAX_RGB_CONTEXT} frames")
        if any(not isinstance(frame, EncodedRGBFrame) for frame in frames):
            raise NavigationContractError("rgb_frames must contain EncodedRGBFrame values")
        if any(current.captured_at_s <= previous.captured_at_s for previous, current in pairwise(frames)):
            raise NavigationContractError("RGB capture timestamps must be strictly increasing")
        if frames[-1].sequence != observation_sequence:
            raise NavigationContractError("latest RGB sequence must match observation sequence")
        object.__setattr__(self, "rgb_frames", frames)
        if self.depth is not None:
            if not isinstance(self.depth, EncodedDepthFrame):
                raise NavigationContractError("depth must be an EncodedDepthFrame or None")
            if self.depth.sequence != observation_sequence:
                raise NavigationContractError("depth sequence must match observation sequence")
            if not math.isclose(
                self.depth.captured_at_s,
                frames[-1].captured_at_s,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise NavigationContractError("depth and latest RGB capture time must match")
            if frames[-1].width is not None and (self.depth.width, self.depth.height) != (
                frames[-1].width,
                frames[-1].height,
            ):
                raise NavigationContractError("registered RGB and depth dimensions must match")
        object.__setattr__(self, "robot_state", metadata(self.robot_state, "robot_state"))
        object.__setattr__(self, "metadata", metadata(self.metadata, "metadata"))

    @property
    def latest_rgb(self) -> EncodedRGBFrame:
        return self.rgb_frames[-1]


__all__ = ["MAX_RGB_CONTEXT", "NavigationObservation"]
