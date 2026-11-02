"""Encoded RGB and registered-depth navigation frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from embodirun.types import Metadata

from ._validation import (
    NavigationContractError,
    encoded_bytes,
    finite,
    metadata,
    positive_int,
    sequence,
)

MAX_IMAGE_BYTES = 16 * 1024 * 1024
_RGB_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


@dataclass(frozen=True, slots=True)
class EncodedRGBFrame:
    sequence: int
    captured_at_s: float
    data: bytes
    media_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", sequence(self.sequence, "rgb.sequence"))
        object.__setattr__(
            self,
            "captured_at_s",
            finite(self.captured_at_s, "rgb.captured_at_s", minimum=0.0),
        )
        if self.media_type not in _RGB_MEDIA_TYPES:
            raise NavigationContractError(f"unsupported RGB media type: {self.media_type!r}")
        signature = b"\xff\xd8" if self.media_type == "image/jpeg" else None
        object.__setattr__(
            self,
            "data",
            encoded_bytes(self.data, "rgb.data", MAX_IMAGE_BYTES, signature),
        )
        if (self.width is None) != (self.height is None):
            raise NavigationContractError("RGB width and height must be provided together")
        if self.width is not None:
            object.__setattr__(self, "width", positive_int(self.width, "rgb.width"))
            object.__setattr__(self, "height", positive_int(self.height, "rgb.height"))


@dataclass(frozen=True, slots=True)
class EncodedDepthFrame:
    sequence: int
    captured_at_s: float
    data: bytes
    width: int
    height: int
    scale_m: float
    registered_to_rgb: bool = True
    encoding: Literal["uint16"] = "uint16"
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", sequence(self.sequence, "depth.sequence"))
        object.__setattr__(
            self,
            "captured_at_s",
            finite(self.captured_at_s, "depth.captured_at_s", minimum=0.0),
        )
        object.__setattr__(
            self,
            "data",
            encoded_bytes(
                self.data,
                "depth.data",
                MAX_IMAGE_BYTES,
                b"\x89PNG\r\n\x1a\n",
            ),
        )
        object.__setattr__(self, "width", positive_int(self.width, "depth.width"))
        object.__setattr__(self, "height", positive_int(self.height, "depth.height"))
        object.__setattr__(
            self,
            "scale_m",
            finite(
                self.scale_m,
                "depth.scale_m",
                minimum=math.nextafter(0.0, 1.0),
                maximum=0.1,
            ),
        )
        if self.encoding != "uint16":
            raise NavigationContractError("depth encoding must be 'uint16'")
        if self.registered_to_rgb is not True:
            raise NavigationContractError("depth must be registered to RGB")
        object.__setattr__(self, "metadata", metadata(self.metadata, "depth.metadata"))


__all__ = ["MAX_IMAGE_BYTES", "EncodedDepthFrame", "EncodedRGBFrame"]
