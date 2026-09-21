"""Parsing and exact validation of negotiated V4L2 raw-video formats."""

from __future__ import annotations

import re

from .types import CameraStreamError, VideoProfile


def parse_v4l2_format(output: str) -> VideoProfile:
    """Parse every raw-video field needed to delimit an unheadered pipe."""

    patterns = {
        "dimensions": r"Width/Height\s*:\s*(\d+)\s*/\s*(\d+)",
        "pixel_format": r"Pixel Format\s*:\s*'([^']+)'",
        "bytes_per_line": r"Bytes per Line\s*:\s*(\d+)",
        "size_image": r"Size Image\s*:\s*(\d+)",
    }
    matches = {name: re.search(pattern, output) for name, pattern in patterns.items()}
    missing = [name for name, match in matches.items() if match is None]
    if missing:
        raise CameraStreamError("v4l2-ctl format output is missing: {}".format(", ".join(missing)))
    dimensions = matches["dimensions"]
    pixel_format = matches["pixel_format"]
    bytes_per_line = matches["bytes_per_line"]
    size_image = matches["size_image"]
    assert dimensions is not None
    assert pixel_format is not None
    assert bytes_per_line is not None
    assert size_image is not None
    return VideoProfile(
        width=int(dimensions.group(1)),
        height=int(dimensions.group(2)),
        pixel_format=pixel_format.group(1).strip(),
        bytes_per_line=int(bytes_per_line.group(1)),
        size_image=int(size_image.group(1)),
    )


def validated_v4l2_format(
    output: str,
    *,
    width: int,
    height: int,
    pixel_format: str,
) -> VideoProfile:
    actual = parse_v4l2_format(output)
    expected_bytes_per_line = width * 2
    expected = VideoProfile(
        width=width,
        height=height,
        pixel_format=pixel_format,
        bytes_per_line=expected_bytes_per_line,
        size_image=expected_bytes_per_line * height,
    )
    mismatches = []
    for field in (
        "width",
        "height",
        "pixel_format",
        "bytes_per_line",
        "size_image",
    ):
        if getattr(actual, field) != getattr(expected, field):
            mismatches.append(f"{field}={getattr(actual, field)} (expected {getattr(expected, field)})")
    if mismatches:
        raise CameraStreamError("unsafe V4L2 format negotiation: {}".format("; ".join(mismatches)))
    return actual


__all__ = ["parse_v4l2_format", "validated_v4l2_format"]
