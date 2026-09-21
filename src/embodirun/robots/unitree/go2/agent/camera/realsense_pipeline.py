"""RealSense pipeline construction for one aligned RGB-D profile."""

from __future__ import annotations

from typing import Any, Optional, Tuple  # noqa: UP035


def configured_pipeline(
    rs: Any,
    *,
    serial: Optional[str],  # noqa: UP045
    width: int,
    height: int,
    fps: int,
) -> Tuple[Any, Any]:  # noqa: UP006
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        width,
        height,
        rs.format.bgr8,
        fps,
    )
    config.enable_stream(
        rs.stream.depth,
        width,
        height,
        rs.format.z16,
        fps,
    )
    return pipeline, config


__all__ = ["configured_pipeline"]
