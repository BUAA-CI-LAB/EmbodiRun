"""Trust-boundary validation for negotiated RealSense frames and profiles."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple  # noqa: UP035

from .types import CameraStreamError, VideoProfile


def validated_video_profile(
    pipeline_profile: Any,
    stream_kind: Any,
    expected_format: Any,
    pixel_format_name: str,
    bytes_per_pixel: int,
    *,
    width: int,
    height: int,
    fps: int,
) -> VideoProfile:
    try:
        profile = pipeline_profile.get_stream(stream_kind).as_video_stream_profile()
        actual_width = int(profile.width())
        actual_height = int(profile.height())
        actual_fps = int(profile.fps())
        actual_format = profile.format()
    except Exception as error:
        raise CameraStreamError(f"could not inspect active RealSense profile: {error}") from error
    mismatches = []
    if actual_width != width:
        mismatches.append(f"width={actual_width}")
    if actual_height != height:
        mismatches.append(f"height={actual_height}")
    if actual_fps != fps:
        mismatches.append(f"fps={actual_fps}")
    if actual_format != expected_format:
        mismatches.append(f"format={actual_format}")
    if mismatches:
        raise CameraStreamError("unsafe RealSense profile: {}".format("; ".join(mismatches)))
    bytes_per_line = actual_width * bytes_per_pixel
    return VideoProfile(
        width=actual_width,
        height=actual_height,
        pixel_format=pixel_format_name,
        bytes_per_line=bytes_per_line,
        size_image=bytes_per_line * actual_height,
        fps=actual_fps,
    )


def validated_frame_bytes(
    frame: Any,
    profile: VideoProfile,
    label: str,
) -> bytes:
    if not frame:
        raise CameraStreamError(f"aligned RealSense {label} frame is missing")
    try:
        width = int(frame.get_width())
        height = int(frame.get_height())
        stride = int(frame.get_stride_in_bytes())
        raw = bytes(frame.get_data())
    except Exception as error:
        raise CameraStreamError(f"could not read RealSense {label} frame: {error}") from error
    mismatches = []
    if width != profile.width:
        mismatches.append(f"width={width}")
    if height != profile.height:
        mismatches.append(f"height={height}")
    if stride != profile.bytes_per_line:
        mismatches.append(f"bytes_per_line={stride}")
    if len(raw) != profile.size_image:
        mismatches.append(f"size_image={len(raw)}")
    if mismatches:
        raise CameraStreamError(f"unsafe RealSense {label} frame: {'; '.join(mismatches)}")
    return raw


def validated_depth_scale(pipeline_profile: Any, expected_depth_scale: float) -> float:
    try:
        actual_depth_scale = float(pipeline_profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception as error:
        raise CameraStreamError(f"could not read RealSense depth scale: {error}") from error
    if actual_depth_scale <= 0 or not math.isclose(
        actual_depth_scale,
        expected_depth_scale,
        rel_tol=1e-4,
        abs_tol=1e-9,
    ):
        raise CameraStreamError(
            f"RealSense depth scale {actual_depth_scale} does not match confirmed {expected_depth_scale}"
        )
    return actual_depth_scale


def realsense_device_name(
    rs: Any,
    pipeline_profile: Any,
    serial: Optional[str],  # noqa: UP045
) -> str:
    if serial:
        return f"realsense:{serial}"
    try:
        detected = pipeline_profile.get_device().get_info(rs.camera_info.serial_number)
        return f"realsense:{detected}"
    except (OSError, RuntimeError, ValueError):
        return "realsense"


def validated_source_identity(
    frame: Any,
    last_frame_number: Optional[int],  # noqa: UP045
    last_timestamp_ms: Optional[float],  # noqa: UP045
) -> Tuple[int, float]:  # noqa: UP006
    try:
        frame_number = int(frame.get_frame_number())
        timestamp_ms = float(frame.get_timestamp())
    except Exception as error:
        raise CameraStreamError(f"could not read RealSense source identity: {error}") from error
    if last_frame_number is not None and frame_number <= last_frame_number:
        raise CameraStreamError("RealSense source frame number did not increase")
    if last_timestamp_ms is not None and timestamp_ms <= last_timestamp_ms:
        raise CameraStreamError("RealSense source timestamp did not increase")
    return frame_number, timestamp_ms


__all__ = [
    "realsense_device_name",
    "validated_depth_scale",
    "validated_frame_bytes",
    "validated_source_identity",
    "validated_video_profile",
]
