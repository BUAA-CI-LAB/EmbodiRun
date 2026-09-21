"""CLI assembly for the Go2 camera service."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from typing import Optional

from .argument_types import (
    ENV_PREFIX,
    add_boolean_option,
    camera_profile,
    environment_bool,
    environment_value,
    port_number,
    positive_float,
    positive_int,
)
from .settings import CameraServiceConfig


def build_parser(environment: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve fresh Go2 RGB-D frames. The RealSense backend uses one "
            "pipeline and aligns depth to color; split V4L2 is diagnostic-only."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("v4l2", "realsense"),
        default=environment_value(environment, "BACKEND", "v4l2"),
    )
    parser.add_argument(
        "--host",
        "--bind",
        dest="host",
        default=environment_value(environment, "HOST", "127.0.0.1"),
        help="HTTP bind host (GO2_CAMERA_HOST)",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=environment_value(environment, "PORT", "8765"),
        help="HTTP port (GO2_CAMERA_PORT)",
    )
    parser.add_argument(
        "--token",
        default=environment_value(environment, "TOKEN"),
        help="bearer token (prefer GO2_CAMERA_TOKEN to avoid process-list exposure)",
    )
    parser.add_argument(
        "--serial",
        "--realsense-serial",
        dest="serial",
        default=environment_value(environment, "SERIAL"),
        help="optional RealSense serial (GO2_CAMERA_SERIAL)",
    )
    parser.add_argument(
        "--profile",
        type=camera_profile,
        default=environment_value(environment, "PROFILE", "640x360@15"),
        metavar="WIDTHxHEIGHT@FPS",
        help="requested RGB-D profile (GO2_CAMERA_PROFILE)",
    )
    parser.add_argument("--width", type=positive_int, default=environment_value(environment, "WIDTH"))
    parser.add_argument("--height", type=positive_int, default=environment_value(environment, "HEIGHT"))
    parser.add_argument(
        "--fps",
        "--realsense-fps",
        dest="fps",
        type=positive_int,
        default=environment_value(environment, "FPS"),
        help="capture FPS override (GO2_CAMERA_FPS)",
    )
    parser.add_argument(
        "--depth-scale",
        type=positive_float,
        default=environment_value(environment, "DEPTH_SCALE"),
        help="confirmed meters per Z16 unit (GO2_CAMERA_DEPTH_SCALE)",
    )
    try:
        depth_calibrated_default = environment_bool(environment, f"{ENV_PREFIX}DEPTH_CALIBRATED")
        aligned_default = environment_bool(environment, f"{ENV_PREFIX}RGB_DEPTH_ALIGNED")
        access_log_default = environment_bool(environment, f"{ENV_PREFIX}ACCESS_LOG")
    except ValueError as error:
        parser.error(str(error))
    add_boolean_option(parser, "depth-calibrated", default=depth_calibrated_default)
    add_boolean_option(
        parser,
        "rgb-depth-aligned",
        default=aligned_default,
        help_text="diagnostic operator claim; RealSense alignment is verified automatically",
    )
    add_boolean_option(parser, "access-log", default=access_log_default)
    parser.add_argument("--device", default=environment_value(environment, "DEVICE", "/dev/video4"))
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=environment_value(environment, "WARMUP_FRAMES", "10"),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=environment_value(environment, "JPEG_QUALITY", "85"),
    )
    parser.add_argument(
        "--jpeg-fps",
        type=positive_float,
        default=environment_value(environment, "JPEG_FPS", "5"),
    )
    parser.add_argument(
        "--retry-delay",
        type=positive_float,
        default=environment_value(environment, "RETRY_DELAY", "2"),
    )
    parser.add_argument(
        "--frame-timeout",
        type=positive_float,
        default=environment_value(environment, "FRAME_TIMEOUT", "2"),
    )
    parser.add_argument(
        "--format-timeout",
        type=positive_float,
        default=environment_value(environment, "FORMAT_TIMEOUT", "5"),
    )
    parser.add_argument(
        "--max-frame-age",
        type=positive_float,
        default=environment_value(environment, "MAX_FRAME_AGE", "2"),
    )
    parser.add_argument(
        "--depth-device",
        default=environment_value(environment, "DEPTH_DEVICE", "/dev/video0"),
    )
    parser.add_argument(
        "--depth-width",
        type=positive_int,
        default=environment_value(environment, "DEPTH_WIDTH"),
    )
    parser.add_argument(
        "--depth-height",
        type=positive_int,
        default=environment_value(environment, "DEPTH_HEIGHT"),
    )
    parser.add_argument(
        "--depth-warmup-frames",
        type=int,
        default=environment_value(environment, "DEPTH_WARMUP_FRAMES", "10"),
    )
    parser.add_argument(
        "--depth-fps",
        type=positive_float,
        default=environment_value(environment, "DEPTH_FPS", "10"),
    )
    parser.add_argument(
        "--max-depth-age",
        type=positive_float,
        default=environment_value(environment, "MAX_DEPTH_AGE", "0.5"),
    )
    parser.add_argument(
        "--max-depth-m",
        type=positive_float,
        default=environment_value(environment, "MAX_DEPTH_M", "10"),
    )
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,  # noqa: UP045
    *,
    environ: Optional[Mapping[str, str]] = None,  # noqa: UP045
) -> CameraServiceConfig:
    """Resolve CLI values over ``GO2_CAMERA_*`` environment defaults."""

    environment = os.environ if environ is None else environ
    parser = build_parser(environment)
    namespace = parser.parse_args(argv)
    profile_width, profile_height, profile_fps = namespace.profile
    width = profile_width if namespace.width is None else namespace.width
    height = profile_height if namespace.height is None else namespace.height
    fps = profile_fps if namespace.fps is None else namespace.fps
    depth_width = width if namespace.depth_width is None else namespace.depth_width
    depth_height = height if namespace.depth_height is None else namespace.depth_height
    config = CameraServiceConfig(
        backend=namespace.backend,
        host=namespace.host,
        port=namespace.port,
        token=namespace.token,
        serial=namespace.serial,
        width=width,
        height=height,
        fps=fps,
        device=namespace.device,
        warmup_frames=namespace.warmup_frames,
        jpeg_quality=namespace.jpeg_quality,
        jpeg_fps=namespace.jpeg_fps,
        retry_delay=namespace.retry_delay,
        frame_timeout=namespace.frame_timeout,
        format_timeout=namespace.format_timeout,
        max_frame_age=namespace.max_frame_age,
        depth_device=namespace.depth_device,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_warmup_frames=namespace.depth_warmup_frames,
        depth_scale=namespace.depth_scale,
        depth_calibrated=namespace.depth_calibrated,
        rgb_depth_alignment_claimed=namespace.rgb_depth_aligned,
        depth_fps=namespace.depth_fps,
        max_depth_age=namespace.max_depth_age,
        max_depth_m=namespace.max_depth_m,
        access_log=namespace.access_log,
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    return config


__all__ = ["build_parser", "parse_args"]
