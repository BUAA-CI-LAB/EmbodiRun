"""Dog-side RGB/RGB-D capture service for Unitree Go2 deployments.

The public package intentionally exposes the dependency-free contracts and
service assembly. OpenCV, NumPy, and pyrealsense2 remain lazy hardware-path
imports, so importing this module is safe on deployment controllers.
"""

from .arguments import parse_args
from .cli import CameraServiceRuntime, build_runtime, main
from .depth_store import DepthStore
from .encoding import analyze_z16_depth, encode_z16_as_png
from .frame_store import FrameStore
from .pipe_io import read_exact_with_timeout
from .realsense import RealSenseWorker
from .server import CameraHTTPServer
from .settings import CameraServiceConfig, validate_bind_token
from .types import (
    CameraStreamError,
    DepthAnalysis,
    DepthSnapshot,
    FrameSnapshot,
    VideoProfile,
)
from .v4l2 import V4L2YUYVSource, V4L2Z16Source
from .v4l2_format import parse_v4l2_format
from .workers import CameraWorker, DepthWorker

__all__ = [
    "CameraHTTPServer",
    "CameraServiceConfig",
    "CameraServiceRuntime",
    "CameraStreamError",
    "CameraWorker",
    "DepthAnalysis",
    "DepthSnapshot",
    "DepthStore",
    "DepthWorker",
    "FrameSnapshot",
    "FrameStore",
    "RealSenseWorker",
    "V4L2YUYVSource",
    "V4L2Z16Source",
    "VideoProfile",
    "analyze_z16_depth",
    "build_runtime",
    "encode_z16_as_png",
    "main",
    "parse_args",
    "parse_v4l2_format",
    "read_exact_with_timeout",
    "validate_bind_token",
]
