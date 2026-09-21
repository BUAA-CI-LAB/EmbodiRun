"""In-memory RGB and Z16 encoders used by camera backends.

Hardware/image dependencies are imported inside the functions so importing
the camera service, showing ``--help``, and running health tests do not require
OpenCV, NumPy, or pyrealsense2 on the host.
"""

from __future__ import annotations

import struct
import zlib

from .types import CameraStreamError, DepthAnalysis


def encode_yuyv_as_jpeg(raw: bytes, width: int, height: int, quality: int) -> bytes:
    """Convert one tightly packed YUYV frame to JPEG."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise CameraStreamError("OpenCV and NumPy are required for JPEG encoding") from error

    expected = width * height * 2
    if len(raw) != expected:
        raise CameraStreamError(f"unexpected YUYV frame size: got {len(raw)}, expected {expected}")
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 2)
    bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise CameraStreamError("OpenCV could not encode the camera frame")
    return encoded.tobytes()


def encode_bgr_as_jpeg(raw: bytes, width: int, height: int, quality: int) -> bytes:
    """Encode one tightly packed BGR8 RealSense color frame to JPEG."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise CameraStreamError("OpenCV and NumPy are required for JPEG encoding") from error

    expected = width * height * 3
    if len(raw) != expected:
        raise CameraStreamError(f"unexpected BGR8 frame size: got {len(raw)}, expected {expected}")
    bgr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise CameraStreamError("OpenCV could not encode the RealSense color frame")
    return encoded.tobytes()


def encode_z16_as_png(raw: bytes, width: int, height: int) -> bytes:
    """Losslessly encode tightly packed little-endian Z16 as grayscale PNG."""

    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("depth PNG dimensions must be positive integers")
    expected = width * height * 2
    if len(raw) != expected:
        raise CameraStreamError(f"unexpected Z16 frame size: got {len(raw)}, expected {expected}")

    # PNG stores 16-bit grayscale samples in network byte order. RealSense Z16
    # is little-endian, so byte-swap each row and use PNG filter type zero.
    row_bytes = width * 2
    scanline_bytes = row_bytes + 1
    scanlines = bytearray(scanline_bytes * height)
    for row_index in range(height):
        source_start = row_index * row_bytes
        source_row = raw[source_start : source_start + row_bytes]
        swapped = bytearray(row_bytes)
        swapped[0::2] = source_row[1::2]
        swapped[1::2] = source_row[0::2]
        destination_start = row_index * scanline_bytes
        scanlines[destination_start] = 0
        scanlines[destination_start + 1 : destination_start + 1 + row_bytes] = swapped

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(scanlines), 3))
        + chunk(b"IEND", b"")
    )


def analyze_z16_depth(
    raw: bytes,
    width: int,
    height: int,
    depth_scale: float,
    roi_width_ratio: float,
    roi_height_ratio: float,
    min_valid_fraction: float,
    max_depth_m: float,
) -> DepthAnalysis:
    """Return median and conservative minimum distance in the center ROI."""

    try:
        import numpy as np
    except ImportError as error:
        raise CameraStreamError("NumPy is required for Z16 depth analysis") from error

    expected = width * height * 2
    if len(raw) != expected:
        raise CameraStreamError(f"unexpected Z16 frame size: got {len(raw)}, expected {expected}")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if not 0 < roi_width_ratio <= 1 or not 0 < roi_height_ratio <= 1:
        raise ValueError("ROI ratios must be in (0, 1]")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between zero and one")

    frame = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    roi_width = max(1, round(width * roi_width_ratio))
    roi_height = max(1, round(height * roi_height_ratio))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    roi = frame[y0 : y0 + roi_height, x0 : x0 + roi_width]
    distances = roi.astype(np.float64) * depth_scale
    valid_mask = (roi > 0) & (distances <= max_depth_m)
    valid_fraction = float(np.count_nonzero(valid_mask)) / float(roi.size)
    if valid_fraction < min_valid_fraction or not np.any(valid_mask):
        return DepthAnalysis(None, None, valid_fraction)

    valid_distances = distances[valid_mask]
    return DepthAnalysis(
        center_distance_m=float(np.median(valid_distances)),
        minimum_distance_m=float(np.min(valid_distances)),
        valid_fraction=valid_fraction,
    )


__all__ = [
    "analyze_z16_depth",
    "encode_bgr_as_jpeg",
    "encode_yuyv_as_jpeg",
    "encode_z16_as_png",
]
