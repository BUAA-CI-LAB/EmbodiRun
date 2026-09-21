"""Freshness, matching, and payload assembly for RGB-D observations."""

from __future__ import annotations

import base64
import math
from typing import Dict, Optional, Protocol, Tuple  # noqa: UP035

from .depth_store import DepthStore
from .frame_store import FrameStore
from .types import CameraStreamError, DepthSnapshot, FrameSnapshot


class ObservationSource(Protocol):
    frame_store: FrameStore
    max_frame_age: float
    depth_store: Optional[DepthStore]  # noqa: UP045
    max_depth_age: float
    calibration_confirmed: bool
    rgb_depth_alignment_claimed: bool
    backend: str
    registration_verified: bool
    depth_scale_m: Optional[float]  # noqa: UP045
    depth_device: Optional[str]  # noqa: UP045


class CameraObservationService:
    """Build fail-closed metadata and immutable matched RGB-D payloads."""

    def __init__(self, source: ObservationSource) -> None:
        self.source = source

    def depth_payload(self) -> Dict[str, object]:  # noqa: UP006
        source = self.source
        rgb_health = source.frame_store.health(source.max_frame_age)
        if source.depth_store is None:
            depth: Dict[str, object] = {  # noqa: UP006
                "available": False,
                "center_distance_m": None,
                "minimum_distance_m": None,
                "captured_at_unix": None,
                "age_s": None,
                "width": None,
                "height": None,
                "frame_sequence": 0,
                "sequence": 0,
                "valid_fraction": 0.0,
                "raw_depth_available": False,
                "capture_running": False,
                "last_error": "depth capture is disabled",
                "depth_device": source.depth_device,
                "depth_profile": None,
                "source_timestamp_ms": None,
                "source_frame_number": None,
            }
        else:
            depth = source.depth_store.status(source.max_depth_age)

        backend = str(depth.get("backend", source.backend))
        registration_verified = bool(depth.get("registration_verified", source.registration_verified))
        rgb_depth_aligned = bool(depth.get("rgb_depth_aligned", False))
        if backend != "realsense" or not registration_verified:
            rgb_depth_aligned = False
        depth_scale_m = depth.get("depth_scale_m")
        if depth_scale_m is None:
            depth_scale_m = source.depth_scale_m
        depth_sequence = depth["frame_sequence"]
        rgb_sequence = rgb_health["frame_sequence"]
        pair_matched = bool(
            isinstance(depth_sequence, int)
            and not isinstance(depth_sequence, bool)
            and depth_sequence > 0
            and depth_sequence == rgb_sequence
        )

        navigation_ready = bool(
            depth["available"]
            and source.calibration_confirmed
            and rgb_depth_aligned
            and backend == "realsense"
            and registration_verified
            and pair_matched
            and rgb_health["status"] == "ok"
            and rgb_health["rgb_profile"] is not None
            and depth["depth_profile"] is not None
        )
        depth.update(
            {
                "calibration_confirmed": source.calibration_confirmed,
                "rgb_depth_aligned": rgb_depth_aligned,
                "rgb_depth_alignment_claimed": source.rgb_depth_alignment_claimed,
                "registration_verified": registration_verified,
                "backend": backend,
                "depth_scale_m": depth_scale_m,
                "rgb_device": rgb_health["rgb_device"],
                "rgb_profile": rgb_health["rgb_profile"],
                "pair_sequence_matched": pair_matched,
                "navigation_ready": navigation_ready,
                "observation_ready": bool(navigation_ready and depth.get("raw_depth_available") is True),
            }
        )
        return depth

    def matched_observation(
        self,
    ) -> Tuple[FrameSnapshot, DepthSnapshot, Dict[str, object]]:  # noqa: UP006
        """Read one immutable, fresh, navigation-ready RealSense pair."""

        source = self.source
        depth_store = source.depth_store
        if depth_store is None:
            raise CameraStreamError("aligned depth capture is disabled")

        # Reading both object slots again after metadata construction detects a
        # publish crossing the read without coupling HTTP readers to a lock.
        for _attempt in range(3):
            frame = source.frame_store.latest()
            depth = depth_store.latest()
            metadata = self.depth_payload()
            if frame is not source.frame_store.latest() or depth is not depth_store.latest():
                continue
            if frame is None or depth is None:
                raise CameraStreamError("no matched RGB-D observation is available")
            if metadata.get("navigation_ready") is not True:
                raise CameraStreamError("matched RGB-D observation is not navigation-ready")
            if depth.depth_png is None:
                raise CameraStreamError("raw aligned depth PNG is unavailable")
            if (
                isinstance(frame.sequence, bool)
                or not isinstance(frame.sequence, int)
                or frame.sequence <= 0
                or frame.sequence != depth.sequence
            ):
                raise CameraStreamError("RGB and depth sequences do not match")
            if metadata.get("sequence") != frame.sequence:
                raise CameraStreamError("RGB-D metadata sequence does not match the snapshot")
            capture_times = (
                frame.captured_at_unix,
                frame.captured_at_monotonic,
                depth.captured_at_unix,
                depth.captured_at_monotonic,
            )
            if not all(math.isfinite(value) for value in capture_times):
                raise CameraStreamError("RGB-D capture timestamps are invalid")
            if (
                frame.captured_at_unix != depth.captured_at_unix
                or frame.captured_at_monotonic != depth.captured_at_monotonic
            ):
                raise CameraStreamError("RGB and depth capture timestamps do not match")
            if (
                frame.source_frame_number is None
                or isinstance(frame.source_frame_number, bool)
                or not isinstance(frame.source_frame_number, int)
                or frame.source_frame_number < 0
                or frame.source_frame_number != depth.source_frame_number
            ):
                raise CameraStreamError("RGB and depth source frame numbers do not match")
            if (
                frame.source_timestamp_ms is None
                or not math.isfinite(frame.source_timestamp_ms)
                or frame.source_timestamp_ms != depth.source_timestamp_ms
            ):
                raise CameraStreamError("RGB and depth source timestamps do not match")
            if source.frame_store.frame_age(frame) > source.max_frame_age:
                raise CameraStreamError("matched RGB frame is stale")
            if depth_store.snapshot_age(depth) > source.max_depth_age:
                raise CameraStreamError("matched depth frame is stale")
            if not (frame.jpeg.startswith(b"\xff\xd8") and frame.jpeg.endswith(b"\xff\xd9")):
                raise CameraStreamError("matched RGB payload is not a complete JPEG")
            if not depth.depth_png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CameraStreamError("matched depth payload is not a PNG")

            depth_scale_m = metadata.get("depth_scale_m")
            if (
                isinstance(depth_scale_m, bool)
                or not isinstance(depth_scale_m, (int, float))
                or not math.isfinite(float(depth_scale_m))
                or float(depth_scale_m) <= 0.0
            ):
                raise CameraStreamError("matched depth scale is invalid")

            rgb_profile = metadata.get("rgb_profile")
            depth_profile = metadata.get("depth_profile")
            if not isinstance(rgb_profile, dict) or not isinstance(depth_profile, dict):
                raise CameraStreamError("matched RGB-D profiles are unavailable")
            rgb_dimensions = (rgb_profile.get("width"), rgb_profile.get("height"))
            depth_dimensions = (
                depth_profile.get("width"),
                depth_profile.get("height"),
            )
            dimensions = (*rgb_dimensions, *depth_dimensions)
            if (
                any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions)
                or rgb_dimensions != depth_dimensions
                or depth_dimensions != (depth_store.width, depth_store.height)
            ):
                raise CameraStreamError("matched RGB-D dimensions do not agree")
            return frame, depth, metadata

        raise CameraStreamError("RGB-D observation changed while it was being read")

    def observation_payload(self) -> Dict[str, object]:  # noqa: UP006
        frame, depth, metadata = self.matched_observation()
        assert depth.depth_png is not None
        rgb_profile = metadata["rgb_profile"]
        depth_profile = metadata["depth_profile"]
        assert isinstance(rgb_profile, dict)
        assert isinstance(depth_profile, dict)
        sequence = frame.sequence
        return {
            "sequence": sequence,
            "captured_at_unix": frame.captured_at_unix,
            "source_frame_number": frame.source_frame_number,
            "source_timestamp_ms": frame.source_timestamp_ms,
            "pair_sequence_matched": True,
            "navigation_ready": True,
            "rgb": {
                "sequence": sequence,
                "media_type": "image/jpeg",
                "width": rgb_profile["width"],
                "height": rgb_profile["height"],
                "data_url": "data:image/jpeg;base64,{}".format(base64.b64encode(frame.jpeg).decode("ascii")),
            },
            "depth": {
                "sequence": sequence,
                "media_type": "image/png",
                "encoding": "uint16",
                "width": depth_profile["width"],
                "height": depth_profile["height"],
                "scale_m": metadata["depth_scale_m"],
                "registered_to_rgb": True,
                "center_distance_m": depth.center_distance_m,
                "minimum_distance_m": depth.minimum_distance_m,
                "valid_fraction": depth.valid_fraction,
                "data_url": "data:image/png;base64,{}".format(base64.b64encode(depth.depth_png).decode("ascii")),
            },
        }


__all__ = ["CameraObservationService"]
