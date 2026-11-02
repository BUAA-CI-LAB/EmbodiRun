"""RGB/RGB-D observation client for the Go2 camera service."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from embodirun.robots.mobile_base import (
    EncodedDepthFrame,
    EncodedRGBFrame,
    NavigationObservation,
)
from embodirun.utils import DataUrlError, HttpClientError, JsonHttpClient, decode_data_url

MAX_CAMERA_RESPONSE_BYTES = 48 * 1024 * 1024


class Go2CameraError(RuntimeError):
    pass


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Go2CameraError(f"{name} must be an object")
    return dict(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Go2CameraError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Go2CameraError(f"{name} must be finite")
    return result


def _sequence(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Go2CameraError(f"{name} must be a non-negative integer")
    return value


class Go2CameraClient:
    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 2.0, require_depth: bool = True):
        self.require_depth = require_depth
        self.http = JsonHttpClient(base_url, token=token, timeout_s=timeout_s)

    def _capture(
        self,
        *,
        episode_id: str,
        reset: bool,
        robot_state: Mapping[str, object],
    ) -> NavigationObservation:
        started = time.monotonic()
        try:
            path = "/observation.json" if self.require_depth else "/observation/rgb.json"
            payload = self.http.request_json("GET", path, maximum_bytes=MAX_CAMERA_RESPONSE_BYTES)
        except HttpClientError as error:
            raise Go2CameraError(str(error)) from error
        received = time.monotonic()
        timing = {}
        if "frame_age_s" in payload:
            age = _finite(payload["frame_age_s"], "frame_age_s")
            if age < 0:
                raise Go2CameraError("frame_age_s must be non-negative")
            # The server's monotonic publish age avoids cross-host wall-clock
            # synchronization. Half the round trip bounds network uncertainty;
            # camera exposure/encoding delay still requires hardware calibration.
            timing = {
                "captured_at_monotonic_s": (started + received) / 2 - age,
                "capture_uncertainty_s": (received - started) / 2,
            }
        if payload.get("navigation_ready" if self.require_depth else "observation_ready") is not True:
            raise Go2CameraError("camera is not navigation-ready")
        sequence = _sequence(payload.get("sequence"), "observation.sequence")
        captured_at = _finite(payload.get("captured_at_unix"), "captured_at_unix")
        rgb_value = _object(payload.get("rgb"), "rgb")
        if _sequence(rgb_value.get("sequence"), "rgb.sequence") != sequence:
            raise Go2CameraError("RGB sequence does not match observation")
        try:
            rgb_data = decode_data_url(rgb_value.get("data_url"), "image/jpeg", maximum_bytes=12 * 1024 * 1024)
        except DataUrlError as error:
            raise Go2CameraError(str(error)) from error
        width = _sequence(rgb_value.get("width"), "rgb.width")
        height = _sequence(rgb_value.get("height"), "rgb.height")
        frame = EncodedRGBFrame(sequence, captured_at, rgb_data, width=width, height=height)
        depth = None
        if self.require_depth:
            depth_value = _object(payload.get("depth"), "depth")
            if _sequence(depth_value.get("sequence"), "depth.sequence") != sequence:
                raise Go2CameraError("depth sequence does not match observation")
            try:
                data = decode_data_url(depth_value.get("data_url"), "image/png", maximum_bytes=16 * 1024 * 1024)
            except DataUrlError as error:
                raise Go2CameraError(str(error)) from error
            depth = EncodedDepthFrame(
                sequence=sequence,
                captured_at_s=captured_at,
                data=data,
                width=_sequence(depth_value.get("width", width), "depth.width"),
                height=_sequence(depth_value.get("height", height), "depth.height"),
                scale_m=_finite(depth_value.get("scale_m"), "depth.scale_m"),
                registered_to_rgb=depth_value.get("registered_to_rgb", True) is True,
                encoding=depth_value.get("encoding", "uint16"),
                metadata={
                    key: value
                    for key, value in depth_value.items()
                    if key not in {"data_url", "sequence", "captured_at_unix", "width", "height", "scale_m"}
                },
            )
        return NavigationObservation(
            episode_id=episode_id,
            sequence=sequence,
            reset=reset,
            rgb_frames=(frame,),
            depth=depth,
            robot_state=robot_state,
            metadata=timing,
        )

    def capture(
        self,
        *,
        episode_id: str,
        reset: bool,
        robot_state: Mapping[str, object] | None = None,
    ) -> NavigationObservation:
        return self._capture(
            episode_id=episode_id,
            reset=reset,
            robot_state={} if robot_state is None else robot_state,
        )


__all__ = ["MAX_CAMERA_RESPONSE_BYTES", "Go2CameraClient", "Go2CameraError"]
