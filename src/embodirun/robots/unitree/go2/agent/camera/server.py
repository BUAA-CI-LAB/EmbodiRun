"""HTTP server assembly for the dog-side Go2 camera service."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Optional, Tuple  # noqa: UP035

from . import http_handler
from .depth_store import DepthStore
from .frame_store import FrameStore
from .observation import CameraObservationService
from .settings import validate_bind_token


class CameraHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],  # noqa: UP006
        store: FrameStore,
        max_frame_age: float,
        depth_store: Optional[DepthStore] = None,  # noqa: UP045
        max_depth_age: float = 0.5,
        api_token: Optional[str] = None,  # noqa: UP045
        calibration_confirmed: bool = False,
        rgb_depth_alignment_claimed: bool = False,
        backend: str = "v4l2",
        registration_verified: bool = False,
        depth_scale_m: Optional[float] = None,  # noqa: UP045
        depth_device: Optional[str] = None,  # noqa: UP045
        access_log: bool = False,
    ) -> None:
        validate_bind_token(str(server_address[0]), api_token)
        self.frame_store = store
        self.max_frame_age = max_frame_age
        self.depth_store = depth_store
        self.max_depth_age = max_depth_age
        self.api_token = api_token
        self.calibration_confirmed = calibration_confirmed
        self.rgb_depth_alignment_claimed = rgb_depth_alignment_claimed
        self.backend = backend
        self.registration_verified = registration_verified
        self.depth_scale_m = depth_scale_m
        self.depth_device = depth_device
        self.access_log = access_log
        self.observations = CameraObservationService(self)
        super().__init__(server_address, http_handler.CameraRequestHandler)


__all__ = ["CameraHTTPServer"]
