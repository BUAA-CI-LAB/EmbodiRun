"""HTTP authentication, routing, and response serialization for camera data."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .types import CameraStreamError

if TYPE_CHECKING:
    from .capture import CameraCapture


class CameraRoutes:
    """Camera routes mixed into the unified HTTP handler; owns no server/auth."""

    @property
    def camera_server(self) -> CameraCapture:
        return self.server  # type: ignore[return-value]

    def serve_camera_get(self) -> None:
        path = urlsplit(self.path).path
        routes = {
            "/health": self._get_health,
            "/frame.jpg": self._get_frame,
            "/depth.json": self._get_depth_metadata,
            "/depth.png": self._get_depth_image,
            "/observation.json": self._get_observation,
            "/observation/rgb.json": self._get_rgb_observation,
            "/health/camera": self._get_health,
        }
        route = routes.get(path)
        if route is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        route()

    def _get_health(self) -> None:
        server = self.camera_server
        health = server.frame_store.health(server.max_frame_age)
        depth = server.observations.depth_payload()
        health.update(
            {
                "depth": depth,
                "calibration_confirmed": depth["calibration_confirmed"],
                "rgb_depth_aligned": depth["rgb_depth_aligned"],
                "rgb_depth_alignment_claimed": depth["rgb_depth_alignment_claimed"],
                "registration_verified": depth["registration_verified"],
                "backend": depth["backend"],
                "depth_scale_m": depth["depth_scale_m"],
                "depth_device": depth["depth_device"],
                "depth_profile": depth["depth_profile"],
                "navigation_ready": depth["navigation_ready"],
                "observation_ready": depth["observation_ready"],
            }
        )
        health["rgb_observation_ready"] = health["status"] == "ok"
        status = HTTPStatus.OK if health["status"] == "ok" else HTTPStatus.SERVICE_UNAVAILABLE
        self._send_json(status, health)

    def _get_frame(self) -> None:
        server = self.camera_server
        frame = server.frame_store.latest()
        if frame is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "no camera frame is available yet"},
            )
            return
        age = server.frame_store.frame_age(frame)
        if age > server.max_frame_age:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": "latest camera frame is stale",
                    "frame_age_s": round(age, 3),
                },
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame.jpeg)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Frame-Sequence", str(frame.sequence))
        self.send_header("X-Frame-Age-Ms", str(int(age * 1000)))
        self.send_header("X-Captured-At-Unix", f"{frame.captured_at_unix:.6f}")
        if frame.source_frame_number is not None:
            self.send_header("X-Source-Frame-Number", str(frame.source_frame_number))
        if frame.source_timestamp_ms is not None:
            self.send_header("X-Source-Timestamp-Ms", f"{frame.source_timestamp_ms:.3f}")
        self.end_headers()
        self.wfile.write(frame.jpeg)

    def _get_depth_metadata(self) -> None:
        depth = self.camera_server.observations.depth_payload()
        status = HTTPStatus.OK if depth["navigation_ready"] else HTTPStatus.SERVICE_UNAVAILABLE
        self._send_json(status, depth)

    def _get_depth_image(self) -> None:
        server = self.camera_server
        try:
            frame, depth, metadata = server.observations.matched_observation()
        except CameraStreamError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": str(error), "available": False},
            )
            return
        assert depth.depth_png is not None
        assert server.depth_store is not None
        age = server.depth_store.snapshot_age(depth)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(depth.depth_png)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Frame-Sequence", str(frame.sequence))
        self.send_header("X-Frame-Age-Ms", str(int(age * 1000)))
        self.send_header("X-Captured-At-Unix", f"{frame.captured_at_unix:.6f}")
        self.send_header("X-Depth-Encoding", "uint16")
        self.send_header("X-Depth-Scale-M", f"{metadata['depth_scale_m']:.12g}")
        self.send_header("X-Source-Frame-Number", str(frame.source_frame_number))
        self.send_header("X-Source-Timestamp-Ms", f"{frame.source_timestamp_ms:.3f}")
        self.end_headers()
        self.wfile.write(depth.depth_png)

    def _get_rgb_observation(self) -> None:
        try:
            observation = self.camera_server.observations.rgb_observation_payload()
        except CameraStreamError as error:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error), "available": False})
            return
        self._send_json(HTTPStatus.OK, observation)

    def _get_observation(self) -> None:
        try:
            observation = self.camera_server.observations.observation_payload()
        except CameraStreamError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": str(error), "available": False},
            )
            return
        self._send_json(HTTPStatus.OK, observation)


__all__ = ["CameraRoutes"]
