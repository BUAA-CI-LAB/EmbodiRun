"""LightNav waypoint binding over the existing model-neutral inference client."""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from embodirun.services.inference import (
    ImagePayload,
    InferenceClient,
    PolicyObservation,
    VvlaHttpClient,
)

ACTION_SPACE = "lightnav0.waypoints.v1"
IMAGE_FIELD = "observation.images.rgb"


@dataclass(frozen=True)
class WaypointChunk:
    """Decoded local waypoints and explicit stop metadata from inference."""

    actions: tuple[tuple[float, ...], ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class WaypointPrediction:
    """One successful response for the deployment control loop."""

    output: WaypointChunk


class LightNav0Client:
    """Serialize session operations without loading models in Deploy.

    The synchronous lock also covers HTTP work after coroutine cancellation, so
    reset/close cannot overtake an in-flight request running in a worker thread.
    """

    def __init__(self, client: InferenceClient, *, robot_id: str = "xlerobot") -> None:
        self._client = client
        self._robot_id = robot_id
        self._sessions: dict[str, tuple[str, int]] = {}
        self._lock = threading.RLock()

    async def start(self) -> None:
        """Check inference readiness before acquiring a physical control lease."""
        health = await asyncio.to_thread(self._client.health)
        if health.get("status") != "ok":
            raise RuntimeError("inference service is not ready")
        capabilities = await asyncio.to_thread(self._client.capabilities)
        adapter = capabilities.get("adapter", {})
        if not isinstance(adapter, Mapping) or adapter.get("action_space") != ACTION_SPACE:
            raise ValueError(f"inference must advertise {ACTION_SPACE!r}")

    async def reset_session(self, session_id: str) -> None:
        """Open or reset one remote episode while keeping local step order."""
        await asyncio.to_thread(self._reset_session, session_id)

    def _reset_session(self, session_id: str) -> None:
        with self._lock:
            previous = self._sessions.get(session_id)
            if previous is None:
                session = self._client.open_session(robot_id=self._robot_id, action_space=ACTION_SPACE)
            else:
                session = self._client.reset(previous[0], request_id=uuid.uuid4().hex)
            self._sessions[session_id] = (session.session_id, 0)

    async def predict(self, rgb: Any, *, instruction: str, session_id: str, timestamp_s: float) -> WaypointPrediction:
        """Send lossless RGB plus instruction/capture time; no map or goal state."""
        return await asyncio.to_thread(
            self._predict,
            rgb,
            instruction=instruction,
            session_id=session_id,
            timestamp_s=timestamp_s,
        )

    def _predict(self, rgb: Any, *, instruction: str, session_id: str, timestamp_s: float) -> WaypointPrediction:
        import numpy as np
        from PIL import Image

        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        pixels = np.asarray(rgb)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
            raise ValueError("RGB must be a non-empty HWC uint8 image")
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="PNG")
        with self._lock:
            if session_id not in self._sessions:
                raise RuntimeError("reset_session must be called before prediction")
            remote_id, step = self._sessions[session_id]
            request = PolicyObservation(
                session_id=remote_id,
                request_id=uuid.uuid4().hex,
                step_id=step,
                instruction=instruction,
                state={},
                images=(ImagePayload(IMAGE_FIELD, "image/png", encoded.getvalue()),),
                metadata={"timestamp_s": float(timestamp_s)},
            )
            result = self._client.step(request)
            if result.session_id != remote_id or result.request_id != request.request_id or result.step_id != step:
                raise ValueError("inference response does not match the current request")
            if result.action_space != ACTION_SPACE or len(result.actions) != 1 or result.actions[0].kind != "waypoints":
                raise ValueError("inference response is not a LightNav-0 waypoint chunk")
            metadata = dict(result.actions[0].values)
            rows = metadata.pop("data", None)
            if not isinstance(rows, (list, tuple)) or len(rows) != 10 or type(metadata.get("stop")) is not bool:
                raise ValueError("LightNav-0 response requires ten waypoints and a boolean stop")
            if any(
                not isinstance(row, (list, tuple))
                or len(row) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)
                for row in rows
            ):
                raise ValueError("LightNav-0 waypoints must contain finite numbers")
            self._sessions[session_id] = (remote_id, step + 1)
            return WaypointPrediction(WaypointChunk(tuple(tuple(float(v) for v in row) for row in rows), metadata))

    async def aclose(self) -> None:
        """Close every owned inference session, retaining failed close records."""
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        errors = []
        with self._lock:
            for alias, (remote_id, _) in list(self._sessions.items()):
                try:
                    self._client.close(remote_id)
                except Exception as error:  # noqa: BLE001 - attempt all closes before reporting failure
                    errors.append(error)
                else:
                    del self._sessions[alias]
        if errors:
            raise RuntimeError("failed to close one or more LightNav-0 sessions") from errors[0]


def create_client(args: argparse.Namespace) -> LightNav0Client:
    """Build an HTTP binding; tokens are read only from the selected environment."""
    token = None
    if args.inference_token_env:
        token = os.environ.get(args.inference_token_env)
        if not token:
            raise ValueError(f"missing inference token environment variable {args.inference_token_env!r}")
    return LightNav0Client(VvlaHttpClient(args.inference_url, token=token, timeout_s=args.inference_timeout_s))
