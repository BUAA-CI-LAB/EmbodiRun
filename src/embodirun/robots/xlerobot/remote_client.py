"""Minimal HTTP client for the inspected XLeRobot ``RemoteRobot`` API.

This is a narrow copy of the sibling quest-teleop client's
``RemoteRobot`` boundary, kept here so
the LightNav deploy checkout has a concrete factory path without importing the
sibling repository.  The client owns only authenticated HTTP
transport and control ownership; model inference, freshness validation and
bounded velocity checks remain in the LightNav adapter.

Construction and ``connect`` are read-only.  ``arm`` is an explicit caller
action.  No demo robot, SDK import, arm side effect or motor command occurs at
import time or construction time.
"""

from __future__ import annotations

import base64
import binascii
import json
import secrets
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any


class RemoteRobotError(RuntimeError):
    """The remote robot service returned an unusable response."""


class RemoteRobot:
    """Synchronous authenticated HTTP client for the existing robot service.

    The constructor signature and endpoint names match the inspected sibling
    legacy teleoperation RemoteRobot.  ``scope="base"`` is the
    normal LightNav setting and keeps arm control separate from stowed arms.
    """

    mode = "remote"
    CONTROL_TIMEOUT_S = 15.0

    def __init__(self, url: str, token: str, *, timeout: float = 2.0, scope: str = "all"):
        if not isinstance(url, str) or not url.startswith(("http://", "https://")) or not url.strip():
            raise ValueError("robot URL is required and must use HTTP(S)")
        if not isinstance(token, str) or not token:
            raise ValueError("robot token is required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if scope not in ("all", "arms", "base"):
            raise ValueError("invalid control scope")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        self.scope = scope
        self.owner = secrets.token_urlsafe(24)
        self.metadata: dict[str, Any] = {"source": "unknown", "allow_motion": False}
        self.armed = False
        self.closed = False
        # Match the sibling client's no-proxy local robot path.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(
        self,
        endpoint: str,
        data: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(endpoint, str) or not endpoint or "/" in endpoint:
            raise ValueError("endpoint must be a single non-empty path component")
        request_timeout = self.timeout if timeout is None else float(timeout)
        body = None if data is None else json.dumps(dict(data), allow_nan=False).encode()
        request = urllib.request.Request(
            self.url + "/robot/" + endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-Teleop-Owner": self.owner,
                "X-Teleop-Scope": self.scope,
            },
        )
        try:
            with self.opener.open(request, timeout=request_timeout) as response:
                result = json.load(response)
        except TimeoutError as exc:
            raise TimeoutError(f"AGX {endpoint} timed out after {request_timeout:g}s") from exc
        except urllib.error.HTTPError as exc:
            try:
                message = json.load(exc).get("error", str(exc))
            except (ValueError, AttributeError, TypeError):
                message = str(exc)
            raise RemoteRobotError(f"AGX: {message}") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TimeoutError(f"AGX {endpoint} timed out after {request_timeout:g}s") from exc
            raise ConnectionError(f"AGX {endpoint} failed: {reason}") from exc
        if not isinstance(result, Mapping):
            raise RemoteRobotError(f"AGX {endpoint} returned a non-object response")
        if result.get("error"):
            raise RemoteRobotError(str(result["error"]))
        return dict(result)

    def connect(self) -> None:
        """Read service status and establish metadata without enabling motion."""

        if self.closed:
            raise RemoteRobotError("robot client is closed")
        info = self._request("status")
        metadata = info.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise RemoteRobotError("AGX status metadata must be an object")
        self.metadata = dict(metadata)
        if self.scope != "all" and self.scope not in self.metadata.get("control_scopes", []):
            raise RemoteRobotError("AGX service does not support independent control scopes; update it first")
        # Do not inherit another client's armed state or control ownership.
        self.armed = False

    def read(self) -> tuple[dict[str, Any], dict[str, bytes]]:
        """Read one AGX observation and the camera bytes it explicitly marked fresh."""

        if self.closed:
            raise RemoteRobotError("robot client is closed")
        payload = self._request("observe")
        observation = payload.get("observation")
        images_payload = payload.get("images")
        if not isinstance(observation, Mapping) or not isinstance(images_payload, Mapping):
            raise RemoteRobotError("AGX observe response must contain observation and images objects")
        observation = dict(observation)
        # AGX and the deploy host need not share a wall clock.  Preserve the
        # source domains and keep local receive time in the adapter layer.
        observation["timestamp_domains"] = {
            "source": "robot",
            "state": "robot",
            "camera": "robot",
            "sent": "robot",
            "received": "gateway",
        }
        metadata = observation.get("metadata", self.metadata)
        if isinstance(metadata, Mapping):
            self.metadata = dict(metadata)
        try:
            images = {str(name): base64.b64decode(value, validate=True) for name, value in images_payload.items()}
        except (TypeError, ValueError, binascii.Error) as exc:
            raise RemoteRobotError("AGX observe image payload is not valid base64") from exc
        if self.armed and (
            observation.get("armed", True) is not True
            or (self.scope != "all" and observation.get("control_owned") is not True)
        ):
            self.armed = False
        return observation, images

    def arm(self) -> dict[str, Any]:
        """Explicitly request control ownership; never called by construction."""

        if self.closed:
            raise RemoteRobotError("robot client is closed")
        self.owner = secrets.token_urlsafe(24)
        result = self._request("arm", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))
        self.armed = result.get("armed") is True
        if not self.armed:
            errors = result.get("errors", [])
            detail = "; ".join(str(error) for error in errors) if isinstance(errors, list) else str(errors)
            raise RemoteRobotError("AGX did not confirm control enable" + (f": {detail}" if detail else ""))
        return result

    def command(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Submit an already bounded named action through the AGX boundary."""

        if self.closed:
            raise RemoteRobotError("robot client is closed")
        if not isinstance(action, Mapping):
            raise TypeError("action must be a mapping")
        return self._request("command", {"action": dict(action)})

    def stop(self) -> dict[str, Any]:
        """Revoke this owner's control and return the concrete stop report."""

        self.armed = False
        return self._request("stop", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))

    def stop_all(self) -> dict[str, Any]:
        """Use the service-wide stop route for explicit external recovery only."""

        self.armed = False
        return self._request("stop_all", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.armed:
                self.stop()
        finally:
            self.closed = True


def build_remote_robot_from_env(
    url: str,
    *,
    token_env: str = "XLEROBOT_TOKEN",
    timeout: float = 2.0,
    scope: str = "base",
    authorize_motion: bool = False,
) -> RemoteRobot:
    """Build, connect, and optionally arm a real client from an environment token.

    ``authorize_motion=False`` is the safe default: the caller can verify the
    connection while the deploy app refuses to issue commands until explicit
    authorization is requested.  The function is intended to be called only
    after model startup by the LightNav app's injected factory route.
    """

    import os

    token = os.environ.get(token_env)
    if not token:
        raise RemoteRobotError(f"robot token environment variable is empty: {token_env}")
    robot = RemoteRobot(url, token, timeout=timeout, scope=scope)
    try:
        robot.connect()
        if authorize_motion:
            result = robot.arm()
            if result.get("armed") is not True:
                raise RemoteRobotError("AGX did not confirm explicit motion authorization")
        return robot
    except BaseException:
        robot.close()
        raise


__all__ = ["RemoteRobot", "RemoteRobotError", "build_remote_robot_from_env"]
