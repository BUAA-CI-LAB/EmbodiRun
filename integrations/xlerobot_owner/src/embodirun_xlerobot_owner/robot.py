"""Synthetic robot and authenticated Mac-to-AGX HTTP client."""

from __future__ import annotations

import base64
import io
import json
import secrets
import time
import urllib.error
import urllib.request
from typing import Any

from .control import CAMERAS, JOINT_NAMES, finite


class DemoRobot:
    mode = "demo"

    def __init__(self):
        self.armed = False
        self._active_scopes: set[str] = set()
        self.state = {name: 50.0 if name.endswith("gripper.pos") else 0.0 for name in JOINT_NAMES}
        self.state.update({"x.vel": 0.0, "theta.vel": 0.0})
        self.metadata = {
            "source": "synthetic",
            "joint_names": list(JOINT_NAMES),
            "joint_unit": "degrees",
            "gripper_unit": "range_0_100",
            "camera_roles_confirmed": True,
            "cameras": list(CAMERAS),
            "control_scopes": ["arms", "base"],
        }

    def connect(self) -> None:
        pass

    def read(self) -> tuple[dict, dict[str, bytes]]:
        from PIL import Image, ImageDraw

        now = time.time_ns()
        images = {}
        for camera, color in zip(CAMERAS, ("#173f46", "#253450", "#473150")):
            frame = Image.new("RGB", (640, 480), color)
            draw = ImageDraw.Draw(frame)
            draw.text((30, 25), f"SYNTHETIC DEMO / {camera}", fill="white")
            draw.text((30, 55), f"{now / 1e9:.3f} | armed={self.armed}", fill="white")
            for i, name in enumerate(JOINT_NAMES):
                value = self.state[name]
                y = 95 + i * 27
                draw.text((20, y), f"{name}: {value:.1f}", fill="white")
                x = 410 + int(value * 0.6)
                draw.ellipse((x - 3, y, x + 3, y + 6), fill="#55efc4")
            output = io.BytesIO()
            frame.save(output, "JPEG", quality=80)
            images[camera] = output.getvalue()
        return {
            "state": dict(self.state),
            "source_timestamp_ns": now,
            "state_timestamp_ns": now,
            "camera_timestamps_ns": dict.fromkeys(CAMERAS, now),
            "metadata": self.metadata,
            "armed": self.armed,
            "control_state": self.control_state(),
        }, images

    def control_state(self) -> dict:
        return {scope: scope in self._active_scopes for scope in ("arms", "base")}

    def arm(self, scope: str = "all") -> dict:
        if scope not in ("all", "arms", "base"):
            raise ValueError("invalid control scope")
        self._active_scopes.update(("arms", "base") if scope == "all" else (scope,))
        self.armed = True
        return {"armed": True, "source": "synthetic"}

    def command(self, action: dict) -> dict:
        if not self.armed:
            raise RuntimeError("demo robot not armed")
        if set(action) not in (set(JOINT_NAMES) | {"x.vel", "theta.vel"}, set(JOINT_NAMES), {"x.vel", "theta.vel"}):
            raise ValueError("action fields do not match dual SO101 and two-wheel base")
        needed = ({"arms"} if set(action) & set(JOINT_NAMES) else set()) | ({"base"} if "x.vel" in action else set())
        if not needed <= self._active_scopes:
            raise RuntimeError("action includes an inactive control scope")
        values = {name: finite(value, name) for name, value in action.items()}
        self.state.update(values)
        return {
            "applied_action": values,
            "sent_timestamp_ns": time.time_ns(),
            "physical_outcome": "synthetic",
            "source": "synthetic",
        }

    def stop(self, scope: str = "all") -> dict:
        if scope not in ("all", "arms", "base"):
            raise ValueError("invalid control scope")
        if scope != "arms":
            self.state.update({"x.vel": 0.0, "theta.vel": 0.0})
        self._active_scopes.difference_update(("arms", "base") if scope == "all" else (scope,))
        self.armed = bool(self._active_scopes)
        return {"stop_confirmed": True, "source": "synthetic"}

    def close(self) -> None:
        self.stop()


class RemoteRobot:
    mode = "remote"
    CONTROL_TIMEOUT_S = 15.0

    def __init__(self, url: str, token: str, *, timeout: float = 2.0, scope: str = "all"):
        if not url.startswith(("http://", "https://")) or not token:
            raise ValueError("robot URL and token required")
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        if scope not in ("all", "arms", "base"):
            raise ValueError("invalid control scope")
        self.scope = scope
        self.owner = secrets.token_urlsafe(24)
        self.metadata: dict[str, Any] = {"source": "unknown", "allow_motion": False}
        self.armed = False
        # Local robot traffic must not pass through HTTP_PROXY or an internet proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(
        self,
        endpoint: str,
        data: dict | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        request_timeout = self.timeout if timeout is None else timeout
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
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
            except (ValueError, AttributeError):
                message = str(exc)
            raise RuntimeError(f"AGX: {message}") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TimeoutError(f"AGX {endpoint} timed out after {request_timeout:g}s") from exc
            raise ConnectionError(f"AGX {endpoint} failed: {reason}") from exc
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def connect(self) -> None:
        info = self._request("status")
        self.metadata = info.get("metadata", {})
        if self.scope != "all" and self.scope not in self.metadata.get("control_scopes", []):
            raise RuntimeError("AGX service does not support independent control scopes; update it first")
        # Don't inherit another client's armed state or control ownership.
        self.armed = False

    def read(self) -> tuple[dict, dict[str, bytes]]:
        payload = self._request("observe")
        observation = payload["observation"]
        # AGX and Mac wall clocks need not be synchronized. Preserve both;
        # dataset alignment uses AGX state, camera and command timestamps only.
        observation["timestamp_domains"] = {
            "source": "robot",
            "state": "robot",
            "camera": "robot",
            "sent": "robot",
            "received": "gateway",
        }
        self.metadata = observation.get("metadata", self.metadata)
        if self.armed and (
            not observation.get("armed", True) or (self.scope != "all" and observation.get("control_owned") is not True)
        ):
            self.armed = False
        return observation, {name: base64.b64decode(value, validate=True) for name, value in payload["images"].items()}

    def arm(self) -> dict:
        self.last_arm_result = None
        self.owner = secrets.token_urlsafe(24)
        result = self._request("arm", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))
        self.last_arm_result = result
        self.armed = result.get("armed") is True
        if not self.armed:
            raise RuntimeError("AGX did not confirm control enable: " + "; ".join(result.get("errors", [])))
        return result

    def command(self, action: dict) -> dict:
        return self._request("command", {"action": action})

    def stop(self) -> dict:
        self.armed = False
        return self._request("stop", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))

    def stop_all(self) -> dict:
        self.armed = False
        return self._request("stop_all", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))

    def release(self) -> dict:
        """Stop only if this exact client still owns its scoped lease."""
        self.armed = False
        return self._request("release", {}, timeout=max(self.timeout, self.CONTROL_TIMEOUT_S))

    def close(self) -> None:
        if self.armed:
            self.stop()
