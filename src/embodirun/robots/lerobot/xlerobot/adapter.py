"""Bounded HTTP proxy for the already running XLeRobot owner service.

This adapter deliberately has no serial or camera SDK fallback.  Remote source
timestamps are retained as foreign timestamps; the proxy never invents local
freshness or a physical stop result.
"""

from __future__ import annotations

import http.client
import json
import math
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from ...adapter import RobotAction, RobotAdapter, RobotObservation, RobotPreparationRefused
from .config import XLeRobotConfig
from .units import validate_action


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class XLeRobotAdapterError(RuntimeError):
    pass


class XLeRobotTransportError(XLeRobotAdapterError):
    """The request may have reached the owner, so motion is uncertain."""


class XLeRobotProtocolError(XLeRobotAdapterError):
    """The owner returned malformed JSON or a malformed response object."""


class XLeRobotAdapter(RobotAdapter):
    mode = "remote"

    def __init__(self, config: XLeRobotConfig):
        self.config = config
        self.robot_id = config.robot_id
        self.url = config.url.rstrip("/")
        self.owner = secrets.token_urlsafe(24)
        self.metadata: dict[str, Any] = {
            "source": "xlerobot.external_owner",
            "control_scopes": [config.scope],
            "allow_motion": False,
        }
        self.prepared = False
        # prepared means the current owner was explicitly armed.  A pending
        # cleanup keeps that owner for one scoped release attempt; uncertainty
        # blocks any implicit re-arm after a possibly delivered request.
        self.stop_unconfirmed = False
        self._last_release: dict[str, Any] | None = None
        self._cleanup_pending = False
        self._connected = False
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = urllib.request.Request(
            self.url + "/robot/" + endpoint,
            data=body,
            method=method,
            headers={
                "Authorization": "Bearer " + self.config.token,
                "Content-Type": "application/json",
                "X-Teleop-Owner": self.owner,
                "X-Teleop-Scope": self.config.scope,
            },
        )
        try:
            with self._opener.open(req, timeout=self.config.timeout_s) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.load(exc).get("error", str(exc))
            except (ValueError, AttributeError):
                detail = str(exc)
            raise XLeRobotAdapterError(f"XLeRobot {endpoint}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise XLeRobotTransportError(f"XLeRobot {endpoint} transport failed") from exc
        except (ValueError, TypeError) as exc:
            raise XLeRobotProtocolError(f"XLeRobot {endpoint} returned malformed JSON") from exc
        if not isinstance(result, dict):
            raise XLeRobotProtocolError(f"XLeRobot {endpoint} returned a non-object")
        if result.get("error"):
            raise XLeRobotAdapterError(str(result["error"]))
        return result

    def connect(self, *, prepare: bool = True) -> None:
        status = self._request("status")
        metadata = status.get("metadata")
        if not isinstance(metadata, Mapping) or self.config.scope not in metadata.get("control_scopes", ()):
            raise XLeRobotAdapterError("owner does not advertise the requested control scope")
        self.metadata = dict(metadata)
        self.metadata["source"] = "xlerobot.external_owner"
        self._connected = True
        self.prepared = False
        if prepare:
            self.prepare()

    def prepare(self) -> None:
        if not self._connected:
            raise XLeRobotAdapterError("XLeRobot proxy is not connected")
        if self.stop_unconfirmed or self._cleanup_pending:
            raise XLeRobotAdapterError("previous XLeRobot stop is unconfirmed")
        if self.prepared:
            return
        self.owner = secrets.token_urlsafe(24)
        self._last_release = None
        self._cleanup_pending = True
        try:
            result = self._request("arm", method="POST", payload={})
        except (XLeRobotTransportError, XLeRobotProtocolError):
            self.stop_unconfirmed = True
            self.prepared = False
            raise
        if result.get("armed") is not True:
            if result.get("armed") is False and result.get("status") == "refused" and result.get("writes") == []:
                self._cleanup_pending = False
                raise RobotPreparationRefused("owner refused preparation before any write", result)
            self.stop_unconfirmed = True
            raise XLeRobotProtocolError("owner did not confirm control preparation")
        self.prepared = True
        self._cleanup_pending = False

    def observe(self) -> RobotObservation:
        request_started_ns = time.monotonic_ns()
        result = self._request("observe")
        observation = result.get("observation")
        if not isinstance(observation, Mapping):
            raise XLeRobotAdapterError("owner observation is missing")
        state = observation.get("state")
        if not isinstance(state, Mapping):
            raise XLeRobotProtocolError("XLeRobot observation state is missing")
        values = _json_value(state)
        timestamp_ns = observation.get("state_timestamp_ns")
        if timestamp_ns is None:
            raise XLeRobotProtocolError("XLeRobot state_timestamp_ns is invalid")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise XLeRobotProtocolError("XLeRobot state_timestamp_ns is invalid")
        source_timestamp_ns = observation.get("source_timestamp_ns")
        if source_timestamp_ns is not None and (
            isinstance(source_timestamp_ns, bool) or not isinstance(source_timestamp_ns, int) or source_timestamp_ns < 0
        ):
            raise XLeRobotProtocolError("XLeRobot source_timestamp_ns is invalid")
        timestamp_s = float(timestamp_ns) / 1e9 if timestamp_ns is not None else 0.0
        if self.prepared and observation.get("control_owned") is not True:
            self.prepared = False
            self._cleanup_pending = True
        # The owner reports elapsed state age, measured on its own monotonic
        # clock. Subtracting it from the local request start conservatively
        # includes the entire HTTP round trip; reception is never capture time.
        age = observation.get("state_age_ns")
        local_capture = None
        if isinstance(age, int) and not isinstance(age, bool) and 0 <= age <= request_started_ns:
            local_capture = request_started_ns - age
        return RobotObservation(
            timestamp_s=timestamp_s,
            values=values,
            metadata={
                **_json_value(self.metadata),
                "clock_domain": "host_monotonic_ns" if local_capture is not None else "remote_robot_wall",
                "captured_timestamp_ns": local_capture if local_capture is not None else timestamp_ns,
                "remote_state_timestamp_ns": timestamp_ns,
                "owner_state_age_ns": age,
                "safety": _json_value(observation.get("safety")),
                "navigation": _json_value(observation.get("navigation")),
                "task_evidence": _json_value(observation.get("task_evidence")),
                "source_timestamp_ns": source_timestamp_ns,
                "state_timestamp_ns": timestamp_ns,
                "camera_timestamps_ns": _json_value(observation.get("camera_timestamps_ns")),
                "remote_control_owned": _json_value(observation.get("control_owned")),
                "remote_armed": _json_value(observation.get("armed")),
                "remote_control_state": _json_value(observation.get("control_state")),
                "remote_errors": _json_value(observation.get("errors")),
                "raw": _json_value(observation.get("raw")),
                "raw_fields": _json_value(observation.get("raw_fields")),
                "wheel_present_blocks": _json_value(observation.get("wheel_present_blocks")),
                "state_cached": _json_value(observation.get("state_cached")),
            },
        )

    def execute(self, action: RobotAction) -> dict[str, Any]:
        if not self.prepared or self._cleanup_pending or self.stop_unconfirmed:
            raise XLeRobotAdapterError("XLeRobot control is not prepared")
        if not isinstance(action.values, Mapping):
            raise ValueError("XLeRobot action must be a mapping")
        values = dict(action.values)
        metadata = action.metadata if isinstance(action.metadata, Mapping) else {}
        # The canonical action space, scope, and per-field units live in one
        # shared contract so the recipe, the adapter, and the tests cannot
        # drift apart.  A missing unit or a base/arms mismatch is refused
        # before any owner command is sent.
        validate_action(values, metadata, scope=self.config.scope)
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{key} must be finite numeric")
        try:
            receipt = self._request("command", method="POST", payload={"action": values})
        except (XLeRobotTransportError, XLeRobotProtocolError):
            self.prepared = False
            self.stop_unconfirmed = True
            self._cleanup_pending = True
            raise
        if receipt.get("command_accepted") is not True or receipt.get("errors"):
            error = XLeRobotAdapterError("owner did not accept command")
            error.receipt = receipt  # type: ignore[attr-defined]
            self.prepared = False
            self._cleanup_pending = True
            raise error
        return receipt

    def stop(self) -> dict[str, Any]:
        if not self.prepared and not self._cleanup_pending:
            return self._last_release or {
                "released": False,
                "control_owned": False,
                "stop_confirmed": None,
                "physical_outcome": "unknown",
            }
        self._cleanup_pending = True
        try:
            result = self._request("release", method="POST", payload={})
        except XLeRobotAdapterError:
            self.prepared = False
            self.stop_unconfirmed = True
            raise
        if result.get("released") is not True or result.get("stop_confirmed") is not True:
            self.prepared = False
            self.stop_unconfirmed = True
            raise XLeRobotAdapterError("owner release did not confirm a stop")
        self.prepared = False
        self.stop_unconfirmed = False
        self._cleanup_pending = False
        self._last_release = dict(result)
        return result

    def close(self) -> None:
        # Passive close never arms or releases another owner's lease.
        if self.prepared or self._cleanup_pending:
            self.stop()
        self._connected = False
