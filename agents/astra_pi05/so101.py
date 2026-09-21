"""Public-media review packet bridge for a configured SO101 integration.

The bridge connects the generic cooperative loop to a named robot-specific
geometry/calibration integration.  It obtains images through
``ControlClient.media`` and requires callers to provide state extraction and
trajectory preview functions.  It never captures a camera or invents a
kinematic mapping.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from embodirun.client import ControlClient, Observation
from embodirun.robots.lerobot.bi_so101 import (
    BI_SO101_ACTION_SPACE,
    BI_SO101_POSITION_FEATURES,
)

from .cooperative import Proposal


class SO101ReviewPacketError(ValueError):
    """A public observation cannot be turned into an Astra review packet."""


def _finite_state(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 12:
        raise SO101ReviewPacketError("SO101 state extractor must return 12 values")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise SO101ReviewPacketError("SO101 state extractor returned a non-finite value")
    return result


class BiSO101ActionEncoder:
    """Encode a named 12-value row for the public dual SO101 adapter.

    The generic cooperative loop cannot infer whether a flat row belongs to a
    flat binding or to the structured ``BiSO101Adapter`` contract.  Callers
    must therefore opt into this encoder when their Deploy binding declares
    ``lerobot.bi_so101.position.v1``.  Input rows are degrees plus normalized
    grippers; the adapter remains responsible for calibrated limits and step
    checks.
    """

    def __init__(
        self,
        feature_names: Sequence[str] = BI_SO101_POSITION_FEATURES,
        *,
        action_space: str = BI_SO101_ACTION_SPACE,
    ) -> None:
        names = tuple(feature_names)
        if names != tuple(BI_SO101_POSITION_FEATURES):
            raise ValueError("BiSO101ActionEncoder requires the declared BI_SO101_POSITION_FEATURES order")
        if not isinstance(action_space, str) or not action_space.strip():
            raise ValueError("action_space must be a non-empty string")
        self.feature_names = names
        self.action_space = action_space
        self.proposal_metadata = {
            "action_semantics": "biso101_so101_v1",
            "action_layout": "left6_right6",
            "action_encoding": "absolute",
            "joint_position_unit": "degrees",
            "action_dim": 12,
            "horizon": 50,
            "feature_names": list(names),
        }

    def __call__(
        self,
        row: Sequence[float],
        *,
        timestamp_s: float,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if len(row) != len(self.feature_names):
            raise ValueError("BiSO101ActionEncoder rows must contain 12 values")
        if metadata.get("joint_position_unit") != "degrees":
            raise ValueError("BiSO101ActionEncoder accepts only proposals explicitly declared in degrees")
        values = [float(item) for item in row]
        if any(not math.isfinite(item) for item in values):
            raise ValueError("BiSO101ActionEncoder rows must contain finite values")
        if any(not -180.0 <= item <= 180.0 for item in values[:5] + values[6:11]):
            raise ValueError("BiSO101ActionEncoder arm joints must be in degree bounds")
        if any(not 0.0 <= values[index] <= 100.0 for index in (5, 11)):
            raise ValueError("BiSO101ActionEncoder grippers must be in [0,100]")
        return {
            "timestamp_s": float(timestamp_s),
            "values": {
                "type": "joint_position",
                "left": {
                    "joint_positions_deg": values[:5],
                    "gripper_position": values[5],
                },
                "right": {
                    "joint_positions_deg": values[6:11],
                    "gripper_position": values[11],
                },
            },
            "metadata": {
                "action_space": self.action_space,
                "position_units": "degrees_and_normalized_gripper",
            },
        }

    def decode(self, values: Mapping[str, Any]) -> list[float]:
        """Flatten one public dual-arm action for the Astra review contract."""

        if not isinstance(values, Mapping) or set(values) != {
            "type",
            "left",
            "right",
        }:
            raise ValueError("BiSO101ActionEncoder expects the public type/left/right action shape")
        if values["type"] != "joint_position":
            raise ValueError("BiSO101ActionEncoder only accepts joint_position actions")
        row: list[float] = []
        for side in ("left", "right"):
            arm = values[side]
            if not isinstance(arm, Mapping) or set(arm) != {
                "joint_positions_deg",
                "gripper_position",
            }:
                raise ValueError(f"BiSO101ActionEncoder {side} target is invalid")
            joints = arm["joint_positions_deg"]
            if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or len(joints) != 5:
                raise ValueError(f"BiSO101ActionEncoder {side} target requires five joints")
            row.extend(float(item) for item in joints)
            row.append(float(arm["gripper_position"]))
        # Reuse the same bounds as the encoder so proposal normalization cannot
        # accept a shape that the subsequent public execute path would reject.
        self(
            row,
            timestamp_s=0.0,
            metadata={"joint_position_unit": "degrees"},
        )
        return row


def extract_bi_so101_state(observation: Observation) -> list[float]:
    """Extract the named state emitted by the public dual SO101 adapter."""

    value = observation.robot
    if not isinstance(value, Mapping):
        raise SO101ReviewPacketError("public SO101 observation.robot must contain named values")
    try:
        state = [value[name] for name in BI_SO101_POSITION_FEATURES]
    except KeyError as error:
        raise SO101ReviewPacketError("public SO101 observation.robot is missing a configured feature") from error
    return _finite_state(state)


def _media_bytes(response: Any, role: str) -> tuple[bytes, str | None]:
    """Decode detached bytes from a public ``ControlClient.media`` response."""

    media_type: str | None = None
    value = response
    if isinstance(value, Mapping):
        media = value.get("media")
        if media is not None:
            # ``ControlClient.media`` returns the application envelope, not a
            # frame payload: {"observation_id": ..., "media": [{...}]}.
            # Match the declared role so a server returning several frames can
            # never silently feed the wrong camera to the reviewer.
            if not isinstance(media, Sequence) or isinstance(media, (str, bytes)):
                raise SO101ReviewPacketError(f"public media {role} has an invalid media list")
            candidates = [item for item in media if isinstance(item, Mapping) and item.get("name") == role]
            if len(candidates) != 1:
                raise SO101ReviewPacketError(f"public media response does not contain exactly one {role!r} frame")
            value = candidates[0]
        if isinstance(value, Mapping):
            media_type = value.get("mime_type", value.get("media_type"))
            # The versioned public API uses data_base64 when include_data=true.
            # Keep the legacy data form only for detached test doubles and
            # older public frame views; never interpret the numeric ``bytes``
            # length from a media reference as frame data.
            if "data_base64" in value:
                nested = value["data_base64"]
            else:
                nested = value.get("data")
                if isinstance(nested, Mapping):
                    media_type = nested.get("mime_type", media_type)
                    nested = nested.get("data_base64", nested.get("data"))
            value = nested
    if isinstance(value, str):
        try:
            value = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as error:
            raise SO101ReviewPacketError(f"public media {role} is not valid base64") from error
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise SO101ReviewPacketError(f"public media {role} has no detached bytes")
    return bytes(value), str(media_type) if media_type is not None else None


def _suffix(media_type: str | None) -> str:
    if media_type and "jpeg" in media_type.lower():
        return ".jpg"
    if media_type and "png" in media_type.lower():
        return ".png"
    return ".bin"


class SO101ReviewPacketBuilder:
    """Build an Astra packet from one retained public Deploy observation.

    ``state_extractor`` and ``trajectory_preview`` are mandatory injection
    points.  The former maps the public robot state to the exact ordered
    12-value BiSO101 state; the latter must use the selected robot geometry and
    calibration.  Keeping both explicit prevents a generic six-dimensional
    approximation from reaching Astra or the executor.
    """

    _roles = ("front", "left_wrist", "right_wrist")

    def __init__(
        self,
        client: ControlClient,
        *,
        media_dir: str | Path,
        state_extractor: Callable[[Observation], Sequence[float]],
        trajectory_preview: Callable[[Sequence[Sequence[float]]], Mapping[str, Any]],
        max_prefix_steps: int = 15,
        media_fetcher: Callable[[str, str], Any] | None = None,
    ) -> None:
        if not isinstance(client, ControlClient):
            raise TypeError("client must be a ControlClient")
        if not callable(state_extractor) or not callable(trajectory_preview):
            raise TypeError("state_extractor and trajectory_preview must be callable")
        if (
            isinstance(max_prefix_steps, bool)
            or not isinstance(max_prefix_steps, int)
            or not 1 <= max_prefix_steps <= 15
        ):
            raise ValueError("max_prefix_steps must be an integer from 1 to 15")
        if media_fetcher is not None and not callable(media_fetcher):
            raise TypeError("media_fetcher must be callable")
        self.client = client
        self.media_dir = Path(media_dir).resolve()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.state_extractor = state_extractor
        self.trajectory_preview = trajectory_preview
        self.max_prefix_steps = max_prefix_steps
        self.media_fetcher = media_fetcher
        self._counter = 0

    def _fetch(self, observation_id: str, role: str) -> Any:
        if self.media_fetcher is not None:
            return self.media_fetcher(observation_id, role)
        return self.client.media(observation_id, frame=role, include_data=True)

    def __call__(
        self,
        observation: Observation,
        proposal: Proposal,
        _base_packet: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._counter += 1
        output = self.media_dir / f"round-{self._counter:03d}"
        output.mkdir()
        paths: dict[str, str] = {}
        for role in self._roles:
            data, media_type = _media_bytes(self._fetch(observation.observation_id, role), role)
            path = output / f"{role}{_suffix(media_type)}"
            path.write_bytes(data)
            paths[role] = str(path)
        state = _finite_state(self.state_extractor(observation))
        preview = self.trajectory_preview(proposal.actions)
        if not isinstance(preview, Mapping):
            raise SO101ReviewPacketError("trajectory_preview must be a mapping")
        packet_observation = dict(observation.payload)
        packet_observation.update(
            {
                "observation_id": observation.observation_id,
                "state": state,
                "images": paths,
            }
        )
        packet_proposal = proposal.to_payload()
        packet_proposal["trajectory_preview"] = dict(preview)
        return {
            "observation": packet_observation,
            "proposal": packet_proposal,
            "max_prefix_steps": self.max_prefix_steps,
        }


__all__ = [
    "BI_SO101_ACTION_SPACE",
    "BI_SO101_POSITION_FEATURES",
    "BiSO101ActionEncoder",
    "extract_bi_so101_state",
    "SO101ReviewPacketBuilder",
    "SO101ReviewPacketError",
]
