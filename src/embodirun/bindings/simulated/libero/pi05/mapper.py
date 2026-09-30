"""Map LIBERO observations and Pi0.5 chunks for the simulated Panda EEF task.

LIBERO renders a Franka Panda with the same end-effector interface the Franka
Panda binding targets, so this mapper owns the LIBERO-specific wire contract:
one eight-float ``observation.state``, the two named policy cameras encoded as
PNG, and the seven-value OSC delta-pose chunk the checkpoint emits.
"""

from __future__ import annotations

import io
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from embodirun.model_services import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
)
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
STATE_FIELD = "observation.state"
STATE_DIMENSION = 8
EEF_POSITION_DIMENSION = 3
GRIPPER_QPOS_DIMENSION = 2
ACTION_DIMENSION = 7
ACTION_FEATURE_NAMES = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")

# Wire camera names the inference server is configured for, paired with the
# frame names the LIBERO adapter (and a raw LIBERO render) can expose.
WIRE_IMAGE_FIELDS = (
    ("image", ("observation.images.image", "image", "agentview_image")),
    ("wrist_image", ("observation.images.wrist_image", "wrist_image", "robot0_eye_in_hand_image")),
)

# LIBERO hands the adapter an (x, y, z, w) quaternion under this raw field.
EEF_QUATERNION_FIELD = "robot0_eef_quat"
EEF_POSITION_FIELD = "robot0_eef_pos"
GRIPPER_QPOS_FIELD = "robot0_gripper_qpos"

# LIBERO's relative OSC actions are defined on [-1, 1]; clipping stays opt-in
# so the binding never changes model output without the operator asking.
LIBERO_ACTION_LIMIT = 1.0
_QUATERNION_SINGULARITY_EPSILON = 1e-10


class LiberoPi05MapperError(RuntimeError):
    """A model result or LIBERO observation violates the binding contract."""


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise LiberoPi05MapperError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LiberoPi05MapperError(f"{name} must be numeric") from None
    if not math.isfinite(number):
        raise LiberoPi05MapperError(f"{name} must be finite")
    return number


def _vector(value: object, *, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LiberoPi05MapperError(f"{name} must be a sequence")
    if len(value) != size:
        raise LiberoPi05MapperError(f"{name} must contain {size} values")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def quaternion_xyzw_to_rotation_vector(
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    """Convert one ``(x, y, z, w)`` quaternion to a 3-value rotation vector.

    LIBERO reports ``robot0_eef_quat`` as a unit quaternion in ``(x, y, z, w)``
    order and the Pi0.5 LIBERO checkpoint expects the proprioceptive state in the
    axis-angle convention the LIBERO data pipeline uses, so this deliberately
    reproduces :func:`LiberoAdapter._quaternion_xyzw_to_axis_angle`:

        ``angle = 2 * acos(clamp(w, -1, 1))`` and ``axis = xyz / sin(angle / 2)``.

    ``sin(angle / 2)`` is ``sqrt(1 - w ** 2)`` for a unit quaternion.  When that
    denominator is at or below ``1e-10`` the rotation is the identity: ``w`` is
    within rounding of ``+1`` (zero angle) or ``-1`` (a full ``2 * pi`` turn),
    and the vector part carries no reliable axis, so the limit ``[0, 0, 0]`` is
    returned instead of dividing by zero.  Near ``pi`` the denominator stays
    near one, so the axis remains well conditioned; the sign of the axis follows
    ``xyz``, which keeps the result identical to the adapter for ``w < 0``.
    """

    if isinstance(quaternion, (str, bytes)) or not isinstance(quaternion, Sequence):
        raise LiberoPi05MapperError("eef quaternion must be a sequence")
    if len(quaternion) != 4:
        raise LiberoPi05MapperError("eef quaternion must contain 4 values")
    x, y, z, w = (
        _number(quaternion[0], "eef quaternion[0]"),
        _number(quaternion[1], "eef quaternion[1]"),
        _number(quaternion[2], "eef quaternion[2]"),
        _number(quaternion[3], "eef quaternion[3]"),
    )
    w = max(-1.0, min(1.0, w))
    sin_half_angle = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half_angle <= _QUATERNION_SINGULARITY_EPSILON:
        return (0.0, 0.0, 0.0)
    scale = 2.0 * math.acos(w) / sin_half_angle
    return (x * scale, y * scale, z * scale)


def libero_state_from_components(
    eef_position: object,
    eef_quaternion: object,
    gripper_qpos: object,
) -> tuple[float, ...]:
    """Assemble LIBERO's eight-float state from its raw robot components."""

    position = _vector(eef_position, size=EEF_POSITION_DIMENSION, name=EEF_POSITION_FIELD)
    quaternion = _vector(eef_quaternion, size=4, name=EEF_QUATERNION_FIELD)
    gripper = _vector(gripper_qpos, size=GRIPPER_QPOS_DIMENSION, name=GRIPPER_QPOS_FIELD)
    rotation_vector = quaternion_xyzw_to_rotation_vector(quaternion)
    return (*position, *rotation_vector, *gripper)


def _raw_components(values: Mapping[str, Any]) -> tuple[object, object, object] | None:
    nested = values.get("robot_state")
    if isinstance(nested, Mapping):
        eef = nested.get("eef")
        gripper = nested.get("gripper")
        if isinstance(eef, Mapping) and isinstance(gripper, Mapping):
            return eef.get("pos"), eef.get("quat"), gripper.get("qpos")
    if all(field in values for field in (EEF_POSITION_FIELD, EEF_QUATERNION_FIELD, GRIPPER_QPOS_FIELD)):
        return (
            values[EEF_POSITION_FIELD],
            values[EEF_QUATERNION_FIELD],
            values[GRIPPER_QPOS_FIELD],
        )
    return None


def _state(values: Mapping[str, Any]) -> tuple[float, ...]:
    if STATE_FIELD in values:
        return _vector(values[STATE_FIELD], size=STATE_DIMENSION, name=STATE_FIELD)
    components = _raw_components(values)
    if components is None:
        raise LiberoPi05MapperError(
            f"LIBERO observation must provide {STATE_FIELD!r} with {STATE_DIMENSION} values "
            f"or the raw {EEF_POSITION_FIELD!r}/{EEF_QUATERNION_FIELD!r}/{GRIPPER_QPOS_FIELD!r} components"
        )
    return libero_state_from_components(*components)


def _png_bytes(frame: CameraFrame) -> bytes:
    """Decode one adapter frame and re-encode it as an RGB PNG payload."""

    try:
        from PIL import Image
    except ImportError as error:
        raise LiberoPi05MapperError(
            "LIBERO Pi0.5 image encoding requires Pillow; install the sim-libero extra"
        ) from error
    try:
        with Image.open(io.BytesIO(frame.data)) as decoded:
            rgb = decoded.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG")
    except (OSError, ValueError) as error:
        raise LiberoPi05MapperError(f"camera frame {frame.name!r} is not a decodable image") from error
    return buffer.getvalue()


def _selected_frames(frames: Sequence[CameraFrame]) -> tuple[tuple[str, CameraFrame], ...]:
    by_name: dict[str, CameraFrame] = {}
    for frame in frames:
        if not isinstance(frame, CameraFrame):
            raise LiberoPi05MapperError("LIBERO frames must contain CameraFrame values")
        if frame.name in by_name:
            raise LiberoPi05MapperError(f"duplicate camera frame {frame.name!r}")
        by_name[frame.name] = frame
    selected: list[tuple[str, CameraFrame]] = []
    missing: list[str] = []
    for wire_name, aliases in WIRE_IMAGE_FIELDS:
        matches = [by_name[alias] for alias in aliases if alias in by_name]
        if not matches:
            missing.append(wire_name)
        elif len(matches) > 1:
            exposed = ", ".join(frame.name for frame in matches)
            raise LiberoPi05MapperError(f"ambiguous camera frames for {wire_name!r}: {exposed}")
        else:
            selected.append((wire_name, matches[0]))
    if missing:
        raise LiberoPi05MapperError("LIBERO input is missing camera frames: " + ", ".join(missing))
    return tuple(selected)


class LiberoPi05Mapper:
    """Translate LIBERO observations and Pi0.5 chunks into EmbodiRun values.

    ``clip_action`` defaults to ``False`` and stays opt-in: LIBERO's relative OSC
    actions live on ``[-1, 1]``, but the binding never changes model output unless
    the operator asks for it.  When enabled, every mapped value is clamped to
    ``[-LIBERO_ACTION_LIMIT, LIBERO_ACTION_LIMIT]``.
    """

    policy_action_space = POLICY_ACTION_SPACE

    def __init__(self, *, clip_action: bool = False) -> None:
        if not isinstance(clip_action, bool):
            raise LiberoPi05MapperError("clip_action must be a boolean")
        self.clip_action = clip_action

    def map_observation(
        self,
        observation: RobotObservation,
        *,
        session_id: str,
        request_id: str,
        step_id: int,
        instruction: str,
        frames: Sequence[CameraFrame],
    ) -> PolicyObservation:
        if not isinstance(observation.values, Mapping):
            raise LiberoPi05MapperError("LIBERO observation values must be an object")
        state = _state(observation.values)
        images = tuple(
            ImagePayload(wire_name, "image/png", _png_bytes(frame)) for wire_name, frame in _selected_frames(frames)
        )
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state={STATE_FIELD: list(state)},
            images=images,
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != self.policy_action_space:
            raise LiberoPi05MapperError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {self.policy_action_space!r}"
            )
        if len(result.actions) != 1:
            raise LiberoPi05MapperError("LIBERO Pi0.5 binding expects exactly one policy action")
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise LiberoPi05MapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise LiberoPi05MapperError("action_chunk values.data must be a sequence")
        if not data:
            raise LiberoPi05MapperError("action_chunk values.data must not be empty")
        declared = raw.values.get("feature_names")
        if declared is not None:
            names = _feature_names(declared)
            if names != ACTION_FEATURE_NAMES:
                raise LiberoPi05MapperError(
                    f"action_chunk feature_names must be {ACTION_FEATURE_NAMES!r}, got {names!r}"
                )
        actions: list[RobotAction] = []
        for row_index, row in enumerate(data):
            values = _action_row(row, row_index)
            if self.clip_action:
                values = tuple(max(-LIBERO_ACTION_LIMIT, min(LIBERO_ACTION_LIMIT, value)) for value in values)
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values={"action": list(values)},
                    metadata={
                        "action_space": POLICY_ACTION_SPACE,
                        "action_feature_names": list(ACTION_FEATURE_NAMES),
                        "request_id": result.request_id,
                        "session_id": result.session_id,
                        "step_id": result.step_id,
                        "session_revision": result.session_revision,
                        "chunk_index": row_index,
                        "chunk_size": len(data),
                        "clipped": self.clip_action,
                    },
                )
            )
        return tuple(actions)


def _feature_names(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LiberoPi05MapperError("action_chunk feature_names must be a sequence")
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise LiberoPi05MapperError("action_chunk feature_names must be strings")
    return names


def _action_row(row: object, row_index: int) -> tuple[float, ...]:
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
        raise LiberoPi05MapperError(f"action_chunk row {row_index} must be a sequence")
    if len(row) != ACTION_DIMENSION:
        raise LiberoPi05MapperError(f"action_chunk row {row_index} must contain {ACTION_DIMENSION} values")
    return tuple(_number(item, f"action_chunk row {row_index}[{index}]") for index, item in enumerate(row))


__all__ = [
    "ACTION_DIMENSION",
    "ACTION_FEATURE_NAMES",
    "LIBERO_ACTION_LIMIT",
    "POLICY_ACTION_SPACE",
    "STATE_DIMENSION",
    "STATE_FIELD",
    "WIRE_IMAGE_FIELDS",
    "LiberoPi05Mapper",
    "LiberoPi05MapperError",
    "libero_state_from_components",
    "quaternion_xyzw_to_rotation_vector",
]
