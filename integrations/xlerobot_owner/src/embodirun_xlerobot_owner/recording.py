"""Crash-tolerant Quest XLeRobot episode recording and LeRobot export.

The recorder deliberately keeps the evidence layers separate:

* ``observation`` is the measured state at time ``t``;
* ``action`` is the command that the gateway sent (or the controller applied);
* ``input_sample`` is the operator/headset input and is retained for debugging;
* ``feedback`` is the independently supplied controller/robot evidence.

The module has no optional imports at module import time.  In particular, the
LeRobot dependency and image decoder are loaded only by :func:`export_lerobot`.
The on-disk format is intentionally plain JSONL plus original JPEG files so a
partially written final record can be recovered without hiding earlier file
corruption.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_JOINT_NAMES: tuple[str, ...] = (
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
)

# Names used by callers that already describe the contract as "canonical".
DEFAULT_CANONICAL_JOINT_NAMES = DEFAULT_JOINT_NAMES
CANONICAL_JOINT_NAMES = DEFAULT_JOINT_NAMES

CAMERA_ROLES: tuple[str, ...] = ("front", "left_wrist", "right_wrist")
KNOWN_CAMERA_ROLES = CAMERA_ROLES

METADATA_FILE = "metadata.json"
FRAMES_FILE = "frames.jsonl"
RESULTS_FILE = "results.json"


class EpisodeError(RuntimeError):
    """Base class for recorder errors."""


class EpisodeCorruptionError(ValueError, EpisodeError):
    """A JSONL episode contains corruption before its final incomplete line."""


class LeRobotExportError(RuntimeError):
    """The requested export cannot be represented or its dependency is absent."""


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _to_jsonable(value: Any, *, name: str = "value") -> Any:
    """Convert common array/scalar values without permitting non-finite JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item, name=f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item, name=f"{name}[{index}]") for index, item in enumerate(value)]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _to_jsonable(tolist(), name=name)
    item = getattr(value, "item", None)
    if callable(item):
        return _to_jsonable(item(), name=name)
    raise TypeError(f"{name} is not JSON serializable")


def _json_bytes(value: Any, *, name: str = "value") -> bytes:
    return json.dumps(
        _to_jsonable(value, name=name),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_joint_names(joint_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(joint_names, (str, bytes, bytearray)) or not isinstance(joint_names, Sequence):
        raise TypeError("joint_names must be a sequence of strings")
    result = tuple(joint_names)
    if not result:
        raise ValueError("joint_names must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError("joint_names must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("joint_names must not contain duplicates")
    return result


def _safe_role(role: Any) -> str:
    if not isinstance(role, str) or not role.strip():
        raise ValueError("image role must be a non-empty string")
    role = role.strip()
    if role in {".", ".."} or "/" in role or "\\" in role or "\x00" in role:
        raise ValueError(f"unsafe image role {role!r}")
    return role


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write one small metadata/blob file and replace it only after fsync."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value, name=str(path)))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EpisodeError(f"episode is missing {path.name}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeCorruptionError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EpisodeCorruptionError(f"{path} must contain a JSON object")
    return raw


def _metadata_payload(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = metadata.get("metadata")
    return nested if isinstance(nested, Mapping) else metadata


def _first_metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    nested = _metadata_payload(metadata)
    for key in keys:
        if key in metadata:
            return metadata[key]
        if nested is not metadata and key in nested:
            return nested[key]
    return None


def _declared_max_ms(metadata: Mapping[str, Any], *keys: str) -> float | None:
    value = _first_metadata_value(metadata, *keys)
    if value is None:
        return None
    try:
        result = _finite_number(value, f"metadata.{keys[0]}")
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _declared_max_gap_ns(metadata: Mapping[str, Any]) -> int | None:
    value_ns = _first_metadata_value(metadata, "max_gap_ns", "max_resample_gap_ns")
    if value_ns is not None:
        try:
            return _nonnegative_int(value_ns, "metadata.max_gap_ns")
        except (TypeError, ValueError):
            return None
    value_ms = _declared_max_ms(
        metadata,
        "max_gap_ms",
        "max_resample_gap_ms",
        "max_action_age_ms",
        "max_observation_age_ms",
    )
    return None if value_ms is None else round(value_ms * 1_000_000.0)


def _timestamp_value(container: Any, key: str) -> Any:
    if isinstance(container, Mapping) and key in container:
        return container[key]
    return None


def _frame_timestamps(frame: Mapping[str, Any]) -> dict[str, Any]:
    observation = frame.get("observation")
    nested = observation.get("timestamps") if isinstance(observation, Mapping) else None
    timestamps = frame.get("timestamps")
    result: dict[str, Any] = {}
    for key in (
        "source_timestamp_ns",
        "state_timestamp_ns",
        "camera_timestamps_ns",
        "received_timestamp_ns",
    ):
        value = _timestamp_value(timestamps, key)
        if value is None:
            value = _timestamp_value(frame, key)
        if value is None:
            value = _timestamp_value(observation, key)
        if value is None:
            value = _timestamp_value(nested, key)
        result[key] = value
    return result


def _frame_timestamp_domains(frame: Mapping[str, Any]) -> dict[str, str]:
    """Return clock domains, using one shared fixture domain when unspecified."""

    result = dict.fromkeys(("source", "state", "camera", "received", "sent"), "__default__")
    observation = frame.get("observation")
    feedback = frame.get("feedback")
    for container in (frame, observation, feedback):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("timestamp_domains")
        if not isinstance(raw, Mapping):
            continue
        for key in result:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()
    return result


def _frame_observation_clock(frame: Mapping[str, Any]) -> tuple[int | None, str]:
    timestamps = _frame_timestamps(frame)
    domains = _frame_timestamp_domains(frame)
    for key in ("state_timestamp_ns", "source_timestamp_ns", "received_timestamp_ns"):
        value = timestamps[key]
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            domain_key = "received" if key == "received_timestamp_ns" else key.removesuffix("_timestamp_ns")
            return value, domains[domain_key]
    value = frame.get("recorded_timestamp_ns")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value, "__recorded__"
    return None, "__recorded__"


def _frame_observation_time(frame: Mapping[str, Any]) -> int | None:
    return _frame_observation_clock(frame)[0]


def _frame_action_time(frame: Mapping[str, Any]) -> int | None:
    action = frame.get("action")
    feedback = frame.get("feedback")
    observation_time, observation_domain = _frame_observation_clock(frame)
    domains = _frame_timestamp_domains(frame)
    for container in (feedback, action):
        if isinstance(container, Mapping):
            for key in (
                "applied_timestamp_ns",
                "executed_timestamp_ns",
                "sent_timestamp_ns",
                "action_timestamp_ns",
                "command_timestamp_ns",
            ):
                value = container.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    if domains["sent"] == observation_domain:
                        return value
                    return None
    return observation_time


def _camera_role_aliases(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Return source-key -> canonical role using only explicit mappings."""

    aliases: dict[str, str] = {}
    roles = _first_metadata_value(metadata, "camera_roles", "camera_map")
    if isinstance(roles, Mapping):
        for role in CAMERA_ROLES:
            source = roles.get(role)
            if isinstance(source, str) and source.strip():
                aliases[source] = role
            source = roles.get(f"observation.images.{role}")
            if isinstance(source, str) and source.strip():
                aliases[source] = role
    for role in CAMERA_ROLES:
        aliases[role] = role
        aliases[f"observation.images.{role}"] = role
    return aliases


def _canonical_image_paths(frame: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, str] | None:
    raw = frame.get("images")
    if not isinstance(raw, Mapping):
        return None
    aliases = _camera_role_aliases(metadata)
    result: dict[str, str] = {}
    for key, value in raw.items():
        role = aliases.get(key)
        if role is None:
            continue
        if isinstance(value, str) and role not in result:
            result[role] = value
    return result


def _candidate_mappings(value: Any, wrappers: Sequence[str]) -> Iterator[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    yielded: set[int] = set()

    def emit(candidate: Any) -> Iterator[Mapping[str, Any]]:
        if isinstance(candidate, Mapping) and id(candidate) not in yielded:
            yielded.add(id(candidate))
            yield candidate

    yield from emit(value)
    for wrapper in wrappers:
        candidate = value.get(wrapper)
        yield from emit(candidate)
        if isinstance(candidate, Mapping):
            for nested in ("values", "state", "joints", "positions", "action", "command"):
                yield from emit(candidate.get(nested))


def _vector_from_container(
    value: Any,
    names: Sequence[str],
    *,
    wrappers: Sequence[str],
    name: str,
) -> tuple[float, ...] | None:
    expected = tuple(names)
    if isinstance(value, Mapping):
        for container in _candidate_mappings(value, wrappers):
            if all(joint in container for joint in expected):
                return tuple(_finite_number(container[joint], f"{name}.{joint}") for joint in expected)

        sequences: list[Any] = []
        for key in wrappers:
            if key in value:
                sequences.append(value[key])
        sequences.append(value)
        raw_names = None
        for key in ("joint_names", "state_joint_names", "action_joint_names", "names", "order"):
            if key in value:
                raw_names = value[key]
                break
        if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes, bytearray)):
            raw_names_tuple = tuple(raw_names)
            if len(raw_names_tuple) == len(expected) and set(raw_names_tuple) == set(expected):
                for sequence in sequences:
                    if (
                        isinstance(sequence, Sequence)
                        and not isinstance(sequence, (str, bytes, bytearray))
                        and len(sequence) == len(raw_names_tuple)
                    ):
                        by_name = dict(zip(raw_names_tuple, sequence))
                        return tuple(_finite_number(by_name[joint], f"{name}.{joint}") for joint in expected)
    return None


def _state_vector(observation: Any) -> tuple[float, ...] | None:
    return _vector_from_container(
        observation,
        DEFAULT_JOINT_NAMES,
        wrappers=("state", "values", "joints", "arm_joints", "positions"),
        name="observation.state",
    )


def _action_vector(action: Any) -> tuple[float, ...] | None:
    return _vector_from_container(
        action,
        DEFAULT_JOINT_NAMES,
        wrappers=("applied", "actual", "sent", "command", "values", "action"),
        name="action",
    )


def _recursive_mapping_values(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _recursive_mapping_values(child)


def _base_velocity(value: Any) -> tuple[float, ...] | None:
    """Find an explicitly named mobile-base velocity without guessing a unit."""

    for container in _recursive_mapping_values(value):
        for key in ("base_velocity", "base_vel", "velocity"):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                fields = ("vx_m_s", "vy_m_s", "yaw_rate_rad_s")
                if all(field in candidate for field in fields):
                    return tuple(_finite_number(candidate[field], f"{key}.{field}") for field in fields)
            elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                if len(candidate) == 3:
                    return tuple(_finite_number(item, f"{key}[{index}]") for index, item in enumerate(candidate))
        if all(key in container for key in ("vx_m_s", "vy_m_s", "yaw_rate_rad_s")):
            return tuple(_finite_number(container[key], key) for key in ("vx_m_s", "vy_m_s", "yaw_rate_rad_s"))
        if "x.vel" in container or "theta.vel" in container:
            values = []
            for key in ("x.vel", "theta.vel"):
                if key in container:
                    values.append(_finite_number(container[key], key))
            if values:
                return tuple(values)
    return None


def _supports_mobile_base(metadata: Mapping[str, Any]) -> bool:
    for key in (
        "mobile_base_supported",
        "base_velocity_supported",
        "lerobot_base_velocity_supported",
        "base_velocity_export_supported",
    ):
        value = _first_metadata_value(metadata, key)
        if value is True:
            return True
    nested = _first_metadata_value(metadata, "mobile_base", "base")
    return isinstance(nested, Mapping) and nested.get("supported") is True


def _base_is_enabled(metadata: Mapping[str, Any]) -> bool:
    for key in ("enable_base", "base_enabled", "mobile_base_enabled", "has_base", "base_present"):
        if _first_metadata_value(metadata, key) is True:
            return True
    nested = _first_metadata_value(metadata, "mobile_base", "base")
    return isinstance(nested, Mapping) and nested.get("enabled") is True


def _unit_values(metadata: Mapping[str, Any], *values: Any) -> tuple[str | None, str | None]:
    joint_unit: Any = _first_metadata_value(
        metadata,
        "joint_position_unit",
        "joint_unit",
        "arm_joint_unit",
        "state_joint_unit",
        "action_joint_unit",
    )
    gripper_unit: Any = _first_metadata_value(
        metadata,
        "gripper_unit",
        "gripper_position_unit",
    )
    units = _first_metadata_value(metadata, "units", "state_units", "action_units")
    if isinstance(units, Mapping):
        if joint_unit is None:
            for key in ("joint_position", "joints", "arm", "arm_joints"):
                if units.get(key) is not None:
                    joint_unit = units[key]
                    break
        if gripper_unit is None:
            for key in ("gripper", "gripper_position"):
                if units.get(key) is not None:
                    gripper_unit = units[key]
                    break
    for value in values:
        if not isinstance(value, Mapping):
            continue
        if joint_unit is None:
            joint_unit = value.get("joint_position_unit") or value.get("joint_unit") or value.get("arm_joint_unit")
        if gripper_unit is None:
            gripper_unit = value.get("gripper_unit") or value.get("gripper_position_unit")
        child_units = value.get("units")
        if isinstance(child_units, Mapping):
            if joint_unit is None:
                joint_unit = child_units.get("joint_position") or child_units.get("joints")
            if gripper_unit is None:
                gripper_unit = child_units.get("gripper") or child_units.get("gripper_position")
    if not isinstance(joint_unit, str) or not joint_unit.strip():
        joint_unit = None
    if not isinstance(gripper_unit, str) or not gripper_unit.strip():
        gripper_unit = None
    return joint_unit, gripper_unit


def _source_value(metadata: Mapping[str, Any]) -> Any:
    return _first_metadata_value(metadata, "source", "recording_source", "provenance", "origin")


def _provenance_state(metadata: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    source = _source_value(metadata)
    source_text = str(source).strip().lower() if source is not None else ""
    collection_mode = _first_metadata_value(metadata, "collection_mode")
    collection_text = str(collection_mode).strip().lower() if collection_mode is not None else ""
    demo = source_text in {"synthetic", "demo", "simulation", "replay"} or collection_text in {
        "observation_only",
        "demo",
        "simulation",
        "replay",
    }
    physical = source_text == "physical"
    teleop = collection_text == "teleoperation"
    return physical, teleop, demo


def _camera_metadata_confirmed(metadata: Mapping[str, Any]) -> bool:
    if _first_metadata_value(metadata, "camera_roles_confirmed") is not True:
        return False
    cameras = _first_metadata_value(metadata, "cameras", "camera_names")
    if isinstance(cameras, Mapping):
        names = {str(key) for key in cameras} | {str(value) for value in cameras.values()}
    elif isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes, bytearray)):
        names = {str(value) for value in cameras}
    else:
        return False
    aliases = _camera_role_aliases(metadata)
    return all(role in {aliases.get(name) for name in names} for role in CAMERA_ROLES)


def _feedback_is_applied(feedback: Any) -> bool:
    if not isinstance(feedback, Mapping):
        return False
    # The command receipt identifies the action actually written to the
    # controller.  physical_outcome may remain unknown until a later measured
    # observation; unknown is not evidence that the write did not happen.
    applied = feedback.get("applied_action")
    return isinstance(applied, Mapping) and bool(applied)


def _jpeg_signature(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            if stream.read(2) != b"\xff\xd8":
                return False
            stream.seek(-2, os.SEEK_END)
            return stream.read(2) == b"\xff\xd9"
    except (OSError, ValueError):
        return False


def _safe_episode_file(episode: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("frame image path must be a non-empty relative string")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"frame image path must stay inside episode: {relative!r}")
    candidate = episode / path
    # ``relative`` has already been checked component by component; this
    # lexical check avoids resolving symlinks and therefore does no extra I/O.
    return candidate


@dataclass
class _FrameScan:
    frame_count: int = 0
    last_frame_index: int | None = None
    last_valid_offset: int = 0
    incomplete_final_line: bool = False
    final_valid_line_without_newline: bool = False
    frame_index_gap_count: int = 0
    explicit_dropped_count: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    previous_timestamp_ns: int | None = None
    timestamp_sample_count: int = 0
    irregular_interval_count: int = 0
    largest_interval_ns: int | None = None
    nonmonotonic_timestamps: bool = False
    callback_errors: int = 0


def _update_scan(scan: _FrameScan, frame: Mapping[str, Any], expected_fps: float | None) -> None:
    index = frame.get("frame_index", frame.get("index"))
    if isinstance(index, int) and not isinstance(index, bool):
        if scan.last_frame_index is not None and index != scan.last_frame_index + 1:
            scan.frame_index_gap_count += 1
        scan.last_frame_index = index
    dropped = frame.get("dropped_frames")
    if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped >= 0:
        scan.explicit_dropped_count += dropped
    feedback = frame.get("feedback")
    if isinstance(feedback, Mapping) and feedback.get("dropped") is True:
        scan.explicit_dropped_count += 1

    timestamp = _frame_observation_time(frame)
    if timestamp is None:
        return
    scan.timestamp_sample_count += 1
    if scan.first_timestamp_ns is None:
        scan.first_timestamp_ns = timestamp
    if scan.previous_timestamp_ns is not None:
        interval = timestamp - scan.previous_timestamp_ns
        if interval <= 0:
            scan.nonmonotonic_timestamps = True
        if scan.largest_interval_ns is None or interval > scan.largest_interval_ns:
            scan.largest_interval_ns = interval
        if expected_fps is not None:
            expected_ns = 1_000_000_000.0 / expected_fps
            if abs(interval - expected_ns) > 1.0:
                scan.irregular_interval_count += 1
    scan.previous_timestamp_ns = timestamp
    scan.last_timestamp_ns = timestamp


def _is_incomplete_json_line(text: str | None, error: Exception) -> bool:
    """Recognize an EOF-style JSON parse error, not arbitrary bad JSON."""

    if text is None or not isinstance(error, json.JSONDecodeError):
        return False
    return error.pos >= len(text) or error.msg.startswith("Unterminated")


def _scan_frames(
    episode: Path,
    *,
    expected_fps: float | None,
    on_frame: Callable[[dict[str, Any]], None] | None = None,
    raise_on_corruption: bool = True,
) -> _FrameScan:
    path = episode / FRAMES_FILE
    scan = _FrameScan()
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise EpisodeError(f"cannot open {path}: {exc}") from exc
    with stream:
        offset = 0
        line_number = 0
        while True:
            line = stream.readline()
            if not line:
                break
            line_number += 1
            offset += len(line)
            has_newline = line.endswith(b"\n")
            raw = line[:-1] if has_newline else line
            decoded: str | None = None
            try:
                decoded = raw.decode("utf-8")
                value = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not has_newline and stream.peek(1) == b"" and _is_incomplete_json_line(decoded, exc):
                    scan.incomplete_final_line = True
                    break
                message = f"corrupt JSONL at line {line_number}: {exc}"
                if raise_on_corruption:
                    raise EpisodeCorruptionError(message) from exc
                scan.callback_errors += 1
                break
            if not isinstance(value, dict):
                message = f"corrupt JSONL at line {line_number}: frame must be an object"
                if raise_on_corruption:
                    raise EpisodeCorruptionError(message)
                scan.callback_errors += 1
                break
            _update_scan(scan, value, expected_fps)
            scan.frame_count += 1
            scan.last_valid_offset = offset
            if not has_newline:
                scan.final_valid_line_without_newline = True
            if on_frame is not None:
                try:
                    on_frame(value)
                except (TypeError, ValueError, OSError) as exc:
                    if raise_on_corruption:
                        raise EpisodeCorruptionError(
                            f"invalid frame {scan.frame_count - 1} at line {line_number}: {exc}"
                        ) from exc
                    scan.callback_errors += 1
                    break
    return scan


def _load_frames(episode: Path) -> Iterator[dict[str, Any]]:
    """Yield complete frames, skipping only one incomplete final JSON line."""

    path = episode / FRAMES_FILE
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise EpisodeError(f"cannot open {path}: {exc}") from exc
    with stream:
        line_number = 0
        while True:
            line = stream.readline()
            if not line:
                return
            line_number += 1
            has_newline = line.endswith(b"\n")
            raw = line[:-1] if has_newline else line
            decoded: str | None = None
            try:
                decoded = raw.decode("utf-8")
                value = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not has_newline and stream.peek(1) == b"" and _is_incomplete_json_line(decoded, exc):
                    return
                raise EpisodeCorruptionError(f"corrupt JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EpisodeCorruptionError(f"corrupt JSONL at line {line_number}: frame must be an object")
            yield value


class EpisodeRecorder:
    """Bounded-memory recorder for one Quest XLeRobot episode at a time."""

    def __init__(
        self,
        root: Path,
        *,
        fps: float = 20.0,
        joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
    ) -> None:
        # Do not touch the filesystem here.  A caller can construct a recorder
        # while deciding whether recording is authorized.
        self.root = Path(root)
        self.fps = _positive_finite(fps, "fps")
        self.joint_names = _validate_joint_names(joint_names)
        self._path: Path | None = None
        self._active = False
        self._frame_count = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self, task: str, metadata: dict[str, Any]) -> Path:
        if self._active:
            raise RuntimeError("episode recorder is already active")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-blank string")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        clean_metadata = _to_jsonable(dict(metadata), name="metadata")
        started_ns = time.time_ns()
        self.root.mkdir(parents=True, exist_ok=True)

        episode: Path | None = None
        for attempt in range(1000):
            candidate = self.root / f"episode-{started_ns}-{attempt}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            episode = candidate
            break
        if episode is None:
            raise FileExistsError("could not allocate a unique episode directory")

        payload = {
            "format": "quest_xlerobot_episode",
            "format_version": 1,
            "episode_id": episode.name,
            "task": task.strip(),
            "fps": self.fps,
            "recording_fps": self.fps,
            "joint_names": list(self.joint_names),
            "started_timestamp_ns": started_ns,
            "status": "recording",
            "metadata": clean_metadata,
        }
        _write_json(episode / METADATA_FILE, payload)
        frames = episode / FRAMES_FILE
        with frames.open("ab") as stream:
            stream.flush()
            os.fsync(stream.fileno())

        self._path = episode
        self._active = True
        self._frame_count = 0
        return episode

    def append(
        self,
        *,
        observation: dict[str, Any],
        action: dict[str, Any] | None,
        images: dict[str, bytes],
        input_sample: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> None:
        if not self._active or self._path is None:
            raise RuntimeError("episode recorder is not active")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if not isinstance(images, Mapping):
            raise TypeError("images must be a mapping of role to JPEG bytes")

        # Preflight JSON serialization before touching image files.  This
        # keeps a bad application payload from creating an apparently complete
        # frame with only some of its images.
        clean_observation = _to_jsonable(dict(observation), name="observation")
        clean_action = None if action is None else _to_jsonable(dict(action), name="action")
        clean_input = None if input_sample is None else _to_jsonable(dict(input_sample), name="input_sample")
        clean_feedback = None if feedback is None else _to_jsonable(dict(feedback), name="feedback")

        frame_index = self._frame_count
        image_payloads: list[tuple[str, bytes, Path]] = []
        for raw_role, raw_bytes in images.items():
            role = _safe_role(raw_role)
            if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
                raise TypeError(f"images[{role!r}] must be bytes")
            payload = bytes(raw_bytes)
            if not payload:
                raise ValueError(f"images[{role!r}] must not be empty")
            relative = Path("images") / role / f"{frame_index:08d}.jpg"
            image_payloads.append((role, payload, self._path / relative))

        recorded_ns = time.time_ns()
        observation_timestamps = {
            key: _frame_timestamps({"observation": clean_observation})[key]
            for key in (
                "source_timestamp_ns",
                "state_timestamp_ns",
                "camera_timestamps_ns",
                "received_timestamp_ns",
            )
        }
        relative_images = {role: str(path.relative_to(self._path)) for role, _, path in image_payloads}
        frame = {
            "frame_index": frame_index,
            "recorded_timestamp_ns": recorded_ns,
            "timestamps": observation_timestamps,
            # These aliases make the source timestamps easy to inspect with a
            # line-oriented tool while ``timestamps`` remains the canonical
            # grouped representation.
            **observation_timestamps,
            "observation": clean_observation,
            "action": clean_action,
            "input_sample": clean_input,
            "feedback": clean_feedback,
            "images": relative_images,
        }
        line = _json_bytes(frame, name="frame") + b"\n"

        written_paths: list[Path] = []
        try:
            for _, payload, final_path in image_payloads:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_bytes(final_path, payload)
                written_paths.append(final_path)
            # The JSONL entry is the commit marker for this frame.  Images are
            # never referenced before every image has been fsynced/replaced.
            with (self._path / FRAMES_FILE).open("ab", buffering=0) as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            for path in written_paths:
                with contextlib.suppress(OSError):
                    path.unlink()
            raise

        self._frame_count += 1

    def finish(
        self,
        success: bool | None = None,
        reason: str = "operator",
        *,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        if not self._active or self._path is None:
            raise RuntimeError("episode recorder is not active")
        if success is not None and not isinstance(success, bool):
            raise TypeError("success must be a bool or None")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-blank string")

        # Re-scan rather than trusting the in-memory counter.  This accounts
        # for a final partial line caused by a disk/process interruption.
        metadata_path = self._path / METADATA_FILE
        metadata = _read_json_object(metadata_path)
        scan = _scan_frames(self._path, expected_fps=self.fps, raise_on_corruption=True)
        gap_analysis = _gap_analysis(scan, self.fps, metadata)
        interrupted = success is None and (
            interrupted
            or reason.strip().lower()
            in {
                "interrupted",
                "crash",
                "disk_error",
                "forced",
            }
        )
        status = "succeeded" if success is True else "failed" if success is False else "aborted"
        if interrupted:
            status = "interrupted"
        finished_ns = time.time_ns()
        result = {
            "episode_id": metadata.get("episode_id", self._path.name),
            "episode_path": str(self._path),
            "success": success,
            "status": status,
            "reason": reason.strip(),
            "finished_timestamp_ns": finished_ns,
            "frame_count": scan.frame_count,
            "dropped_gap_analysis": gap_analysis,
            "gap_analysis": gap_analysis,
            "incomplete_final_line": scan.incomplete_final_line,
        }
        _write_json(self._path / RESULTS_FILE, result)
        metadata.update(
            {
                "status": status,
                "finished_timestamp_ns": finished_ns,
                "result": result,
            }
        )
        _write_json(metadata_path, metadata)
        self._frame_count = scan.frame_count
        self._active = False
        return result

    def force_interrupted_close(self, reason: str = "interrupted") -> dict[str, Any]:
        """Persist an interrupted/unknown result after a gateway failure."""

        return self.finish(None, reason=reason, interrupted=True)

    def interrupt(self, reason: str = "interrupted") -> dict[str, Any]:
        return self.force_interrupted_close(reason)

    def close(self, *, interrupted: bool = False, reason: str = "operator") -> dict[str, Any]:
        return self.finish(None, reason="interrupted" if interrupted else reason)

    @classmethod
    def resume(cls, path: Path) -> EpisodeRecorder:
        """Reopen an unfinished episode and truncate only an incomplete final line."""

        episode = Path(path)
        metadata = _read_json_object(episode / METADATA_FILE)
        if (episode / RESULTS_FILE).exists() or metadata.get("status") not in {None, "recording"}:
            raise RuntimeError("only an unfinished recording episode can be resumed")
        fps_value = metadata.get("fps", metadata.get("recording_fps", 20.0))
        joints = metadata.get("joint_names", DEFAULT_JOINT_NAMES)
        recorder = cls(episode.parent, fps=fps_value, joint_names=joints)
        scan = _scan_frames(episode, expected_fps=recorder.fps, raise_on_corruption=True)
        frames_path = episode / FRAMES_FILE
        if scan.incomplete_final_line:
            with frames_path.open("r+b") as stream:
                stream.truncate(scan.last_valid_offset)
                stream.flush()
                os.fsync(stream.fileno())
        elif scan.final_valid_line_without_newline:
            with frames_path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        recorder._path = episode
        recorder._active = True
        recorder._frame_count = scan.frame_count
        return recorder


def _gap_analysis(scan: _FrameScan, fps: float, metadata: Mapping[str, Any]) -> dict[str, Any]:
    expected_ns = 1_000_000_000.0 / fps
    max_gap_ms = _declared_max_ms(metadata, "max_gap_ms", "max_resample_gap_ms")
    return {
        "policy": "report_explicit_frame_indices_and_timestamp_irregularity; no implicit timing threshold",
        "expected_interval_ms": expected_ns / 1_000_000.0,
        "timestamp_sample_count": scan.timestamp_sample_count,
        "irregular_interval_count": scan.irregular_interval_count,
        "largest_interval_ms": None if scan.largest_interval_ns is None else scan.largest_interval_ns / 1_000_000.0,
        "nonmonotonic_timestamps": scan.nonmonotonic_timestamps,
        "frame_index_gap_count": scan.frame_index_gap_count,
        "explicit_dropped_count": scan.explicit_dropped_count,
        "configured_max_gap_ms": max_gap_ms,
    }


@dataclass
class _Validation:
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_codes: set[str] = field(default_factory=set)
    warning_codes: set[str] = field(default_factory=set)
    frame_count: int = 0
    expected_frame_index: int = 0
    state_ok: bool = True
    action_ok: bool = True
    camera_ok: bool = True
    timestamp_ok: bool = True
    feedback_ok: bool = True
    units_ok: bool = True
    provenance_ok: bool = True
    base_ok: bool = True
    joint_names_ok: bool = True

    def error(self, code: str, message: str) -> None:
        self.error_codes.add(code)
        if message not in self.errors and len(self.errors) < 100:
            self.errors.append(message)

    def warning(self, code: str, message: str) -> None:
        self.warning_codes.add(code)
        if message not in self.warnings and len(self.warnings) < 100:
            self.warnings.append(message)


def _validate_frame(frame: Mapping[str, Any], report: _Validation, metadata: Mapping[str, Any]) -> None:
    index = frame.get("frame_index", frame.get("index"))
    if not isinstance(index, int) or isinstance(index, bool):
        report.error("frame_index", f"frame {report.frame_count} has no integer frame_index")
    elif index != report.expected_frame_index:
        report.warning(
            "frame_index_gap",
            f"frame index gap near frame {report.frame_count}: expected {report.expected_frame_index}, got {index}",
        )
    report.expected_frame_index = (index + 1) if isinstance(index, int) else report.expected_frame_index + 1

    observation = frame.get("observation")
    action = frame.get("action")
    action_values = _action_vector(action)
    if not isinstance(observation, Mapping):
        report.state_ok = False
        report.error("observation", f"frame {report.frame_count} is missing observation")
    elif _state_vector(observation) is None:
        report.state_ok = False
        report.error(
            "observation_state", f"frame {report.frame_count} lacks exactly the canonical 12-joint observation.state"
        )
    if action_values is None:
        report.action_ok = False
        report.error("action", f"frame {report.frame_count} lacks a canonical 12-joint command action")

    paths = _canonical_image_paths(frame, metadata)
    if paths is None or any(role not in paths for role in CAMERA_ROLES):
        report.camera_ok = False
        report.error(
            "camera_roles", f"frame {report.frame_count} must contain front, left_wrist, and right_wrist images"
        )
    else:
        for role in CAMERA_ROLES:
            try:
                image_path = _safe_episode_file(report.path, paths[role])
            except (TypeError, ValueError) as exc:
                report.camera_ok = False
                report.error("camera_path", f"frame {report.frame_count} {role} image path is invalid: {exc}")
                continue
            if not image_path.is_file():
                report.camera_ok = False
                report.error("camera_file", f"frame {report.frame_count} {role} image is missing: {paths[role]}")
            elif not _jpeg_signature(image_path):
                report.camera_ok = False
                report.error("camera_jpeg", f"frame {report.frame_count} {role} image is not a complete JPEG")

    timestamps = _frame_timestamps(frame)
    domains = _frame_timestamp_domains(frame)
    source = timestamps["source_timestamp_ns"]
    state = timestamps["state_timestamp_ns"]
    cameras = timestamps["camera_timestamps_ns"]
    for name, value in (("source_timestamp_ns", source), ("state_timestamp_ns", state)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            report.timestamp_ok = False
            report.error("timestamps", f"frame {report.frame_count} is missing a valid {name}")
    if not isinstance(cameras, Mapping):
        report.timestamp_ok = False
        report.error("timestamps", f"frame {report.frame_count} is missing camera_timestamps_ns")
    else:
        aliases = _camera_role_aliases(metadata)
        canonical_cameras: dict[str, Any] = {}
        for key, value in cameras.items():
            role = aliases.get(key)
            if role is not None:
                canonical_cameras[role] = value
        for role in CAMERA_ROLES:
            value = canonical_cameras.get(role)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                report.timestamp_ok = False
                report.error("timestamps", f"frame {report.frame_count} is missing camera timestamp for {role}")
    max_skew_ms = _declared_max_ms(metadata, "max_skew_ms", "max_timestamp_skew_ms")
    if max_skew_ms is None:
        report.timestamp_ok = False
        report.error("timestamp_policy", "metadata.max_skew_ms is required to validate timestamp alignment")
    elif (
        isinstance(source, int)
        and not isinstance(source, bool)
        and isinstance(state, int)
        and not isinstance(state, bool)
        and isinstance(cameras, Mapping)
        and domains["source"] == domains["state"] == domains["camera"]
    ):
        values = [state]
        aliases = _camera_role_aliases(metadata)
        for key, value in cameras.items():
            if aliases.get(key) in CAMERA_ROLES and isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
        skew_ns = max(abs(value - source) for value in values) if values else 0
        if skew_ns > max_skew_ms * 1_000_000.0:
            report.timestamp_ok = False
            report.error(
                "timestamp_alignment",
                f"frame {report.frame_count} timestamp skew {skew_ns / 1_000_000.0:.3f} ms exceeds configured max_skew_ms={max_skew_ms}",
            )
    elif isinstance(source, int) and not isinstance(source, bool):
        mismatched_domains = [name for name in ("state", "camera") if domains[name] != domains["source"]]
        if mismatched_domains:
            report.timestamp_ok = False
            report.error(
                "timestamp_alignment",
                f"frame {report.frame_count} has timestamp domains {mismatched_domains} that differ from source",
            )
    received = timestamps["received_timestamp_ns"]
    if received is not None:
        if not isinstance(received, int) or isinstance(received, bool) or received < 0:
            report.timestamp_ok = False
            report.error("timestamps", f"frame {report.frame_count} has an invalid received_timestamp_ns")
        elif domains["received"] != domains["source"]:
            report.warning(
                "timestamp_domains",
                f"frame {report.frame_count} received timestamp is in a different clock domain; freshness order is not compared",
            )
        elif isinstance(source, int) and not isinstance(source, bool):
            if received < source:
                report.timestamp_ok = False
                report.error(
                    "timestamp_freshness",
                    f"frame {report.frame_count} received_timestamp_ns precedes source_timestamp_ns",
                )
            max_age_ms = _declared_max_ms(
                metadata,
                "max_observation_age_ms",
                "max_source_age_ms",
                "max_freshness_ms",
            )
            if max_age_ms is None:
                report.warning(
                    "freshness_policy",
                    "received timestamps are present but metadata has no max_observation_age_ms; freshness is not thresholded",
                )
            elif isinstance(source, int) and received - source > max_age_ms * 1_000_000.0:
                report.timestamp_ok = False
                report.error(
                    "timestamp_freshness",
                    f"frame {report.frame_count} source age {(received - source) / 1_000_000.0:.3f} ms exceeds configured max_observation_age_ms={max_age_ms}",
                )

    joint_unit, gripper_unit = _unit_values(metadata, observation, action)
    if joint_unit != "degrees":
        report.units_ok = False
        report.error("units", "joint position unit must be explicitly declared as degrees")
    if gripper_unit != "range_0_100":
        report.units_ok = False
        report.error("units", "gripper unit must be explicitly declared as range_0_100")

    feedback = frame.get("feedback")
    sent_timestamp = feedback.get("sent_timestamp_ns") if isinstance(feedback, Mapping) else None
    if not isinstance(sent_timestamp, int) or isinstance(sent_timestamp, bool) or sent_timestamp < 0:
        report.feedback_ok = False
        report.error("sent_timestamp", f"frame {report.frame_count} lacks the controller write sent_timestamp_ns")
    elif domains["sent"] != domains["source"]:
        report.timestamp_ok = False
        report.error(
            "timestamp_alignment",
            f"frame {report.frame_count} sent timestamp domain differs from source",
        )
    elif isinstance(source, int) and not isinstance(source, bool) and max_skew_ms is not None:
        sent_skew_ns = abs(sent_timestamp - source)
        if sent_skew_ns > max_skew_ms * 1_000_000.0:
            report.timestamp_ok = False
            report.error(
                "timestamp_alignment",
                f"frame {report.frame_count} sent/source skew {sent_skew_ns / 1_000_000.0:.3f} ms exceeds configured max_skew_ms={max_skew_ms}",
            )
    applied_feedback = feedback.get("applied_action") if isinstance(feedback, Mapping) else None
    applied_values = _action_vector(applied_feedback)
    if not _feedback_is_applied(feedback) or applied_values is None:
        report.feedback_ok = False
        report.error("applied_feedback", f"frame {report.frame_count} lacks a canonical applied-action feedback record")
    elif action_values != applied_values:
        report.action_ok = False
        report.feedback_ok = False
        report.error("action_provenance", f"frame {report.frame_count} action differs from applied-action feedback")

    observation_base = _base_velocity(observation)
    if observation_base is None and _base_is_enabled(metadata):
        report.base_ok = False
        report.error("mobile_base", f"frame {report.frame_count} is missing enabled mobile-base velocity evidence")
    for base in (observation_base, _base_velocity(action)):
        if base is not None and any(value != 0.0 for value in base) and not _supports_mobile_base(metadata):
            report.base_ok = False
            report.error("mobile_base", f"frame {report.frame_count} contains nonzero mobile-base velocity")


def validate_episode(path: Path) -> dict[str, Any]:
    """Validate an episode without importing LeRobot or mutating its files."""

    episode = Path(path)
    report = _Validation(episode)
    try:
        report.metadata = _read_json_object(episode / METADATA_FILE)
    except EpisodeError as exc:
        report.error("metadata", str(exc))
        return _validation_result(report, None)

    declared_joints = report.metadata.get("joint_names")
    try:
        canonical_joints = tuple(declared_joints)
    except TypeError:
        canonical_joints = ()
    if canonical_joints != DEFAULT_JOINT_NAMES:
        report.joint_names_ok = False
        report.error("joint_names", "episode must declare the canonical left6-then-right6 12 arm joints")
    source = _source_value(report.metadata)
    physical, teleop, demo = _provenance_state(report.metadata)
    if demo:
        report.provenance_ok = False
        report.error("provenance", "simulation/demo/replay/read-only source is not trainable by default")
    if not physical:
        report.provenance_ok = False
        report.error("provenance", "episode source must explicitly identify physical hardware")
    if not teleop:
        report.provenance_ok = False
        report.error("provenance", "episode source must explicitly identify teleoperation")
    if not _camera_metadata_confirmed(report.metadata):
        report.camera_ok = False
        report.error(
            "camera_metadata",
            "metadata must confirm camera_roles_confirmed=true and name front, left_wrist, and right_wrist",
        )

    try:
        fps = _positive_finite(report.metadata.get("fps", report.metadata.get("recording_fps")), "metadata.fps")
    except (TypeError, ValueError):
        fps = None
        report.error("fps", "episode metadata must contain a positive finite fps")

    scan: _FrameScan | None = None
    try:

        def check(frame: dict[str, Any]) -> None:
            report.frame_count += 1
            _validate_frame(frame, report, report.metadata)

        scan = _scan_frames(
            episode,
            expected_fps=fps,
            on_frame=check,
            raise_on_corruption=True,
        )
    except EpisodeError as exc:
        report.error("json_corruption", str(exc))
    if scan is not None and scan.incomplete_final_line:
        report.warning(
            "incomplete_final_line", "the incomplete final JSONL line is ignored; resume may truncate only that line"
        )
        report.error_codes.add("incomplete_final_line")
    if report.frame_count == 0 and scan is not None:
        report.error("empty", "episode contains no complete frames")
    if scan is not None and scan.nonmonotonic_timestamps:
        report.timestamp_ok = False
        report.error("timestamps", "observation timestamps are not strictly increasing")

    return _validation_result(report, scan, source=source)


def _validation_result(
    report: _Validation,
    scan: _FrameScan | None,
    *,
    source: Any = None,
) -> dict[str, Any]:
    fps_value = report.metadata.get("fps", report.metadata.get("recording_fps"))
    try:
        fps = _positive_finite(fps_value, "metadata.fps")
    except (TypeError, ValueError):
        fps = None
    gap = _gap_analysis(scan, fps, report.metadata) if scan is not None and fps is not None else None
    fatal_codes = {
        "metadata",
        "json_corruption",
        "empty",
        "observation",
        "observation_state",
        "action",
        "camera_roles",
        "camera_metadata",
        "camera_path",
        "camera_file",
        "camera_jpeg",
        "frame_index",
        "timestamps",
        "timestamp_policy",
        "timestamp_alignment",
        "timestamp_freshness",
        "units",
        "applied_feedback",
        "sent_timestamp",
        "joint_names",
        "mobile_base",
        "provenance",
        "fps",
        "incomplete_final_line",
    }
    trainable = (
        report.frame_count > 0
        and not (report.error_codes & fatal_codes)
        and report.state_ok
        and report.action_ok
        and report.camera_ok
        and report.timestamp_ok
        and report.feedback_ok
        and report.units_ok
        and report.provenance_ok
        and report.base_ok
        and report.joint_names_ok
        and _first_metadata_value(report.metadata, "trainable") is not False
    )
    return {
        "path": str(report.path),
        "episode_id": report.metadata.get("episode_id", report.path.name),
        "source": source if source is not None else _source_value(report.metadata),
        "task": report.metadata.get("task"),
        "fps": fps,
        "frame_count": report.frame_count,
        "trainable": trainable,
        "timestamp_ok": report.timestamp_ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "error_codes": sorted(report.error_codes),
        "warning_codes": sorted(report.warning_codes),
        "gap_analysis": gap,
        "dropped_gap_analysis": gap,
        "metadata": report.metadata,
    }


@dataclass
class _EpisodeBounds:
    first_ns: int | None = None
    last_ns: int | None = None
    first_action_ns: int | None = None
    last_action_ns: int | None = None
    source_fps: float | None = None
    frame_count: int = 0
    irregular: bool = False


def _episode_bounds(episode: Path, metadata: Mapping[str, Any]) -> _EpisodeBounds:
    source_fps_value = metadata.get("fps", metadata.get("recording_fps"))
    try:
        source_fps = _positive_finite(source_fps_value, "metadata.fps")
    except (TypeError, ValueError):
        source_fps = None
    bounds = _EpisodeBounds(source_fps=source_fps)
    previous: int | None = None
    expected_ns = None if source_fps is None else 1_000_000_000.0 / source_fps
    for frame in _load_frames(episode):
        timestamp = _frame_observation_time(frame)
        if timestamp is None:
            raise LeRobotExportError(f"{episode}: every exported frame needs an observation timestamp")
        if bounds.first_ns is None:
            bounds.first_ns = timestamp
        bounds.last_ns = timestamp
        bounds.frame_count += 1
        action_timestamp = _frame_action_time(frame)
        if action_timestamp is not None:
            if bounds.first_action_ns is None:
                bounds.first_action_ns = action_timestamp
            bounds.last_action_ns = action_timestamp
        if previous is not None:
            if timestamp <= previous:
                raise LeRobotExportError(f"{episode}: observation timestamps are not strictly increasing")
            if expected_ns is None or abs((timestamp - previous) - expected_ns) > 1.0:
                bounds.irregular = True
        previous = timestamp
    return bounds


def _decode_jpeg(path: Path, image_module: Any, numpy_module: Any) -> Any:
    try:
        with image_module.open(path) as image:
            return numpy_module.asarray(image.convert("RGB"), dtype=numpy_module.uint8).copy()
    except Exception as exc:
        raise LeRobotExportError(f"cannot decode JPEG {path}: {exc}") from exc


def _validate_export_frame(frame: Mapping[str, Any], episode: Path, metadata: Mapping[str, Any]) -> None:
    if _state_vector(frame.get("observation")) is None:
        raise LeRobotExportError(f"{episode}: frame lacks canonical 12D observation.state")
    if _action_vector(frame.get("action")) is None:
        raise LeRobotExportError(f"{episode}: frame lacks canonical 12D action")
    paths = _canonical_image_paths(frame, metadata)
    if paths is None or any(role not in paths for role in CAMERA_ROLES):
        raise LeRobotExportError(f"{episode}: frame lacks the three known camera roles")
    for role in CAMERA_ROLES:
        path = _safe_episode_file(episode, paths[role])
        if not path.is_file() or not _jpeg_signature(path):
            raise LeRobotExportError(f"{episode}: {role} image is missing or is not a complete JPEG")
    timestamps = _frame_timestamps(frame)
    if not isinstance(timestamps["source_timestamp_ns"], int) or not isinstance(timestamps["state_timestamp_ns"], int):
        raise LeRobotExportError(f"{episode}: frame needs source_timestamp_ns and state_timestamp_ns")
    if not isinstance(timestamps["camera_timestamps_ns"], Mapping):
        raise LeRobotExportError(f"{episode}: frame needs camera_timestamps_ns")


def _resample_policy(
    metadata: Mapping[str, Any], *, source_fps: float | None, target_fps: float, irregular: bool
) -> dict[str, Any]:
    mismatch = source_fps is None or abs(source_fps - target_fps) > 1e-9
    if not mismatch and not irregular:
        return {
            "mode": "direct_regular",
            "source_fps": source_fps,
            "target_fps": target_fps,
            "irregular_source": False,
            "policy": "source timestamps are exactly nominal; no timestamps are fabricated",
            "max_gap_ms": None,
        }
    max_gap_ns = _declared_max_gap_ns(metadata)
    if max_gap_ns is None:
        raise LeRobotExportError(
            "irregular or mismatched episode sampling requires metadata.max_gap_ms "
            "(or max_gap_ns/max_resample_gap_ms) for nearest-previous resampling"
        )
    return {
        "mode": "nearest_previous_valid",
        "source_fps": source_fps,
        "target_fps": target_fps,
        "irregular_source": irregular,
        "policy": "hold nearest previous valid observation and action; reject age above configured max gap",
        "max_gap_ms": max_gap_ns / 1_000_000.0,
    }


def _features(image_shapes: Mapping[str, tuple[int, ...]], *, include_base: bool) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": list(DEFAULT_JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": list(DEFAULT_JOINT_NAMES),
        },
    }
    for role in CAMERA_ROLES:
        result[f"observation.images.{role}"] = {
            "dtype": "image",
            "shape": image_shapes[role],
            "names": ["height", "width", "channel"],
        }
    if include_base:
        result["observation.base_velocity"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["vx_m_s", "vy_m_s", "yaw_rate_rad_s"],
        }
    return result


def _read_direct_frame(
    frame: Mapping[str, Any],
    episode: Path,
    metadata: Mapping[str, Any],
    *,
    image_module: Any,
    numpy_module: Any,
    include_base: bool,
) -> dict[str, Any]:
    _validate_export_frame(frame, episode, metadata)
    state = _state_vector(frame["observation"])
    action = _action_vector(frame["action"])
    assert state is not None and action is not None
    paths = _canonical_image_paths(frame, metadata)
    assert paths is not None
    output: dict[str, Any] = {
        "observation.state": numpy_module.asarray(state, dtype=numpy_module.float32),
        "action": numpy_module.asarray(action, dtype=numpy_module.float32),
        "task": metadata.get("task", ""),
    }
    shapes: dict[str, tuple[int, ...]] = {}
    for role in CAMERA_ROLES:
        image = _decode_jpeg(_safe_episode_file(episode, paths[role]), image_module, numpy_module)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise LeRobotExportError(f"{episode}: {role} JPEG did not decode to HWC RGB")
        output[f"observation.images.{role}"] = image
        shapes[role] = tuple(int(size) for size in image.shape)
    if include_base:
        base = _base_velocity(frame.get("observation"))
        if base is None:
            raise LeRobotExportError(f"{episode}: explicit mobile-base support requires base velocity on every frame")
        if len(base) != 3:
            raise LeRobotExportError(f"{episode}: supported base velocity must contain vx, vy, yaw rate")
        output["observation.base_velocity"] = numpy_module.asarray(base, dtype=numpy_module.float32)
    output["_image_shapes"] = shapes
    return output


def _add_frame(dataset: Any, frame: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in frame.items() if not key.startswith("_")}
    try:
        dataset.add_frame(payload)
    except Exception as exc:
        raise LeRobotExportError(f"LeRobotDataset.add_frame failed: {exc}") from exc


def _export_direct_episode(
    dataset: Any,
    episode: Path,
    metadata: Mapping[str, Any],
    *,
    image_module: Any,
    numpy_module: Any,
    include_base: bool,
    image_shapes: dict[str, tuple[int, ...]] | None,
) -> tuple[int, dict[str, tuple[int, ...]]]:
    count = 0
    shapes = image_shapes or {}
    for frame in _load_frames(episode):
        converted = _read_direct_frame(
            frame,
            episode,
            metadata,
            image_module=image_module,
            numpy_module=numpy_module,
            include_base=include_base,
        )
        current_shapes = converted.pop("_image_shapes")
        if shapes and current_shapes != shapes:
            raise LeRobotExportError(f"{episode}: camera image shape changed within/across episodes")
        shapes.update(current_shapes)
        _add_frame(dataset, converted)
        count += 1
    if count == 0:
        raise LeRobotExportError(f"{episode}: no complete frames to export")
    try:
        dataset.save_episode()
    except Exception as exc:
        raise LeRobotExportError(f"LeRobotDataset.save_episode failed for {episode}: {exc}") from exc
    return count, shapes


def _export_resampled_episode(
    dataset: Any,
    episode: Path,
    metadata: Mapping[str, Any],
    *,
    target_fps: float,
    bounds: _EpisodeBounds,
    image_module: Any,
    numpy_module: Any,
    include_base: bool,
    image_shapes: dict[str, tuple[int, ...]] | None,
) -> tuple[int, dict[str, tuple[int, ...]], dict[str, Any]]:
    assert bounds.first_ns is not None and bounds.last_ns is not None
    max_gap_ns = _declared_max_gap_ns(metadata)
    assert max_gap_ns is not None
    period_ns = 1_000_000_000.0 / target_fps
    target_start_ns = max(bounds.first_ns, bounds.first_action_ns or bounds.first_ns)
    if target_start_ns > bounds.last_ns:
        raise LeRobotExportError(f"{episode}: no time has both an observation and a command action")
    target_count = math.floor((bounds.last_ns - target_start_ns) / period_ns + 1e-9) + 1
    observation_source = iter(_load_frames(episode))
    action_source = iter(_load_frames(episode))
    next_observation: dict[str, Any] | None = next(observation_source, None)
    next_action: dict[str, Any] | None = next(action_source, None)
    previous_observation: tuple[int, dict[str, Any]] | None = None
    previous_action: tuple[int, dict[str, Any]] | None = None
    shapes = image_shapes or {}
    emitted = 0
    max_observation_age = 0
    max_action_age = 0
    gap_count = 0

    for target_index in range(target_count):
        # Keep the epoch-sized origin as an integer.  Converting an absolute
        # nanosecond timestamp (about 1e18) to float can move the first target
        # by hundreds of nanoseconds and incorrectly place it before the first
        # valid observation/action.
        target_ns = target_start_ns + round(target_index * period_ns)
        while next_observation is not None:
            timestamp = _frame_observation_time(next_observation)
            if timestamp is None or timestamp > target_ns:
                break
            _validate_export_frame(next_observation, episode, metadata)
            previous_observation = (timestamp, next_observation)
            next_observation = next(observation_source, None)

        while next_action is not None:
            action_timestamp = _frame_action_time(next_action)
            if action_timestamp is None or action_timestamp > target_ns:
                break
            _validate_export_frame(next_action, episode, metadata)
            previous_action = (action_timestamp, next_action)
            next_action = next(action_source, None)

        if previous_observation is None or previous_action is None:
            gap_count += 1
            raise LeRobotExportError(
                f"{episode}: no previous valid observation/action exists for resample time {target_ns}"
            )
        observation_age = target_ns - previous_observation[0]
        action_age = target_ns - previous_action[0]
        if observation_age < 0 or action_age < 0 or observation_age > max_gap_ns or action_age > max_gap_ns:
            gap_count += 1
            raise LeRobotExportError(
                f"{episode}: nearest-previous sample exceeds configured max gap at output frame {target_index}"
            )
        max_observation_age = max(max_observation_age, observation_age)
        max_action_age = max(max_action_age, action_age)

        chosen = previous_observation[1]
        action_frame = previous_action[1]
        state = _state_vector(chosen["observation"])
        action = _action_vector(action_frame["action"])
        assert state is not None and action is not None
        paths = _canonical_image_paths(chosen, metadata)
        assert paths is not None
        output: dict[str, Any] = {
            "observation.state": numpy_module.asarray(state, dtype=numpy_module.float32),
            "action": numpy_module.asarray(action, dtype=numpy_module.float32),
            "task": metadata.get("task", ""),
        }
        for role in CAMERA_ROLES:
            image = _decode_jpeg(_safe_episode_file(episode, paths[role]), image_module, numpy_module)
            current_shape = tuple(int(size) for size in image.shape)
            if shapes and current_shape != shapes[role]:
                raise LeRobotExportError(f"{episode}: camera image shape changed within/across episodes")
            shapes[role] = current_shape
            output[f"observation.images.{role}"] = image
        if include_base:
            base = _base_velocity(chosen.get("observation"))
            if base is None:
                raise LeRobotExportError(f"{episode}: supported base velocity is missing on a selected frame")
            output["observation.base_velocity"] = numpy_module.asarray(base, dtype=numpy_module.float32)
        _add_frame(dataset, output)
        emitted += 1
    if emitted == 0:
        raise LeRobotExportError(f"{episode}: resampling produced no frames")
    try:
        dataset.save_episode()
    except Exception as exc:
        raise LeRobotExportError(f"LeRobotDataset.save_episode failed for {episode}: {exc}") from exc
    return (
        emitted,
        shapes,
        {
            "mode": "nearest_previous_valid",
            "source_fps": bounds.source_fps,
            "target_fps": target_fps,
            "irregular_source": bounds.irregular,
            "policy": "hold nearest previous valid observation and action; reject age above configured max gap",
            "max_gap_ms": max_gap_ns / 1_000_000.0,
            "max_observation_age_ms": max_observation_age / 1_000_000.0,
            "max_action_age_ms": max_action_age / 1_000_000.0,
            "gap_count": gap_count,
            "output_frame_count": emitted,
        },
    )


def export_lerobot(
    episode_paths: Sequence[Path],
    *,
    repo_id: str,
    output: Path,
    fps: float = 20.0,
    allow_demo: bool = False,
    image_writer_threads: int = 0,
) -> dict[str, Any]:
    """Export validated episodes through the official LeRobot dataset writer.

    The output path must not exist.  With ``allow_demo=False`` every source
    episode must be trainable according to :func:`validate_episode`; demo or
    simulation data can only be exported explicitly and remains marked
    non-trainable in this function's report.
    """

    if isinstance(episode_paths, (str, bytes, bytearray, Path)) or not isinstance(episode_paths, Sequence):
        raise TypeError("episode_paths must be a non-empty sequence of paths")
    paths = [Path(path) for path in episode_paths]
    if not paths:
        raise ValueError("episode_paths must not be empty")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-blank string")
    if isinstance(image_writer_threads, bool) or not isinstance(image_writer_threads, int):
        raise TypeError("image_writer_threads must be an integer")
    if image_writer_threads < 0:
        raise ValueError("image_writer_threads must be non-negative")
    target_fps = _positive_finite(fps, "fps")
    destination = Path(output)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to append to existing LeRobot output: {destination}")

    reports = [validate_episode(path) for path in paths]
    for report in reports:
        if not allow_demo and not report["trainable"]:
            raise LeRobotExportError(f"episode is not trainable: {report['path']}; errors={report['errors']}")
        fatal = {
            "metadata",
            "json_corruption",
            "empty",
            "observation",
            "observation_state",
            "action",
            "action_provenance",
            "camera_roles",
            "camera_path",
            "camera_file",
            "camera_jpeg",
            "timestamps",
            "timestamp_alignment",
            "units",
            "joint_names",
            "frame_index",
            "fps",
            "incomplete_final_line",
        }
        if fatal.intersection(report["error_codes"]):
            raise LeRobotExportError(
                f"episode cannot be structurally exported: {report['path']}; errors={report['errors']}"
            )

    # Dependencies are intentionally imported only after the no-append and
    # source validation checks, so an invalid request does not create output.
    try:
        dataset_module = importlib.import_module("lerobot.datasets.lerobot_dataset")
        dataset_class = dataset_module.LeRobotDataset
    except Exception as exc:
        raise LeRobotExportError(
            "LeRobot export requires the modern LeRobot package (0.4+/0.5); "
            "install the project extra, for example `pip install 'lerobot>=0.4'`."
        ) from exc
    try:
        numpy_module = importlib.import_module("numpy")
    except Exception as exc:
        raise LeRobotExportError(
            "LeRobot export requires NumPy; install the project's xlerobot/lerobot extra."
        ) from exc
    try:
        image_module = importlib.import_module("PIL.Image")
    except Exception as exc:
        raise LeRobotExportError(
            "LeRobot export requires Pillow to decode the recorded JPEGs; install `pillow>=10`."
        ) from exc

    for path, report in zip(paths, reports):
        metadata = report["metadata"]
        support_base = _supports_mobile_base(metadata)
        base_enabled = _base_is_enabled(metadata)
        for frame in _load_frames(path):
            observation_base = _base_velocity(frame.get("observation"))
            if base_enabled and observation_base is None:
                raise LeRobotExportError(
                    f"{path}: metadata enables the mobile base but frame lacks measured base velocity evidence"
                )
            for base in (observation_base, _base_velocity(frame.get("action"))):
                if base is not None and any(value != 0.0 for value in base) and not support_base:
                    raise LeRobotExportError(
                        f"{path}: nonzero mobile-base velocity cannot be exported with the 12D arm schema"
                    )

    bounds = [_episode_bounds(path, report["metadata"]) for path, report in zip(paths, reports)]
    include_base = any(
        _supports_mobile_base(report["metadata"])
        and any(_base_velocity(frame.get("observation")) is not None for frame in _load_frames(path))
        for path, report in zip(paths, reports)
    )
    image_shapes: dict[str, tuple[int, ...]] | None = None
    features: dict[str, dict[str, Any]] | None = None
    dataset: Any = None
    total_frames = 0
    resampling: list[dict[str, Any]] = []
    try:
        # Determine the image schema before create; LeRobot requires fixed
        # dimensions for each image feature and we never resize raw evidence.
        for path, report, bound in zip(paths, reports, bounds):
            policy = _resample_policy(
                report["metadata"],
                source_fps=bound.source_fps,
                target_fps=target_fps,
                irregular=bound.irregular,
            )
            first_frame = next(_load_frames(path), None)
            if first_frame is None:
                raise LeRobotExportError(f"{path}: no complete frames to export")
            converted = _read_direct_frame(
                first_frame,
                path,
                report["metadata"],
                image_module=image_module,
                numpy_module=numpy_module,
                include_base=include_base,
            )
            current_shapes = converted.pop("_image_shapes")
            if image_shapes is None:
                image_shapes = current_shapes
            elif current_shapes != image_shapes:
                raise LeRobotExportError(f"{path}: camera image shape differs from another episode")
            resampling.append(policy)
        assert image_shapes is not None
        features = _features(image_shapes, include_base=include_base)
        try:
            dataset = dataset_class.create(
                repo_id=repo_id.strip(),
                fps=target_fps,
                features=features,
                root=destination,
                robot_type="xlerobot",
                use_videos=False,
                image_writer_threads=image_writer_threads,
            )
        except Exception as exc:
            raise LeRobotExportError(f"LeRobotDataset.create failed: {exc}") from exc

        # The first-frame decode above is only schema discovery; all episode
        # frames are read again and passed through the writer.
        for index, (path, report, bound) in enumerate(zip(paths, reports, bounds)):
            if resampling[index]["mode"] == "direct_regular":
                count, image_shapes = _export_direct_episode(
                    dataset,
                    path,
                    report["metadata"],
                    image_module=image_module,
                    numpy_module=numpy_module,
                    include_base=include_base,
                    image_shapes=image_shapes,
                )
                total_frames += count
                resampling[index] = {
                    **resampling[index],
                    "output_frame_count": count,
                    "max_observation_age_ms": 0.0,
                    "max_action_age_ms": 0.0,
                }
            else:
                count, image_shapes, detail = _export_resampled_episode(
                    dataset,
                    path,
                    report["metadata"],
                    target_fps=target_fps,
                    bounds=bound,
                    image_module=image_module,
                    numpy_module=numpy_module,
                    include_base=include_base,
                    image_shapes=image_shapes,
                )
                total_frames += count
                resampling[index] = detail
    finally:
        if dataset is not None:
            finalize = getattr(dataset, "finalize", None)
            if not callable(finalize):
                raise LeRobotExportError("LeRobotDataset does not provide required finalize()")
            try:
                finalize()
            except Exception as exc:
                raise LeRobotExportError(f"LeRobotDataset.finalize failed: {exc}") from exc

    return {
        "repo_id": repo_id.strip(),
        "output": str(destination),
        "episode_count": len(paths),
        "frame_count": total_frames,
        "fps": target_fps,
        "features": features,
        "allow_demo": allow_demo,
        "image_writer_threads": image_writer_threads,
        "trainable": bool(not allow_demo and all(report["trainable"] for report in reports)),
        "adapter": "official LeRobotDataset.create/add_frame/save_episode/finalize",
        "episodes": reports,
        "resampling": resampling,
    }


__all__ = [
    "CAMERA_ROLES",
    "CANONICAL_JOINT_NAMES",
    "DEFAULT_CANONICAL_JOINT_NAMES",
    "DEFAULT_JOINT_NAMES",
    "FRAMES_FILE",
    "KNOWN_CAMERA_ROLES",
    "METADATA_FILE",
    "RESULTS_FILE",
    "EpisodeCorruptionError",
    "EpisodeError",
    "EpisodeRecorder",
    "LeRobotExportError",
    "export_lerobot",
    "validate_episode",
]
