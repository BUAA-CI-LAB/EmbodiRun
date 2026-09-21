"""Serialization and model input views over immutable shared snapshots."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from embodirun.robots import RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

from .values import ObservationSnapshot


class ObservationViewError(RuntimeError):
    """A snapshot cannot provide the requested state view."""


def detached_value(value: Any) -> Any:
    """Convert immutable snapshot containers to a transport/runtime value."""

    if isinstance(value, Mapping):
        return {key: detached_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [detached_value(item) for item in value]
    if isinstance(value, frozenset):
        return [detached_value(item) for item in value]
    return value


def snapshot_robot_observation(snapshot: ObservationSnapshot) -> RobotObservation:
    """Build the model's state value from the exact shared snapshot."""

    if snapshot.state is None:
        raise ObservationViewError(f"snapshot {snapshot.observation_id!r} has no robot state")
    state_metadata = snapshot.metadata.get("state_metadata", {})
    metadata = dict(state_metadata) if isinstance(state_metadata, Mapping) else {}
    timestamp_s = metadata.get("timestamp_s")
    if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
        raise ObservationViewError(f"snapshot {snapshot.observation_id!r} has no robot timestamp_s")
    metadata.update(
        {
            "observation_id": snapshot.observation_id,
            "snapshot_id": snapshot.observation_id,
            "observation_captured_timestamp_ns": snapshot.source_timestamps_ns.get("state"),
            "observation_clock_domain": snapshot.clock_domains.get("state"),
        }
    )
    return RobotObservation(
        timestamp_s=float(timestamp_s),
        values=detached_value(snapshot.state),
        metadata=metadata,
    )


def snapshot_robot_payload(snapshot: ObservationSnapshot) -> dict[str, Any]:
    """Return the backward-compatible robot observe shape."""

    observation = snapshot_robot_observation(snapshot)
    # Snapshots freeze nested mappings; thaw the complete tree before the
    # HTTP layer serializes it, while leaving the shared snapshot immutable.
    metadata = detached_value(observation.metadata)
    metadata.pop("observation_id", None)
    metadata.pop("snapshot_id", None)
    metadata.pop("observation_captured_timestamp_ns", None)
    metadata.pop("observation_clock_domain", None)
    # ``timestamp_s`` is retained as the top-level legacy field; the producer
    # stores it in state metadata only so it cannot be used as freshness time.
    metadata.pop("timestamp_s", None)
    return {
        "timestamp_s": observation.timestamp_s,
        "values": detached_value(observation.values),
        "metadata": metadata,
    }


def frame_payload(frame: CameraFrame) -> dict[str, Any]:
    """Encode a frame while preserving the legacy observe response shape."""

    return {
        "name": frame.name,
        "mime_type": frame.mime_type,
        "data": base64.b64encode(frame.data).decode("ascii"),
    }


def snapshot_payload(
    snapshot: ObservationSnapshot,
    frames: Sequence[CameraFrame],
    runtime_id: str,
    required_source_ids: Sequence[str],
    *,
    require_state: bool,
    include_robot: bool,
) -> dict[str, Any]:
    """Build service output from one immutable snapshot and one runtime view."""

    required = tuple(dict.fromkeys(required_source_ids))
    errors = {source_id: snapshot.errors[source_id] for source_id in required if source_id in snapshot.errors}
    timestamps = [snapshot.source_timestamps_ns.get(source_id) for source_id in required]
    domains = {
        snapshot.clock_domains.get(source_id)
        for source_id in required
        if snapshot.clock_domains.get(source_id) is not None
    }
    missing = [
        source_id
        for source_id, timestamp in zip(required, timestamps)
        if source_id not in snapshot.source_timestamps_ns or timestamp is None
    ]
    aggregate_domain = snapshot.metadata.get("clock_domain")
    view_status = {
        "available": not errors and not missing,
        "stale": bool(errors or missing or (domains and (len(domains) != 1 or aggregate_domain not in domains))),
        "errors": errors,
        "missing_sources": missing,
        "captured_timestamp_ns": max(timestamp for timestamp in timestamps if timestamp is not None)
        if any(timestamp is not None for timestamp in timestamps)
        else None,
        "received_timestamp_ns": max(
            timestamp
            for source_id in required
            if (timestamp := snapshot.source_received_timestamps_ns.get(source_id)) is not None
        )
        if any(snapshot.source_received_timestamps_ns.get(source_id) is not None for source_id in required)
        else None,
        "clock_domains": {source_id: snapshot.clock_domains.get(source_id) for source_id in required},
        "skew_ns": (
            max(timestamp for timestamp in timestamps if timestamp is not None)
            - min(timestamp for timestamp in timestamps if timestamp is not None)
            if len(timestamps) == len(required)
            and required
            and all(timestamp is not None for timestamp in timestamps)
            and len(domains) == 1
            else None
        ),
        "global": {
            "available": snapshot.available,
            "stale": snapshot.stale,
            "errors": dict(snapshot.errors),
        },
    }
    result: dict[str, Any] = {
        "status": "ok",
        "runtime_id": runtime_id,
        "frames": [frame_payload(frame) for frame in frames],
        "snapshot_id": snapshot.observation_id,
        "timestamps": {
            "captured_timestamp_ns": snapshot.captured_timestamp_ns,
            "received_timestamp_ns": snapshot.received_timestamp_ns,
            "published_timestamp_ns": snapshot.published_timestamp_ns,
            "source_timestamps_ns": dict(snapshot.source_timestamps_ns),
            "source_received_timestamps_ns": dict(snapshot.source_received_timestamps_ns),
            "clock_domains": dict(snapshot.clock_domains),
            "skew_ns": snapshot.skew_ns,
        },
        "observation_status": view_status,
    }
    if include_robot:
        result["robot"] = snapshot_robot_payload(snapshot)
    elif require_state and snapshot.state is not None:
        # Internal callers may require state for model input while keeping the
        # old observe response camera-only unless explicitly requested.
        result["state_available"] = True
    return result


__all__ = [
    "ObservationViewError",
    "detached_value",
    "frame_payload",
    "snapshot_payload",
    "snapshot_robot_observation",
    "snapshot_robot_payload",
]
