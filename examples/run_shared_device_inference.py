#!/usr/bin/env python3
"""Run the software-only EmbodiInfer shared-device experiment.

This helper deliberately keeps the recorded camera source in ``examples/``.
It is a replay source, not a camera driver: original capture metadata is kept
in the frame profile while a synthetic host-monotonic virtual-sensor delivery
timestamp feeds the existing freshness path.
The helper never turns a recorded frame into live hardware evidence.

The policy-vector binding used by this experiment is explicit about its
uncertain dataset-native units.  ``--model-units`` is required so a future
caller cannot accidentally reuse this script with a degree or normalized
checkpoint.  No conversion is performed here.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import mimetypes
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.contracts import ControlServiceConfig
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.observation_values import ObservationSubscriptionClosed
from embodirun.services.control.recordings import ObservationRecorder
from embodirun.services.control.server import ControlHttpServer, ControlService
from embodirun.services.inference import (
    EmbodiInferHttpClient,
    PolicyObservation,
    PolicyResult,
    Session,
)

RUNTIME_ID = "pi05-policy-vector-replay"
ROBOT_ID = "policy-vector-replay"
FRONT_NAME = "observation.images.front"
WRIST_NAME = "observation.images.wrist"
POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
POLICY_FEATURE_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
STATE_NATIVE_NAMES = tuple(name.removesuffix(".pos") for name in POLICY_FEATURE_NAMES)
MODEL_UNITS = "dataset_native_unverified"
PROPOSED_STEPS = 50
PREFIX_STEPS = 3


class ReplayInputError(ValueError):
    """The recorded input is missing an unambiguous replay value."""


class ExperimentError(RuntimeError):
    """The software-only experiment cannot be safely assembled or run."""


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """One selected front/wrist pair and its provenance from ``frames.jsonl``."""

    line_number: int
    sequence: int
    front_path: Path
    wrist_path: Path
    captured_timestamp_ns: int | None
    original_clock_domain: str | None
    raw_timestamp: Any
    measured_state: Any
    commanded_action: Any

    def to_report(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "sequence": self.sequence,
            "front_path": str(self.front_path),
            "wrist_path": str(self.wrist_path),
            "captured_timestamp_ns": self.captured_timestamp_ns,
            "original_clock_domain": self.original_clock_domain,
            "raw_timestamp": self.raw_timestamp,
            "has_measured_present": self.measured_state is not None,
            "has_commanded_action": self.commanded_action is not None,
            "units": MODEL_UNITS,
        }


def native_state_from_record(record: ReplayRecord) -> tuple[float, ...]:
    """Read the measured ``present`` vector in the declared feature order.

    The raw export uses joint names under ``present`` and keeps ``action`` and
    ``sent`` as command-side values.  This function never substitutes those
    command fields for measured state and never converts the values.
    """

    value = record.measured_state
    if isinstance(value, Mapping):
        candidate = value.get("state_native")
        if candidate is None:
            candidate = value.get("values")
        if candidate is None and all(name in value for name in STATE_NATIVE_NAMES):
            candidate = [value[name] for name in STATE_NATIVE_NAMES]
        if candidate is None and all(name in value for name in POLICY_FEATURE_NAMES):
            candidate = [value[name] for name in POLICY_FEATURE_NAMES]
    else:
        candidate = value
    if isinstance(candidate, (str, bytes, bytearray)) or not isinstance(candidate, Sequence):
        raise ReplayInputError(f"record line {record.line_number} has no six-value measured present vector")
    if len(candidate) != 6:
        raise ReplayInputError(f"record line {record.line_number} measured present vector must contain 6 values")
    values: list[float] = []
    for index, item in enumerate(candidate):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReplayInputError(f"record line {record.line_number} present[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ReplayInputError(f"record line {record.line_number} present[{index}] must be finite")
        values.append(number)
    return tuple(values)


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield dotted paths without treating action values as camera files."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            path = f"{prefix}.{key}" if prefix else key
            yield path, child
            yield from _walk(child, path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield path, child
            yield from _walk(child, path)


def _path_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("path", "file", "filename", "uri"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _image_ref(
    record: Mapping[str, Any],
    requested_key: str | None,
    aliases: Sequence[str],
) -> str | None:
    """Resolve one image path while excluding command/state fields.

    Current LeRobot exports commonly use direct keys such as
    ``observation.images.camera2``.  The nested ``frames`` form is accepted as
    well so the helper remains useful for a copied recording manifest.
    """

    aliases_lower = tuple(alias.lower() for alias in aliases)
    requested_lower = requested_key.lower() if requested_key else None

    def label_matches(label: str) -> bool:
        normalized = label.lower()
        return bool(
            (requested_lower and (normalized == requested_lower or normalized.endswith("." + requested_lower)))
            or any(alias in normalized for alias in aliases_lower)
        )

    def nested_named_ref(value: Any) -> str | None:
        if isinstance(value, Mapping):
            labels = [
                item
                for key in ("name", "camera", "camera_name", "stream", "id")
                if isinstance((item := value.get(key)), str)
            ]
            ref = next(
                (
                    item
                    for key in ("path", "file", "filename", "uri")
                    if (item := value.get(key)) is not None and _path_value(item) is not None
                ),
                None,
            )
            if ref is not None and any(label_matches(label) for label in labels):
                return _path_value(ref)
            for child in value.values():
                found = nested_named_ref(child)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                found = nested_named_ref(child)
                if found is not None:
                    return found
        return None

    named = nested_named_ref(record)
    if named is not None:
        return named
    candidates: list[tuple[str, str]] = []
    for path, value in _walk(record):
        resolved = _path_value(value)
        if resolved is None:
            continue
        key = path.lower()
        leaf = key.rsplit(".", 1)[-1]
        if leaf in {"name", "camera", "camera_name", "stream", "id"}:
            continue
        if any(part in key for part in ("action", "sent", "goal", "command")):
            continue
        candidates.append((key, resolved))
    for key, value in candidates:
        if requested_lower and (key == requested_lower or key.endswith("." + requested_lower)):
            return value
    for key, value in candidates:
        if any(alias in key for alias in aliases_lower):
            return value
    return None


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    flattened = {path.rsplit(".", 1)[-1]: value for path, value in _walk(record)}
    for name in names:
        if name in flattened:
            return flattened[name]
    return None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _resolve_manifest(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "frames.jsonl"
    if not candidate.is_file():
        raise ReplayInputError(f"recorded input is not a file: {candidate}")
    return candidate


def load_replay_records(
    input_path: str | Path,
    *,
    front_key: str | None = None,
    wrist_key: str | None = None,
    record_line: int | None = None,
) -> tuple[ReplayRecord, ...]:
    """Load bounded camera pairs without using ``action`` as ``present``.

    ``captured_timestamp_ns`` is accepted only when its unit is explicit.  A
    generic ``timestamp`` is retained as provenance but is not guessed to be
    nanoseconds or seconds.
    """

    manifest = _resolve_manifest(input_path)
    records: list[ReplayRecord] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReplayInputError(f"cannot read recorded input {manifest}: {error}") from error
    if record_line is not None and (
        isinstance(record_line, bool) or not isinstance(record_line, int) or record_line <= 0
    ):
        raise ReplayInputError("record_line must be a positive line number")
    for line_number, line in enumerate(lines, 1):
        if record_line is not None and line_number != record_line:
            continue
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReplayInputError(f"line {line_number} is not JSON: {error}") from error
        if not isinstance(payload, Mapping):
            raise ReplayInputError(f"line {line_number} must be a JSON object")
        front_ref = _image_ref(
            payload,
            front_key,
            ("observation.images.front", "cameras.2", "camera2", "front", "main"),
        )
        wrist_ref = _image_ref(
            payload,
            wrist_key,
            ("observation.images.wrist", "cameras.0", "camera0", "wrist"),
        )
        if front_ref is None or wrist_ref is None:
            continue
        front_path = (manifest.parent / front_ref).resolve()
        wrist_path = (manifest.parent / wrist_ref).resolve()
        for label, value in (("front", front_path), ("wrist", wrist_path)):
            if not value.is_file():
                raise ReplayInputError(f"line {line_number} {label} image does not exist: {value}")
        sequence = _integer(_first(payload, ("index", "frame_index", "sequence", "step")))
        if sequence is None:
            sequence = line_number
        captured = _integer(_first(payload, ("captured_timestamp_ns", "capture_timestamp_ns", "timestamp_ns")))
        records.append(
            ReplayRecord(
                line_number=line_number,
                sequence=sequence,
                front_path=front_path,
                wrist_path=wrist_path,
                captured_timestamp_ns=captured,
                original_clock_domain=_first(payload, ("clock_domain", "timestamp_domain")),
                raw_timestamp=_first(payload, ("timestamp", "time")),
                measured_state=_first(
                    payload,
                    ("present", "measured", "observation.state", "state"),
                ),
                commanded_action=_first(payload, ("action", "sent", "goal", "command")),
            )
        )
    if not records:
        raise ReplayInputError(f"no front/wrist image pair found in {manifest}; use --front-key and --wrist-key")
    return tuple(records)


def _mime_type(path: Path, data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed in {"image/png", "image/jpeg"}:
        return guessed
    raise ReplayInputError(f"unsupported image encoding for {path}")


class RecordedReplayCameraSource:
    """Bounded replay source with explicit original-vs-delivery timing."""

    def __init__(self, records: Sequence[ReplayRecord]) -> None:
        if not records:
            raise ValueError("at least one replay record is required")
        self.records = tuple(records)
        self._index = 0
        self.capture_count = 0
        self._lock = threading.Lock()
        self._closed = False

    def capture(self) -> tuple[CameraFrame, ...]:
        with self._lock:
            if self._closed:
                raise ReplayInputError("recorded replay source is closed")
            record = self.records[self._index]
            self._index = (self._index + 1) % len(self.records)
            self.capture_count += 1
        delivered_ns = time.monotonic_ns()
        frames: list[CameraFrame] = []
        for name, path in ((FRONT_NAME, record.front_path), (WRIST_NAME, record.wrist_path)):
            data = path.read_bytes()
            frames.append(
                CameraFrame(
                    name=name,
                    mime_type=_mime_type(path, data),
                    data=data,
                    # This is the synthetic capture/delivery time of the
                    # virtual replay sensor.  The dataset capture time stays
                    # in profile metadata and is never presented as live.
                    captured_timestamp_ns=delivered_ns,
                    received_timestamp_ns=delivered_ns,
                    clock_domain="host_monotonic_ns",
                    profile={
                        "simulated": True,
                        "replay": True,
                        "hardware_access": False,
                        "source_kind": "recorded_file",
                        "record_line": record.line_number,
                        "record_sequence": record.sequence,
                        "original_capture_timestamp_ns": record.captured_timestamp_ns,
                        "original_timestamp": record.raw_timestamp,
                        "original_clock_domain": record.original_clock_domain,
                        "replay_delivery_timestamp_ns": delivered_ns,
                        "capture_semantics": "simulated_replay_delivery",
                        "units": MODEL_UNITS,
                    },
                )
            )
        return tuple(frames)

    def close(self) -> None:
        with self._lock:
            self._closed = True


def validate_model_units(value: str) -> str:
    """Require the explicit experiment label; never infer a physical unit."""

    if value != MODEL_UNITS:
        raise ExperimentError(
            f"model units must be explicitly {MODEL_UNITS!r}; no degree or normalized conversion is implemented"
        )
    return value


def _nested(value: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in value:
            return value[name]
    for key in ("adapter", "policy", "model", "capabilities"):
        child = value.get(key)
        if isinstance(child, Mapping):
            found = _nested(child, *names)
            if found is not None:
                return found
    return None


def validate_capabilities(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the policy-vector protocol shape, not physical meaning."""

    action_space = _nested(capabilities, "action_space")
    state_fields = _nested(capabilities, "state_fields")
    feature_names = _nested(capabilities, "action_feature_names", "feature_names")
    return_steps = _nested(capabilities, "return_steps", "horizon")
    if action_space != POLICY_ACTION_SPACE:
        raise ExperimentError(f"EmbodiInfer action_space {action_space!r} does not match {POLICY_ACTION_SPACE!r}")
    if tuple(state_fields or ()) != ("state_native",):
        raise ExperimentError(f"EmbodiInfer state_fields must be ['state_native'], got {state_fields!r}")
    if tuple(feature_names or ()) != POLICY_FEATURE_NAMES:
        raise ExperimentError("EmbodiInfer action_feature_names do not match the six SO feature names")
    if isinstance(return_steps, bool) or not isinstance(return_steps, int) or return_steps < PROPOSED_STEPS:
        raise ExperimentError(f"EmbodiInfer return_steps must support the {PROPOSED_STEPS}-step proposal")
    return {
        "action_space": action_space,
        "state_fields": list(state_fields),
        "action_feature_names": list(feature_names),
        "return_steps": return_steps,
        "units": MODEL_UNITS,
    }


def _policy_result_payload(result: PolicyResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "session_id": result.session_id,
        "step_id": result.step_id,
        "session_revision": result.session_revision,
        "action_space": result.action_space,
        "actions": [{"type": action.kind, "values": dict(action.values)} for action in result.actions],
        "timing": dict(result.timing),
        "policy_revision": result.policy_revision,
    }


def _result_prefix_vector(result: Mapping[str, Any]) -> tuple[float, ...]:
    actions = result.get("actions")
    if not isinstance(actions, Sequence) or len(actions) != 1:
        raise ExperimentError("raw EmbodiInfer result must contain one action chunk")
    action = actions[0]
    if not isinstance(action, Mapping):
        raise ExperimentError("raw EmbodiInfer action must be an object")
    values = action.get("values")
    if not isinstance(values, Mapping):
        raise ExperimentError("raw EmbodiInfer action values must be an object")
    names = values.get("feature_names")
    data = values.get("data")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or len(names) != len(POLICY_FEATURE_NAMES)
        or any(not isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(POLICY_FEATURE_NAMES)
    ):
        raise ExperimentError("raw EmbodiInfer feature_names do not match the six SO names")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ExperimentError("raw EmbodiInfer action data must be a sequence")
    if len(data) != PROPOSED_STEPS:
        raise ExperimentError(f"raw EmbodiInfer proposal must contain {PROPOSED_STEPS} rows, got {len(data)}")
    row = data[PREFIX_STEPS - 1]
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 6:
        raise ExperimentError("raw EmbodiInfer prefix row must contain six values")
    output: list[float] = []
    for index, item in enumerate(row):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ExperimentError(f"raw EmbodiInfer prefix value {index} is not finite")
        output.append(float(item))
    by_name = dict(zip(names, output))
    return tuple(by_name[name] for name in POLICY_FEATURE_NAMES)


def _observed_native_state(payload: Mapping[str, Any]) -> tuple[float, ...]:
    robot = payload.get("robot")
    values = robot.get("values") if isinstance(robot, Mapping) else None
    state = values.get("state_native") if isinstance(values, Mapping) else None
    if isinstance(state, (str, bytes, bytearray)) or not isinstance(state, Sequence) or len(state) != 6:
        raise ExperimentError("Control observe did not return six state_native values")
    result = tuple(float(item) for item in state)
    if any(not math.isfinite(item) for item in result):
        raise ExperimentError("Control observe returned a non-finite state_native value")
    return result


def _same_vector(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9) for a, b in zip(left, right)
    )


def _observation_identifier(payload: Mapping[str, Any]) -> str | None:
    """Read the shared ID from either current snapshot or legacy payloads."""

    for key in ("observation_id", "snapshot_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _observation_stale(payload: Mapping[str, Any]) -> bool | None:
    """Read freshness from the current nested HTTP observation status."""

    status = payload.get("observation_status")
    if isinstance(status, Mapping) and isinstance(status.get("stale"), bool):
        return bool(status["stale"])
    value = payload.get("stale")
    return value if isinstance(value, bool) else None


def _wait_for_readback(
    server: ControlHttpServer,
    previous: Mapping[str, Any],
    expected_state: Sequence[float],
    *,
    timeout_s: float = 3.0,
) -> tuple[int, dict[str, Any]]:
    """Poll HTTP observe until the producer publishes post-action state.

    The shared producer is intentionally asynchronous.  A fast simulated
    action can finish before its next periodic capture, so one immediate GET
    may still return the pre-action snapshot.  A new observation ID and the
    measured state are required together; the commanded prefix is never used
    as a readback substitute.
    """

    deadline = time.monotonic() + timeout_s
    last_status = 0
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_status, last_payload = _http_json(
            server,
            "GET",
            "/v1/observe?include_robot=true",
        )
        if (
            last_status == 200
            and _observation_identifier(last_payload) != _observation_identifier(previous)
            and _same_vector(_observed_native_state(last_payload), expected_state)
        ):
            return last_status, last_payload
        time.sleep(0.02)
    raise ExperimentError(
        "timed out waiting for a new measured readback: "
        f"expected={tuple(expected_state)!r} last_status={last_status} "
        f"last_payload={last_payload!r}"
    )


class InferenceTrace:
    """Capture bounded raw model responses and host call timings."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.sessions: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def _call(self, name: str, started_ns: int, **detail: Any) -> None:
        self.calls.append({"name": name, "elapsed_ns": max(0, time.monotonic_ns() - started_ns), **detail})

    def write_raw(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"endpoint": self.endpoint, "results": self.results},
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class TracedEmbodiInferClient:
    """Forward real EmbodiInfer calls while retaining their raw responses."""

    def __init__(self, endpoint: str, trace: InferenceTrace, token: str | None, timeout_s: float) -> None:
        self._inner = EmbodiInferHttpClient(endpoint, token=token, timeout_s=timeout_s)
        self.trace = trace

    def health(self) -> Mapping[str, Any]:
        started = time.monotonic_ns()
        value = self._inner.health()
        self.trace._call("health", started, status=value.get("status"))
        return value

    def capabilities(self) -> Mapping[str, Any]:
        started = time.monotonic_ns()
        value = self._inner.capabilities()
        self.trace._call("capabilities", started)
        return value

    def open_session(self, *, robot_id: str, action_space: str, metadata: Mapping[str, Any] | None = None) -> Session:
        started = time.monotonic_ns()
        session = self._inner.open_session(
            robot_id=robot_id,
            action_space=action_space,
            metadata=metadata,
        )
        self.trace.sessions.append(
            {"session_id": session.session_id, "revision": session.revision, "robot_id": robot_id}
        )
        self.trace._call("open_session", started, session_id=session.session_id)
        return session

    def step(self, observation: PolicyObservation) -> PolicyResult:
        started = time.monotonic_ns()
        result = self._inner.step(observation)
        payload = _policy_result_payload(result)
        payload["elapsed_ns"] = max(0, time.monotonic_ns() - started)
        self.trace.results.append(payload)
        self.trace._call("step", started, request_id=observation.request_id, step_id=observation.step_id)
        return result

    def reset(self, session_id: str, *, request_id: str) -> Session:
        started = time.monotonic_ns()
        value = self._inner.reset(session_id, request_id=request_id)
        self.trace._call("reset", started, session_id=session_id)
        return value

    def close(self, session_id: str) -> None:
        started = time.monotonic_ns()
        self._inner.close(session_id)
        self.trace._call("close", started, session_id=session_id)


def build_experiment_config(
    endpoint: str,
    *,
    port: int,
    initial_state_native: Sequence[float],
    token: str | None = None,
    robot_options: Mapping[str, Any] | None = None,
) -> ControlServiceConfig:
    """Build the explicit fake-only service config for the policy-vector binding."""

    if isinstance(initial_state_native, (str, bytes, bytearray)) or len(initial_state_native) != 6:
        raise ExperimentError("initial_state_native must contain six measured values")
    options = {"initial_state_native": [float(value) for value in initial_state_native]}
    if robot_options is not None:
        options.update(dict(robot_options))
    inference_options = {} if token is None else {"token": token}
    return ControlServiceConfig(
        runtime_id=RUNTIME_ID,
        binding_kind="simulated.policy_vector.pi05",
        bind="127.0.0.1",
        port=port,
        inference_transport="http",
        inference_endpoint=endpoint,
        inference_options=inference_options,
        robot_id=ROBOT_ID,
        robot_kind="simulated.policy_vector",
        robot_options=options,
        inputs=(
            SensorInput(
                sensor_id="recorded-camera-pair",
                name=FRONT_NAME,
                kind="recorded_replay",
                options={"simulated": True},
            ),
        ),
        runtime_options={},
        node_id="local-replay",
        device_resources=(
            {
                "identity": "local-replay:robot:policy-vector-replay",
                "node": "local-replay",
                "kind": "robot",
                "value": "policy-vector-replay",
            },
            {
                "identity": "local-replay:sensor:recorded-camera-pair",
                "node": "local-replay",
                "kind": "sensor",
                "value": "recorded-camera-pair",
                "sensor_id": "recorded-camera-pair",
            },
        ),
    )


@contextmanager
def running_experiment(
    config: ControlServiceConfig,
    records: Sequence[ReplayRecord],
    *,
    state_dir: Path,
    trace: InferenceTrace,
    token: str | None = None,
    timeout_s: float = 30.0,
    trace_path: Path | None = None,
) -> Iterator[tuple[ControlHttpServer, ControlService, RecordedReplayCameraSource]]:
    """Start only loopback Control and in-memory/replay resources."""

    state_dir.mkdir(parents=True, exist_ok=True)
    source = RecordedReplayCameraSource(records)
    client = TracedEmbodiInferClient(config.inference_endpoint, trace, token, timeout_s)
    manager = DeviceManager(
        config.node_id,
        owner_id=f"policy-vector-replay:{threading.get_ident()}",
        lock_dir=state_dir / "locks",
        state_path=state_dir / "device-state.json",
    )
    service = ControlService(
        config,
        camera_factory=lambda _inputs: source,
        client_factory=lambda _config, _timeout: client,
        device_manager=manager,
    )
    recorder = ObservationRecorder(
        service.observation_store,
        state_dir / "recordings",
        "policy-vector-replay",
        # SharedSensorHub namespaces raw source frames with their physical
        # resource identity.  The expected names are filled from the first
        # real snapshot after the service has attached that source, before
        # recording starts; using the runtime view names here would report
        # valid source bytes as missing.
        expected_frame_names=(),
        require_state=True,
    )
    service.attach_recorder(recorder)
    server = ControlHttpServer(service, state_dir=state_dir)
    thread = threading.Thread(target=server.serve_forever, name="policy-vector-control", daemon=True)
    thread.start()
    try:
        yield server, service, source
    finally:
        try:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
        finally:
            # Drain recording while the shared producer and robot state are
            # still alive.  Closing the service first can turn the final
            # in-flight snapshot into a false missing-state diagnostic.
            try:
                recorder.stop(timeout_s=1.0)
            finally:
                try:
                    service.close()
                finally:
                    if trace_path is not None:
                        # Preserve the original experiment/cleanup error;
                        # successful runs still fail loudly below if the
                        # required raw trace is absent.
                        with suppress(OSError):
                            trace.write_raw(Path(trace_path))


def _configure_recording_frame_names(service: ControlService) -> tuple[str, ...]:
    """Use the canonical names in the first shared snapshot for recording."""

    hub = getattr(service, "_shared_observations", None)
    latest = hub.latest() if hub is not None else None
    frames = tuple(getattr(latest, "cameras", ())) if latest is not None else ()
    names = tuple(frame.name for frame in frames)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ExperimentError(
            f"recording requires one unambiguous front/wrist source pair; shared snapshot names were {names!r}"
        )
    recorder = getattr(service, "_recorder", None)
    if not isinstance(recorder, ObservationRecorder):
        raise ExperimentError("shared-device experiment recorder is unavailable")
    recorder.expected_frame_names = frozenset(names)
    return names


def _validate_recording_artifacts(
    recording: Mapping[str, Any],
    expected_frame_names: Sequence[str],
) -> dict[str, Any]:
    """Require paired source media and state in the completed raw recording."""

    if recording.get("state") != "stopped":
        raise ExperimentError(f"recording did not stop cleanly: {recording!r}")
    if recording.get("incomplete") is True:
        raise ExperimentError(f"recording is incomplete: {recording!r}")
    if recording.get("missing_frames") != 0 or recording.get("missing_states") != 0:
        raise ExperimentError(f"recording has missing sources/state: {recording!r}")
    action_count = recording.get("action_count")
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count <= 0:
        raise ExperimentError(f"recording contains no action events: {recording!r}")
    root_value = recording.get("path")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ExperimentError("recording status has no output path")
    root = Path(root_value)
    observations_path = root / "observations.jsonl"
    if not observations_path.is_file():
        raise ExperimentError(f"recording observations are missing: {observations_path}")
    paired_ids: list[str] = []
    for line in observations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping) or value.get("kind") != "observation":
            continue
        cameras = value.get("cameras")
        if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
            continue
        media_by_name = {
            item.get("name"): item
            for item in cameras
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        if not set(expected_frame_names).issubset(media_by_name):
            continue
        for name in expected_frame_names:
            media = media_by_name[name]
            relative = media.get("path")
            if not isinstance(relative, str) or not relative:
                raise ExperimentError(f"recorded frame {name!r} has no media path")
            path = root / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise ExperimentError(f"recorded frame {name!r} has no media bytes")
        observation_id = value.get("observation_id")
        if isinstance(observation_id, str) and observation_id:
            paired_ids.append(observation_id)
    if not paired_ids:
        raise ExperimentError("recording contains no complete front/wrist observation")
    return {
        "paired_observations": len(paired_ids),
        "observation_ids": paired_ids,
        "frame_names": list(expected_frame_names),
    }


def _http_json(
    server: ControlHttpServer, method: str, path: str, body: Mapping[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    host, port = server.server_address[:2]
    # A first remote EmbodiInfer request may include model warm-up.  Keep this bounded
    # but longer than the per-request inference budget so the HTTP caller does
    # not abandon a still-valid task while its result is being serialized.
    connection = http.client.HTTPConnection(host, port, timeout=120.0)
    encoded = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    if not isinstance(payload, dict):
        raise ExperimentError(f"Control returned a non-object payload for {path}")
    return response.status, payload


def run_experiment(
    endpoint: str,
    input_path: str | Path,
    *,
    output_dir: str | Path,
    state_dir: str | Path,
    model_units: str,
    port: int = 18901,
    token: str | None = None,
    control_hz: float = 10.0,
    inference_timeout_s: float = 120.0,
    front_key: str | None = None,
    wrist_key: str | None = None,
    record_line: int | None = None,
) -> dict[str, Any]:
    validate_model_units(model_units)
    records = load_replay_records(
        input_path,
        front_key=front_key,
        wrist_key=wrist_key,
        record_line=record_line,
    )
    trace = InferenceTrace(endpoint)
    preflight_client = TracedEmbodiInferClient(endpoint, trace, token, inference_timeout_s)
    health = dict(preflight_client.health())
    capabilities = dict(preflight_client.capabilities())
    capability_summary = validate_capabilities(capabilities)
    if health.get("status") != "ok":
        raise ExperimentError(f"EmbodiInfer endpoint is not healthy: {health!r}")

    initial_state = native_state_from_record(records[0])
    config = build_experiment_config(
        endpoint,
        port=port,
        initial_state_native=initial_state,
        token=token,
    )
    state_dir_path = Path(state_dir).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir_path / "raw-model-results.json"
    tasks: list[dict[str, Any]] = []
    subscription_snapshots: list[str] = []
    owner_ids: list[int] = []
    camera_source_count = 0
    robot_connect_count: int | None = None
    expected_state = initial_state
    with running_experiment(
        config,
        records,
        state_dir=state_dir_path,
        trace=trace,
        token=token,
        timeout_s=inference_timeout_s,
        trace_path=raw_path,
    ) as (server, service, _source):
        subscription = service.subscribe(max_queue=8, replay_latest=True)
        try:
            status, before = _http_json(server, "GET", "/v1/observe?include_robot=true")
            if status != 200:
                raise ExperimentError(f"initial observe failed: {status} {before}")
            if not _same_vector(_observed_native_state(before), expected_state):
                raise ExperimentError("initial Control state does not match recorded present")
            recording_frame_names = _configure_recording_frame_names(service)
            status, recording = _http_json(server, "POST", "/v1/recordings/start", {})
            if status != 200:
                raise ExperimentError(f"recording start failed: {status} {recording}")
            for index in (1, 2):
                if not _same_vector(_observed_native_state(before), expected_state):
                    raise ExperimentError("task input state differs from prior policy readback")
                request_id = f"policy-vector-replay-{index}"
                task = {
                    "schema": "embodirun.control.task.v1",
                    "request_id": request_id,
                    "runtime_id": RUNTIME_ID,
                    "prompt": "pick up the blue cube and place it in the bowl",
                    "chunk_steps": PREFIX_STEPS,
                    "max_steps": 1,
                    "control_hz": control_hz,
                    "inference_timeout_s": inference_timeout_s,
                    "wait": True,
                }
                status, response = _http_json(server, "POST", "/v1/tasks", task)
                if status != 200 or response.get("schema") != "embodirun.control.result.v1":
                    raise ExperimentError(f"task {request_id} failed: {status} {response}")
                if response.get("completed_steps") != 1:
                    raise ExperimentError(f"task {request_id} did not complete one policy step")
                if len(trace.results) != index:
                    raise ExperimentError(
                        f"expected one raw model result for task {request_id}, got {len(trace.results)}"
                    )
                expected_state = _result_prefix_vector(trace.results[-1])
                status_after, after = _wait_for_readback(
                    server,
                    before,
                    expected_state,
                )
                robot = getattr(service, "_robot", None)
                if robot is not None:
                    owner_ids.append(id(robot))
                    calls = getattr(robot, "connect_calls", None)
                    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
                        robot_connect_count = len(calls)
                    elif isinstance(getattr(robot, "connect_count", None), int):
                        robot_connect_count = int(robot.connect_count)
                tasks.append(
                    {
                        "request_id": request_id,
                        "http_status": status,
                        "response": response,
                        "observation_before": _observation_identifier(before),
                        "observation_after": _observation_identifier(after),
                        "observation_after_http_status": status_after,
                        "stale_after": _observation_stale(after),
                    }
                )
                before = after
                while True:
                    try:
                        snapshot = subscription.get_nowait()
                    except (Empty, ObservationSubscriptionClosed):
                        break
                    subscription_snapshots.append(snapshot.observation_id)
            status, stopped = _http_json(server, "POST", "/v1/recordings/stop", {})
            if status != 200:
                raise ExperimentError(f"recording stop failed: {status} {stopped}")
            status, recording_status = _http_json(server, "GET", "/v1/recordings/status")
            if status != 200:
                raise ExperimentError(f"recording status failed: {status} {recording_status}")
            recording_validation = _validate_recording_artifacts(
                recording_status,
                recording_frame_names,
            )
            camera_source_count = len(getattr(service, "_camera_sources", {}))
        finally:
            subscription.close()

    if len(trace.results) != 2:
        raise ExperimentError(f"expected two raw model results, got {len(trace.results)}")
    session_ids = [item["session_id"] for item in trace.sessions]
    if len(session_ids) != 2 or len(set(session_ids)) != 2:
        raise ExperimentError(f"expected two distinct policy sessions, got {session_ids!r}")
    if len(owner_ids) != 2 or owner_ids[0] != owner_ids[1]:
        raise ExperimentError("the two tasks did not reuse one robot owner")
    if camera_source_count != 1:
        raise ExperimentError(f"expected one shared camera source, got {camera_source_count}")
    if not subscription_snapshots:
        raise ExperimentError("no shared observation reached the consumer subscription")

    report = {
        "schema": "embodirun.example.shared_device_inference.v1",
        "status": "software_complete",
        "software_only": True,
        "physical_robot": False,
        "hardware_access": False,
        "execution_semantics": "simulated_policy_vector_only",
        "model_units": model_units,
        "units_claim": "dataset-native values preserved; physical units unknown",
        "proposal_steps": PROPOSED_STEPS,
        "executed_prefix_steps": PREFIX_STEPS,
        "capabilities": capability_summary,
        "input": {
            "manifest": str(_resolve_manifest(input_path)),
            "replay": True,
            "records": len(records),
            "selected": [record.to_report() for record in records[:3]],
            "remaining_record_count": max(0, len(records) - 3),
        },
        "tasks": tasks,
        "shared_consumer_observation_ids": subscription_snapshots,
        "owners": {
            "robot_owner_reused": len(owner_ids) == 2 and owner_ids[0] == owner_ids[1],
            "camera_source_count": camera_source_count,
            "robot_connect_count": robot_connect_count,
        },
        "recording": recording_status,
        "recording_validation": recording_validation,
        "model_trace": {
            "sessions": trace.sessions,
            "step_count": len(trace.results),
            "calls": trace.calls,
            "raw_results_path": str(raw_path),
        },
        "physical_success": None,
        "scene_live": False,
    }
    (output_dir_path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_shared_device_inference")
    parser.add_argument("--endpoint", required=True, help="EmbodiInfer HTTP endpoint")
    parser.add_argument("--input", required=True, help="frames.jsonl or its recording directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--model-units", required=True, choices=(MODEL_UNITS,))
    parser.add_argument("--token")
    parser.add_argument("--port", type=int, default=18901)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--inference-timeout-s", type=float, default=120.0)
    parser.add_argument("--front-key")
    parser.add_argument("--wrist-key")
    parser.add_argument("--record-line", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_model_units(args.model_units)
        records = load_replay_records(
            args.input,
            front_key=args.front_key,
            wrist_key=args.wrist_key,
            record_line=args.record_line,
        )
        trace = InferenceTrace(args.endpoint)
        client = TracedEmbodiInferClient(args.endpoint, trace, args.token, args.inference_timeout_s)
        health = dict(client.health())
        capabilities = dict(client.capabilities())
        summary = validate_capabilities(capabilities)
        if health.get("status") != "ok":
            raise ExperimentError(f"EmbodiInfer endpoint is not healthy: {health!r}")
        if args.preflight_only:
            report = {
                "schema": "embodirun.example.shared_device_inference.v1",
                "status": "preflight_only",
                "software_only": True,
                "physical_robot": False,
                "model_units": args.model_units,
                "capabilities": summary,
                "input": {
                    "manifest": str(_resolve_manifest(args.input)),
                    "replay": True,
                    "records": len(records),
                    "selected": [record.to_report() for record in records[:3]],
                },
                "model_trace": {"calls": trace.calls},
                "physical_success": None,
            }
        else:
            report = run_experiment(
                args.endpoint,
                args.input,
                output_dir=args.output_dir,
                state_dir=args.state_dir,
                model_units=args.model_units,
                port=args.port,
                token=args.token,
                control_hz=args.control_hz,
                inference_timeout_s=args.inference_timeout_s,
                front_key=args.front_key,
                wrist_key=args.wrist_key,
                record_line=args.record_line,
            )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0
    except (ReplayInputError, ExperimentError, OSError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
