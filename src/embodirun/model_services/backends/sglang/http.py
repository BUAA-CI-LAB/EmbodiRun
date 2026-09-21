"""SGLang inference service integration over its native HTTP action API."""

from __future__ import annotations

import base64
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any

from ...contracts import (
    PolicyAction,
    PolicyObservation,
    PolicyResult,
    Session,
)
from ...protocols.http import (
    MAX_RESPONSE_BYTES,
    HttpTransport,
    HttpTransportError,
    UrllibHttpTransport,
)


class SglangHttpError(RuntimeError):
    """An SGLang action request or response violated the client contract."""


@dataclass(slots=True)
class _SessionState:
    action_space: str
    robot_id: str
    metadata: dict[str, Any]
    revision: int = 0
    next_step: int = 0
    reset_pending: bool = True
    lock: Lock = field(default_factory=Lock)


class SglangHttpClient:
    """Expose SGLang's stateless action API as a Deploy inference client.

    SGLang owns model execution.  Deploy owns the lightweight session and step
    bookkeeping needed by robot and simulator runtimes.  Session metadata is
    included in each observation so stateful SGLang policy pipelines can keep
    their recurrent state at the model boundary.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 5.0,
        transport: HttpTransport | None = None,
        image_keys: Mapping[str, str] | None = None,
        state_fields: Sequence[str] = (),
        action_feature_names: Sequence[str] = (),
        output_action_dim: int | None = None,
        parameters: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if token is not None and not isinstance(token, str):
            raise TypeError("token must be a string or None")
        if output_action_dim is not None and (
            isinstance(output_action_dim, bool) or not isinstance(output_action_dim, int) or output_action_dim <= 0
        ):
            raise ValueError("output_action_dim must be a positive integer")

        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = float(timeout_s)
        self.transport = transport or UrllibHttpTransport()
        self.image_keys = _string_mapping(image_keys or {}, "image_keys")
        self.state_fields = _string_sequence(state_fields, "state_fields")
        self.action_feature_names = _string_sequence(
            action_feature_names,
            "action_feature_names",
        )
        if (
            output_action_dim is not None
            and self.action_feature_names
            and output_action_dim != len(self.action_feature_names)
        ):
            raise ValueError("output_action_dim must match action_feature_names when both are set")
        self.output_action_dim = output_action_dim or (
            len(self.action_feature_names) if self.action_feature_names else None
        )
        self.parameters = _mapping(parameters or {}, "parameters")
        self.runtime = _mapping(runtime or {}, "runtime")
        self._sessions: dict[str, _SessionState] = {}
        self._sessions_lock = RLock()

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/health", None)
        if not response:
            return {"status": "ok"}
        payload = _json_object(response, "GET /health")
        payload.setdefault("status", "ok")
        return payload

    def capabilities(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/actions/metadata")

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise ValueError("robot_id must be a non-empty string")
        if not isinstance(action_space, str) or not action_space.strip():
            raise ValueError("action_space must be a non-empty string")
        state = _SessionState(
            action_space=action_space,
            robot_id=robot_id,
            metadata=_mapping(metadata or {}, "session metadata"),
        )
        with self._sessions_lock:
            while True:
                session_id = f"sglang-{uuid.uuid4().hex}"
                if session_id not in self._sessions:
                    self._sessions[session_id] = state
                    break
        return Session(session_id, state.revision)

    def step(self, observation: PolicyObservation) -> PolicyResult:
        state = self._session(observation.session_id)
        with state.lock:
            if observation.step_id != state.next_step:
                raise SglangHttpError(
                    f"session {observation.session_id!r} expected step {state.next_step}, got {observation.step_id}"
                )
            reset = state.reset_pending or observation.reset
            payload = self._request_json(
                "POST",
                "/v1/actions/generations",
                self._step_payload(observation, state, reset=reset),
                extra_headers={"Idempotency-Key": observation.request_id},
            )
            result = self._policy_result(payload, observation, state)
            state.next_step += 1
            state.reset_pending = False
            return result

    def reset(self, session_id: str, *, request_id: str) -> Session:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        state = self._session(session_id)
        with state.lock:
            state.revision += 1
            state.next_step = 0
            state.reset_pending = True
            return Session(session_id, state.revision)

    def close(self, session_id: str) -> None:
        with self._sessions_lock:
            if self._sessions.pop(session_id, None) is None:
                raise SglangHttpError(f"unknown SGLang session {session_id!r}")

    def _session(self, session_id: str) -> _SessionState:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SglangHttpError(f"unknown SGLang session {session_id!r}")
        return state

    def _step_payload(
        self,
        observation: PolicyObservation,
        state: _SessionState,
        *,
        reset: bool,
    ) -> dict[str, Any]:
        images = {
            self.image_keys.get(image.name, _short_image_name(image.name)): {
                "base64": base64.b64encode(image.data).decode("ascii"),
                "mime_type": image.mime_type,
            }
            for image in observation.images
        }
        if len(images) != len(observation.images):
            raise SglangHttpError("multiple observation images map to the same key")
        backend_observation = {
            "images": images,
            "state": self._state(observation.state),
            "session_id": observation.session_id,
            "session_revision": state.revision,
            "step_id": observation.step_id,
            "reset": reset,
            "robot_id": state.robot_id,
            "metadata": {
                **state.metadata,
                **dict(observation.metadata),
            },
        }
        runtime = {
            **self.runtime,
            "output_format": "list",
            "response_format": "envelope",
        }
        runtime.setdefault("return_timing", True)
        return {
            "request_id": observation.request_id,
            "input": {
                "task": observation.instruction,
                "observation": backend_observation,
            },
            "parameters": dict(self.parameters),
            "runtime": runtime,
        }

    def _state(self, values: Mapping[str, Any]) -> Any:
        if not self.state_fields:
            return dict(values)
        result: list[float] = []
        for name in self.state_fields:
            if name not in values:
                raise SglangHttpError(f"policy state is missing field {name!r}")
            _flatten_numbers(values[name], name, result)
        return result

    def _policy_result(
        self,
        payload: Mapping[str, Any],
        observation: PolicyObservation,
        state: _SessionState,
    ) -> PolicyResult:
        response_id = payload.get("id")
        if response_id is not None and response_id != observation.request_id:
            raise SglangHttpError("action response id does not match the request")
        data = payload.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise SglangHttpError("action response data must be a list")
        if len(data) != 1 or not isinstance(data[0], Mapping):
            raise SglangHttpError("action response must contain exactly one result")
        action = data[0].get("action")
        if not isinstance(action, Mapping):
            raise SglangHttpError("action response result must contain an action")
        values = action.get("values")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise SglangHttpError("action values must be a list")
        rows = _action_rows(values, self.output_action_dim)
        action_values: dict[str, Any] = {"data": rows}
        if self.action_feature_names:
            action_values["feature_names"] = list(self.action_feature_names)
        return PolicyResult(
            request_id=observation.request_id,
            session_id=observation.session_id,
            step_id=observation.step_id,
            session_revision=state.revision,
            action_space=state.action_space,
            actions=(PolicyAction("action_chunk", action_values),),
            timing=_timing(payload.get("timings", {})),
            policy_revision=(str(payload["model"]) if payload.get("model") is not None else None),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        body = None
        if payload is not None:
            try:
                body = json.dumps(
                    dict(payload),
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise SglangHttpError("request payload is not valid JSON") from error
        response = self._request(
            method,
            path,
            body,
            extra_headers=extra_headers,
        )
        return _json_object(response, f"{method} {path}")

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> bytes:
        headers = {"Accept": "application/json", **dict(extra_headers or {})}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self.transport.request(
                method,
                self.base_url + path,
                headers=headers,
                body=body,
                timeout_s=self.timeout_s,
                maximum_bytes=MAX_RESPONSE_BYTES,
            )
        except HttpTransportError as error:
            raise SglangHttpError(str(error)) from error
        return response.body


def _json_object(value: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SglangHttpError(f"{context} returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise SglangHttpError(f"{context} must return a JSON object")
    return payload


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _string_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError(f"{name} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{name} values must be non-empty strings")
        result[key] = item
    if len(result.values()) != len(set(result.values())):
        raise ValueError(f"{name} values must be unique")
    return result


def _string_sequence(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a list of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _short_image_name(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def _flatten_numbers(value: object, name: str, output: list[float]) -> None:
    if isinstance(value, Mapping):
        for child_name, child in value.items():
            _flatten_numbers(child, f"{name}.{child_name}", output)
        return
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SglangHttpError(f"policy state field {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise SglangHttpError(f"policy state field {name!r} must be finite")
        output.append(number)
        return
    for index, item in enumerate(value):
        _flatten_numbers(item, f"{name}[{index}]", output)


def _action_rows(value: Sequence[Any], dimension: int | None) -> list[Any]:
    rows: list[Any] = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise SglangHttpError(f"action row {index} must be a list")
        result = list(row)
        if dimension is not None:
            if len(result) < dimension:
                raise SglangHttpError(f"action row {index} has {len(result)} values, expected at least {dimension}")
            result = result[:dimension]
        rows.append(result)
    if not rows:
        raise SglangHttpError("action values must not be empty")
    return rows


def _timing(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise SglangHttpError("action response timings must be an object")
    result: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        number = float(raw)
        if math.isfinite(number) and number >= 0:
            result[str(name)] = number
    return result


def sglang_server_command(
    *,
    checkpoint: str,
    bind: str,
    port: int,
    executable: str = "sglang",
    pipeline: str | None = None,
    pipeline_config: str | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build the documented ``sglang serve`` VLA command-line contract."""

    argv = [
        executable,
        "serve",
        checkpoint,
        "--model-type",
        "diffusion",
    ]
    if pipeline is not None:
        argv.extend(("--pipeline-class-name", pipeline))
    if pipeline_config is not None:
        argv.extend(("--pipeline-config-path", pipeline_config))
    argv.extend(extra_args)
    argv.extend(("--host", bind, "--port", str(port)))
    return tuple(argv)


__all__ = ["SglangHttpClient", "SglangHttpError", "sglang_server_command"]
