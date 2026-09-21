"""Versioned configuration and task messages exchanged by Host and Control."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from embodirun.model_services.providers import provider
from embodirun.robots.sensors import SensorInput

CONTROL_CONFIG_SCHEMA = "rlinf.control.config.v1"
TASK_REQUEST_SCHEMA = "rlinf.control.task.v1"
TASK_RESULT_SCHEMA = "rlinf.control.result.v1"
ERROR_SCHEMA = "rlinf.control.error.v1"


class ControlContractError(ValueError):
    """A Host-to-Control message is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class ControlRuntimeProfile:
    """The model/binding side of one runtime sharing a robot owner."""

    runtime_id: str
    binding_kind: str
    inference_transport: str
    inference_endpoint: str
    inference_options: Mapping[str, Any]
    inputs: tuple[SensorInput, ...]
    runtime_options: Mapping[str, Any]
    inference_backend: str = "vvla"

    def __post_init__(self) -> None:
        for name in (
            "runtime_id",
            "binding_kind",
            "inference_transport",
            "inference_endpoint",
            "inference_backend",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ControlContractError(f"{name} must not be empty")
        try:
            selected_provider = provider(self.inference_backend)
        except ValueError as error:
            raise ControlContractError(str(error)) from error
        if not selected_provider.supports(self.inference_transport):
            raise ControlContractError(
                f"inference provider {self.inference_backend!r} does not support transport {self.inference_transport!r}"
            )
        if not selected_provider.action_capable:
            raise ControlContractError(f"inference provider {self.inference_backend!r} has no action capability")
        object.__setattr__(
            self,
            "inference_options",
            _mapping(self.inference_options, "profile.inference_options"),
        )
        object.__setattr__(
            self,
            "runtime_options",
            _mapping(self.runtime_options, "profile.runtime_options"),
        )
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ControlContractError("control runtime profile inputs must be non-empty")
        if any(not isinstance(item, SensorInput) for item in self.inputs):
            raise ControlContractError("control runtime profile inputs are invalid")
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ControlContractError("control runtime profile input names must be unique")

    def to_payload(self) -> dict[str, Any]:
        return {
            "binding": self.binding_kind,
            "inference": {
                "backend": self.inference_backend,
                "transport": self.inference_transport,
                "endpoint": self.inference_endpoint,
                "options": dict(self.inference_options),
            },
            "inputs": [
                {
                    "sensor_id": item.sensor_id,
                    "name": item.name,
                    "type": item.kind,
                    "options": dict(item.options),
                }
                for item in self.inputs
            ],
            "runtime_options": dict(self.runtime_options),
        }

    @classmethod
    def from_payload(cls, runtime_id: str, value: object) -> ControlRuntimeProfile:
        item = _mapping(value, f"control config.runtime_profiles.{runtime_id}")
        context = f"control config.runtime_profiles.{runtime_id}"
        _reject_unknown(item, {"binding", "inference", "inputs", "runtime_options"}, context)
        inference = _mapping(item.get("inference"), f"{context}.inference")
        _reject_unknown(
            inference,
            {"backend", "transport", "endpoint", "options"},
            f"{context}.inference",
        )
        inputs = item.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ControlContractError(f"{context}.inputs must be a non-empty list")
        return cls(
            runtime_id=runtime_id,
            binding_kind=_string(item, "binding", context),
            inference_backend=(
                _string(inference, "backend", f"{context}.inference") if "backend" in inference else "vvla"
            ),
            inference_transport=_string(inference, "transport", f"{context}.inference"),
            inference_endpoint=_string(inference, "endpoint", f"{context}.inference"),
            inference_options=_mapping(inference.get("options", {}), f"{context}.inference.options"),
            inputs=tuple(_sensor_input(input_value, index) for index, input_value in enumerate(inputs)),
            runtime_options=_mapping(item.get("runtime_options", {}), f"{context}.runtime_options"),
        )


@dataclass(frozen=True, slots=True)
class ControlServiceConfig:
    """Static Host-supplied configuration loaded once when Control starts."""

    runtime_id: str
    binding_kind: str
    bind: str
    port: int
    inference_transport: str
    inference_endpoint: str
    inference_options: Mapping[str, Any]
    robot_id: str
    robot_kind: str
    robot_options: Mapping[str, Any]
    inputs: tuple[SensorInput, ...]
    runtime_options: Mapping[str, Any]
    inference_backend: str = "vvla"
    # Optional extensions are emitted only when a Host plan needs them, which
    # keeps legacy single-runtime YAML and generated config byte-compatible in
    # shape.  ``runtime_profiles`` is the per-runtime binding map for a shared
    # robot owner.
    node_id: str = "local"
    device_resources: tuple[Mapping[str, Any], ...] = ()
    runtime_profiles: Mapping[str, ControlRuntimeProfile] = field(default_factory=dict)
    inference_enabled: bool = True

    def __post_init__(self) -> None:
        required_names = (
            "runtime_id",
            "binding_kind",
            "bind",
            "robot_id",
            "robot_kind",
            "node_id",
        )
        if self.inference_enabled:
            required_names += (
                "inference_backend",
                "inference_transport",
                "inference_endpoint",
            )
        for name in required_names:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ControlContractError(f"{name} must not be empty")
        if self.bind not in {"127.0.0.1", "localhost", "::1"}:
            raise ControlContractError("control service must bind to a loopback address")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ControlContractError("control service port must be between 1 and 65535")
        if self.inference_enabled and self.inference_transport not in {
            "http",
            "wireless",
        }:
            raise ControlContractError("inference transport must be http or wireless")
        if not self.inference_enabled and self.inference_transport != "disabled":
            raise ControlContractError("a device-only control service must use disabled inference")
        if self.inference_enabled:
            try:
                selected_provider = provider(self.inference_backend)
            except ValueError as error:
                raise ControlContractError(str(error)) from error
            if not selected_provider.supports(self.inference_transport):
                raise ControlContractError(
                    f"inference provider {self.inference_backend!r} does not support "
                    f"transport {self.inference_transport!r}"
                )
            if not selected_provider.action_capable:
                raise ControlContractError(f"inference provider {self.inference_backend!r} has no action capability")
        for name in ("inference_options", "robot_options", "runtime_options"):
            value = _mapping(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if not isinstance(self.device_resources, tuple):
            raise ControlContractError("device_resources must be a tuple")
        for index, resource in enumerate(self.device_resources):
            item = _mapping(resource, f"device_resources[{index}]")
            if not isinstance(item.get("identity"), str) or not item["identity"].strip():
                raise ControlContractError(f"device_resources[{index}].identity must not be empty")
            if not isinstance(item.get("kind"), str) or not item["kind"].strip():
                raise ControlContractError(f"device_resources[{index}].kind must not be empty")
        if not isinstance(self.inference_enabled, bool):
            raise ControlContractError("inference_enabled must be a boolean")
        profiles = _mapping(self.runtime_profiles, "runtime_profiles")
        if any(not isinstance(key, str) or not key.strip() for key in profiles):
            raise ControlContractError("runtime_profiles keys must not be empty")
        if any(not isinstance(value, ControlRuntimeProfile) for value in profiles.values()):
            raise ControlContractError("runtime_profiles values must be ControlRuntimeProfile")
        if any(key != value.runtime_id for key, value in profiles.items()):
            raise ControlContractError("runtime_profiles keys must match runtime_id")
        object.__setattr__(self, "runtime_profiles", profiles)
        if not isinstance(self.inputs, tuple) or (self.inference_enabled and not self.inputs):
            raise ControlContractError("control service inputs must be a non-empty tuple when inference is enabled")
        if any(not isinstance(item, SensorInput) for item in self.inputs):
            raise ControlContractError("control service inputs must contain SensorInput values")
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ControlContractError("control service input names must be unique")

    @classmethod
    def device_only(
        cls,
        *,
        runtime_id: str,
        bind: str,
        port: int,
        robot_id: str,
        robot_kind: str,
        robot_options: Mapping[str, Any],
        node_id: str = "local",
        inputs: tuple[SensorInput, ...] = (),
        device_resources: tuple[Mapping[str, Any], ...] = (),
    ) -> ControlServiceConfig:
        """Build an explicit robot service with no model or inference peer.

        This mode is for lifecycle, describe, observe, and later explicit
        control attachment.  It has no placeholder model ID or endpoint, and
        task execution remains rejected until a model runtime is configured.
        """

        return cls(
            runtime_id=runtime_id,
            binding_kind="device-only",
            bind=bind,
            port=port,
            inference_backend="vvla",
            inference_transport="disabled",
            inference_endpoint="",
            inference_options={},
            robot_id=robot_id,
            robot_kind=robot_kind,
            robot_options=robot_options,
            inputs=inputs,
            runtime_options={},
            node_id=node_id,
            device_resources=device_resources,
            inference_enabled=False,
        )

    def to_json(self) -> str:
        """Serialize the private generated configuration written during ``up``."""

        inference = (
            {"enabled": False}
            if not self.inference_enabled
            else {
                "backend": self.inference_backend,
                "transport": self.inference_transport,
                "endpoint": self.inference_endpoint,
                "options": dict(self.inference_options),
            }
        )
        payload = {
            "schema": CONTROL_CONFIG_SCHEMA,
            "runtime_id": self.runtime_id,
            "binding": self.binding_kind,
            "server": {"bind": self.bind, "port": self.port},
            "inference": inference,
            "robot": {
                "id": self.robot_id,
                "type": self.robot_kind,
                "options": dict(self.robot_options),
            },
            "inputs": [
                {
                    "sensor_id": item.sensor_id,
                    "name": item.name,
                    "type": item.kind,
                    "options": dict(item.options),
                }
                for item in self.inputs
            ],
            "runtime_options": dict(self.runtime_options),
        }
        if self.node_id != "local":
            payload["node_id"] = self.node_id
        if self.device_resources:
            payload["device_resources"] = [dict(item) for item in self.device_resources]
        if self.runtime_profiles:
            payload["runtime_profiles"] = {
                runtime_id: profile.to_payload() for runtime_id, profile in sorted(self.runtime_profiles.items())
            }
        if not self.inference_enabled:
            payload["inference"]["enabled"] = False
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> ControlServiceConfig:
        """Parse a generated configuration without consulting Host YAML."""

        try:
            root = _mapping(json.loads(value), "control config")
        except json.JSONDecodeError as error:
            raise ControlContractError(f"control config is invalid JSON: {error}") from error
        if root.get("schema") != CONTROL_CONFIG_SCHEMA:
            raise ControlContractError("unsupported control config schema")
        _reject_unknown(
            root,
            {
                "schema",
                "runtime_id",
                "binding",
                "server",
                "inference",
                "robot",
                "inputs",
                "runtime_options",
                "node_id",
                "device_resources",
                "runtime_profiles",
            },
            "control config",
        )
        server = _mapping(root.get("server"), "control config.server")
        inference = _mapping(root.get("inference"), "control config.inference")
        robot = _mapping(root.get("robot"), "control config.robot")
        _reject_unknown(server, {"bind", "port"}, "control config.server")
        _reject_unknown(
            inference,
            {"backend", "transport", "endpoint", "options", "enabled"},
            "control config.inference",
        )
        _reject_unknown(robot, {"id", "type", "options"}, "control config.robot")
        inference_enabled = inference.get("enabled", True)
        if not isinstance(inference_enabled, bool):
            raise ControlContractError("control config.inference.enabled must be boolean")
        inputs = root.get("inputs")
        if not isinstance(inputs, list) or (inference_enabled and not inputs):
            raise ControlContractError("control config.inputs must be a non-empty list when inference is enabled")
        resources = root.get("device_resources", [])
        if not isinstance(resources, list):
            raise ControlContractError("control config.device_resources must be a list")
        profiles_value = root.get("runtime_profiles", {})
        profiles_mapping = _mapping(profiles_value, "control config.runtime_profiles")
        profiles = {
            runtime_id: ControlRuntimeProfile.from_payload(runtime_id, item)
            for runtime_id, item in profiles_mapping.items()
        }
        if inference_enabled:
            inference_backend = (
                _string(inference, "backend", "control config.inference") if "backend" in inference else "vvla"
            )
            inference_transport = _string(inference, "transport", "control config.inference")
            inference_endpoint = _string(inference, "endpoint", "control config.inference")
        else:
            inference_backend = (
                _string(inference, "backend", "control config.inference") if "backend" in inference else "vvla"
            )
            inference_transport = "disabled"
            inference_endpoint = ""
        return cls(
            runtime_id=_string(root, "runtime_id", "control config"),
            binding_kind=_string(root, "binding", "control config"),
            bind=_string(server, "bind", "control config.server"),
            port=server.get("port"),
            inference_backend=inference_backend,
            inference_transport=inference_transport,
            inference_endpoint=inference_endpoint,
            inference_options=_mapping(inference.get("options", {}), "control config.inference.options"),
            robot_id=_string(robot, "id", "control config.robot"),
            robot_kind=_string(robot, "type", "control config.robot"),
            robot_options=_mapping(robot.get("options", {}), "control config.robot.options"),
            inputs=tuple(_sensor_input(item, index) for index, item in enumerate(inputs)),
            runtime_options=_mapping(root.get("runtime_options", {}), "control config.runtime_options"),
            node_id=(_string(root, "node_id", "control config") if "node_id" in root else "local"),
            device_resources=tuple(
                _mapping(item, f"control config.device_resources[{index}]") for index, item in enumerate(resources)
            ),
            runtime_profiles=profiles,
            inference_enabled=inference_enabled,
        )

    def profile_for_runtime(self, runtime_id: str) -> ControlRuntimeProfile:
        """Return the selected runtime profile without opening inference/robot IO."""

        if not self.inference_enabled:
            raise ControlContractError("device-only control service has no runtime profile")
        profile = self.runtime_profiles.get(runtime_id)
        if profile is not None:
            return profile
        if runtime_id != self.runtime_id:
            raise ControlContractError(f"runtime {runtime_id!r} is not configured")
        return ControlRuntimeProfile(
            runtime_id=self.runtime_id,
            binding_kind=self.binding_kind,
            inference_backend=self.inference_backend,
            inference_transport=self.inference_transport,
            inference_endpoint=self.inference_endpoint,
            inference_options=self.inference_options,
            inputs=self.inputs,
            runtime_options=self.runtime_options,
        )

    def for_profile(self, profile: ControlRuntimeProfile) -> ControlServiceConfig:
        """Build a legacy-shaped view consumed by existing client factories."""

        return replace(
            self,
            runtime_id=profile.runtime_id,
            binding_kind=profile.binding_kind,
            inference_backend=profile.inference_backend,
            inference_transport=profile.inference_transport,
            inference_endpoint=profile.inference_endpoint,
            inference_options=profile.inference_options,
            inputs=profile.inputs,
            runtime_options=profile.runtime_options,
        )


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """Dynamic task parameters sent by the host for one configured runtime."""

    request_id: str
    runtime_id: str
    prompt: str
    chunk_steps: int
    max_steps: int
    control_hz: float
    inference_timeout_s: float

    def __post_init__(self) -> None:
        for name in ("request_id", "runtime_id", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ControlContractError(f"{name} must not be empty")
        for name in ("chunk_steps", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ControlContractError(f"{name} must be a positive integer")
        for name in ("control_hz", "inference_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ControlContractError(f"{name} must be a finite positive number")

    def to_payload(self) -> dict[str, Any]:
        """Return the transport-neutral JSON object sent to Control."""

        return {
            "schema": TASK_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "runtime_id": self.runtime_id,
            "prompt": self.prompt,
            "chunk_steps": self.chunk_steps,
            "max_steps": self.max_steps,
            "control_hz": self.control_hz,
            "inference_timeout_s": self.inference_timeout_s,
        }

    @classmethod
    def from_payload(cls, value: object) -> TaskRequest:
        """Parse one versioned task request received by Control."""

        payload = _mapping(value, "task request")
        if payload.get("schema") != TASK_REQUEST_SCHEMA:
            raise ControlContractError("unsupported task request schema")
        _reject_unknown(
            payload,
            {
                "schema",
                "request_id",
                "runtime_id",
                "prompt",
                "chunk_steps",
                "max_steps",
                "control_hz",
                "inference_timeout_s",
            },
            "task request",
        )
        return cls(
            request_id=_string(payload, "request_id", "task request"),
            runtime_id=_string(payload, "runtime_id", "task request"),
            prompt=_string(payload, "prompt", "task request"),
            chunk_steps=payload.get("chunk_steps"),
            max_steps=payload.get("max_steps"),
            control_hz=payload.get("control_hz"),
            inference_timeout_s=payload.get("inference_timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Successful completion returned after Control releases robot resources."""

    request_id: str
    runtime_id: str
    completed_steps: int

    def __post_init__(self) -> None:
        for name in ("request_id", "runtime_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ControlContractError(f"{name} must not be empty")
        if (
            isinstance(self.completed_steps, bool)
            or not isinstance(self.completed_steps, int)
            or self.completed_steps < 0
        ):
            raise ControlContractError("completed_steps must be a non-negative integer")

    def to_payload(self) -> dict[str, Any]:
        """Return the versioned success response sent back to Host."""

        return {
            "schema": TASK_RESULT_SCHEMA,
            "request_id": self.request_id,
            "runtime_id": self.runtime_id,
            "completed_steps": self.completed_steps,
        }

    @classmethod
    def from_payload(cls, value: object) -> TaskResult:
        """Parse one successful control response."""

        payload = _mapping(value, "task result")
        if payload.get("schema") != TASK_RESULT_SCHEMA:
            raise ControlContractError("unsupported task result schema")
        _reject_unknown(
            payload,
            {"schema", "request_id", "runtime_id", "completed_steps"},
            "task result",
        )
        return cls(
            request_id=_string(payload, "request_id", "task result"),
            runtime_id=_string(payload, "runtime_id", "task result"),
            completed_steps=payload.get("completed_steps"),
        )


def error_payload(message: str) -> dict[str, str]:
    """Build the stable error envelope returned by the control server."""

    if not isinstance(message, str) or not message.strip():
        raise ControlContractError("error message must not be empty")
    return {"schema": ERROR_SCHEMA, "error": message}


def error_message(value: object) -> str | None:
    """Read a control error without accepting unrelated JSON as an error."""

    if not isinstance(value, Mapping) or value.get("schema") != ERROR_SCHEMA:
        return None
    message = value.get("error")
    if not isinstance(message, str) or not message.strip():
        return None
    return message


def _sensor_input(value: object, index: int) -> SensorInput:
    context = f"control config.inputs[{index}]"
    item = _mapping(value, context)
    _reject_unknown(item, {"sensor_id", "name", "type", "options"}, context)
    try:
        return SensorInput(
            sensor_id=_string(item, "sensor_id", context),
            name=_string(item, "name", context),
            kind=_string(item, "type", context),
            options=_mapping(item.get("options", {}), f"{context}.options"),
        )
    except ValueError as error:
        raise ControlContractError(f"{context} is invalid: {error}") from error


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ControlContractError(f"{context} must be a JSON object")
    return dict(value)


def _string(value: Mapping[str, object], name: str, context: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ControlContractError(f"{context}.{name} must be a non-empty string")
    return result


def _reject_unknown(
    value: Mapping[str, object],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ControlContractError(f"{context} contains unknown fields: {', '.join(unknown)}")


__all__ = [
    "CONTROL_CONFIG_SCHEMA",
    "ControlContractError",
    "ControlServiceConfig",
    "ERROR_SCHEMA",
    "TASK_REQUEST_SCHEMA",
    "TASK_RESULT_SCHEMA",
    "TaskRequest",
    "TaskResult",
    "error_message",
    "error_payload",
]
