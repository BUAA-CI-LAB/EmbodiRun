"""Embodiment-to-model runtime binding configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .server import ServerConfig, parse_server
from .transport import InferenceClientConfig, parse_inference_client
from .validation import ConfigError, mapping, optional_string, string


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runtime_id: str
    robot: str | None
    simulator: str | None
    # Model and binding are a pair.  Omitting both selects a robot-only
    # lifecycle service; simulators still require an inference runtime.
    model: str | None
    binding: str | None
    inputs: dict[str, str]
    server: ServerConfig
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    inference_client: InferenceClientConfig | None = None


def parse_runtime(runtime_id: str, value: dict[str, Any]) -> RuntimeConfig:
    context = f"runtimes.{runtime_id}"
    server = parse_server(value.get("server"), f"{context}.server")
    if server.bind not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError(f"{context}.server.bind must be a loopback address because Host connects through SSH")
    inputs_value = mapping(value.get("inputs", {}), f"{context}.inputs")
    inputs: dict[str, str] = {}
    for input_name, sensor_id in inputs_value.items():
        if not input_name.strip():
            raise ConfigError(f"{context}.inputs keys must not be empty")
        if not isinstance(sensor_id, str) or not sensor_id.strip():
            raise ConfigError(f"{context}.inputs.{input_name} must reference a sensor ID")
        inputs[input_name] = sensor_id
    robot = value.get("robot")
    simulator = value.get("simulator")
    if (robot is None) == (simulator is None):
        raise ConfigError(f"{context} must define exactly one of robot or simulator")
    target_name = "robot" if robot is not None else "simulator"
    target = string(value, target_name, context)
    model = optional_string(value, "model", context)
    binding = optional_string(value, "binding", context)
    if (model is None) != (binding is None):
        raise ConfigError(f"{context}.model and {context}.binding must be provided together")
    if target_name == "simulator" and model is None:
        raise ConfigError(
            f"{context} simulator runtimes require model and binding; device-only mode is available for robots only"
        )
    return RuntimeConfig(
        runtime_id=runtime_id,
        robot=target if target_name == "robot" else None,
        simulator=target if target_name == "simulator" else None,
        model=model,
        binding=binding,
        inputs=inputs,
        server=server,
        inference_client=(
            parse_inference_client(value["inference_client"], f"{context}.inference_client")
            if "inference_client" in value
            else None
        ),
        options={
            name: option
            for name, option in value.items()
            if name
            not in {
                "robot",
                "simulator",
                "model",
                "binding",
                "inputs",
                "server",
                "inference_client",
            }
        },
    )


__all__ = ["RuntimeConfig"]
