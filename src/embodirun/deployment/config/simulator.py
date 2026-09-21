"""Simulator instance configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodirun.simulators import simulator_definition

from .validation import ConfigError, string


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    simulator_id: str
    kind: str
    node: str
    options: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_simulator(simulator_id: str, value: dict[str, Any]) -> SimulatorConfig:
    context = f"simulators.{simulator_id}"
    kind = string(value, "type", context)
    options = {name: option for name, option in value.items() if name not in {"type", "node"}}
    try:
        definition = simulator_definition(kind)
    except (KeyError, TypeError):
        raise ConfigError(f"{context}.type {kind!r} is not supported") from None
    try:
        definition.config_factory(simulator_id, options)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{context}: {error}") from error
    return SimulatorConfig(
        simulator_id=simulator_id,
        kind=kind,
        node=string(value, "node", context),
        options=options,
    )


__all__ = ["SimulatorConfig", "parse_simulator"]
