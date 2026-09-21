"""Simulator adapters and definitions."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any

from .adapter import SimulationStep, SimulatorAdapter, SimulatorObservation


@dataclass(frozen=True, slots=True)
class SimulatorDefinition:
    """Static information needed to construct one simulator adapter."""

    kind: str
    embodiment_kind: str
    image_fields: tuple[str, ...]
    config_factory: Callable[[str, Mapping[str, Any]], Any]
    adapter_type: type[SimulatorAdapter]
    environment_group: str
    python: str | None = None

    def __post_init__(self) -> None:
        for name in ("kind", "embodiment_kind", "environment_group"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"simulator {name} must not be empty")
        if not callable(self.config_factory):
            raise TypeError("simulator config_factory must be callable")
        if not issubclass(self.adapter_type, SimulatorAdapter):
            raise TypeError("simulator adapter_type must inherit SimulatorAdapter")
        if not self.image_fields or any(not isinstance(field, str) or not field.strip() for field in self.image_fields):
            raise ValueError("simulator image_fields must contain non-empty strings")
        if len(self.image_fields) != len(set(self.image_fields)):
            raise ValueError("simulator image_fields must be unique")


_SIMULATOR_KIND = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


@cache
def simulator_definition(kind: str) -> SimulatorDefinition:
    """Load ``simulators.<kind>.SIMULATOR_DEFINITION`` by package convention."""

    if not isinstance(kind, str) or _SIMULATOR_KIND.fullmatch(kind) is None:
        raise KeyError(kind)
    module_name = f"{__name__}.{kind}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is not None and (error.name == module_name or module_name.startswith(f"{error.name}.")):
            raise KeyError(kind) from None
        raise
    try:
        definition = module.SIMULATOR_DEFINITION
    except AttributeError:
        raise TypeError(f"{module_name} does not declare SIMULATOR_DEFINITION") from None
    if not isinstance(definition, SimulatorDefinition):
        raise TypeError(f"{module_name}.SIMULATOR_DEFINITION is invalid")
    if definition.kind != kind:
        raise TypeError(f"{module_name}.SIMULATOR_DEFINITION declares kind {definition.kind!r}")
    return definition


__all__ = [
    "SimulationStep",
    "SimulatorAdapter",
    "SimulatorDefinition",
    "SimulatorObservation",
    "simulator_definition",
]
