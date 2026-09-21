"""Physical robot drivers, SDK bindings, and robot-resident services."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any

from .adapter import RobotAction, RobotAdapter, RobotObservation, RobotPreparationRefused


@dataclass(frozen=True, slots=True)
class RobotDefinition:
    """Static information Deploy needs before constructing a robot adapter."""

    kind: str
    config_factory: Callable[[str, Mapping[str, Any]], Any]
    adapter_type: type[RobotAdapter]
    environment_group: str
    python: str | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("robot kind must not be empty")
        if not self.environment_group.strip():
            raise ValueError("robot environment group must not be empty")
        if not callable(self.config_factory):
            raise TypeError("config_factory must be callable")
        if not issubclass(self.adapter_type, RobotAdapter):
            raise TypeError("adapter_type must inherit RobotAdapter")


_ROBOT_KIND = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z")


@cache
def robot_definition(kind: str) -> RobotDefinition:
    """Load ``robots.<kind>.ROBOT_DEFINITION`` by package convention."""

    if not isinstance(kind, str) or _ROBOT_KIND.fullmatch(kind) is None:
        raise KeyError(kind)
    module_name = f"{__name__}.{kind}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is not None and (error.name == module_name or module_name.startswith(f"{error.name}.")):
            raise KeyError(kind) from None
        raise
    try:
        definition = module.ROBOT_DEFINITION
    except AttributeError:
        raise TypeError(f"{module_name} does not declare ROBOT_DEFINITION") from None
    if not isinstance(definition, RobotDefinition):
        raise TypeError(f"{module_name}.ROBOT_DEFINITION is invalid")
    if definition.kind != kind:
        raise TypeError(f"{module_name}.ROBOT_DEFINITION declares kind {definition.kind!r}")
    return definition


__all__ = [
    "RobotAction",
    "RobotAdapter",
    "RobotDefinition",
    "RobotObservation",
    "RobotPreparationRefused",
    "robot_definition",
]
