"""Explicit policy-output to hardware bindings."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any, Protocol

from embodirun.model_services import PolicyObservation, PolicyResult
from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame


class BindingMapper(Protocol):
    """Translate observations and results for one policy-robot binding."""

    policy_action_space: str

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
        """Build one policy request without performing inference."""

    def map_result(self, result: PolicyResult) -> Sequence[RobotAction]:
        """Map one policy result to an ordered, non-empty action chunk."""


@dataclass(frozen=True, slots=True)
class BindingDefinition:
    """Static compatibility and execution information for one binding."""

    kind: str
    robot_kind: str
    model_kind: str
    mapper_factory: Callable[[], BindingMapper]
    maximum_chunk_steps: int
    adapter_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("binding kind must not be empty")
        if not self.robot_kind.strip():
            raise ValueError("binding robot kind must not be empty")
        if not self.model_kind.strip():
            raise ValueError("binding model kind must not be empty")
        if not callable(self.mapper_factory):
            raise TypeError("mapper_factory must be callable")
        if (
            isinstance(self.maximum_chunk_steps, bool)
            or not isinstance(self.maximum_chunk_steps, int)
            or self.maximum_chunk_steps <= 0
        ):
            raise ValueError("maximum_chunk_steps must be a positive integer")
        if self.adapter_config is not None:
            if not isinstance(self.adapter_config, Mapping) or any(
                not isinstance(key, str) for key in self.adapter_config
            ):
                raise TypeError("adapter_config must be an object with string keys")
            object.__setattr__(self, "adapter_config", dict(self.adapter_config))


_BINDING_KIND = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z")


@cache
def binding_definition(kind: str) -> BindingDefinition:
    """Load ``bindings.<kind>.BINDING_DEFINITION`` by package convention."""

    if not isinstance(kind, str) or _BINDING_KIND.fullmatch(kind) is None:
        raise KeyError(kind)
    module_name = f"{__name__}.{kind}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is not None and (error.name == module_name or module_name.startswith(f"{error.name}.")):
            raise KeyError(kind) from None
        raise
    try:
        definition = module.BINDING_DEFINITION
    except AttributeError:
        raise TypeError(f"{module_name} does not declare BINDING_DEFINITION") from None
    if not isinstance(definition, BindingDefinition):
        raise TypeError(f"{module_name}.BINDING_DEFINITION is invalid")
    if definition.kind != kind:
        raise TypeError(f"{module_name}.BINDING_DEFINITION declares kind {definition.kind!r}")
    return definition


__all__ = [
    "BindingMapper",
    "BindingDefinition",
    "binding_definition",
]
