"""StreamVLN policy-to-Go2 binding."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["BINDING_DEFINITION", "StreamVLNGo2Mapper", "main"]
_SYMBOL_MODULES = {"main": ".cli", "StreamVLNGo2Mapper": ".mapper"}


def __getattr__(name: str) -> Any:
    if name == "BINDING_DEFINITION":
        from .... import BindingDefinition
        from .mapper import StreamVLNGo2Mapper

        value = BindingDefinition(
            kind="unitree.go2.streamvln",
            robot_kind="unitree.go2",
            model_kind="streamvln",
            mapper_factory=StreamVLNGo2Mapper,
            maximum_chunk_steps=4,
            adapter_config={},
        )
        globals()[name] = value
        return value
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
