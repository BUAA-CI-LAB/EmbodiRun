"""Canonical Unitree Go2 hardware bindings and robot-resident agent.

The host-only bindings stay lazy because ``agent`` is deployed to the Go2's
Python 3.8 environment and must be importable without the host task stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GO2_ACTION_SPACE",
    "ROBOT_DEFINITION",
    "Go2Adapter",
    "Go2AdapterError",
    "Go2CameraClient",
    "Go2CameraError",
    "Go2ClientError",
    "Go2Config",
    "Go2ControlClient",
]

_SYMBOL_MODULES = {
    "GO2_ACTION_SPACE": ".adapter",
    "Go2Adapter": ".adapter",
    "Go2AdapterError": ".adapter",
    "Go2CameraClient": ".camera",
    "Go2CameraError": ".camera",
    "Go2ClientError": ".client",
    "Go2Config": ".config",
    "Go2ControlClient": ".client",
}


def __getattr__(name: str) -> Any:
    if name == "ROBOT_DEFINITION":
        from ... import RobotDefinition
        from .adapter import Go2Adapter
        from .config import Go2Config

        value = RobotDefinition(
            kind="unitree.go2",
            config_factory=Go2Config.from_mapping,
            adapter_type=Go2Adapter,
            environment_group="robot-go2",
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
