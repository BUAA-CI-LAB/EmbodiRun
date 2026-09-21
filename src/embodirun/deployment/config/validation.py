"""Shared field-level validation for deployment configuration objects."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar

RESOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

ConfigValue = TypeVar("ConfigValue")


class ConfigError(ValueError):
    """The deployment YAML does not satisfy the supported schema."""


def named_section(
    value: Any,
    name: str,
    parser: Callable[[str, dict[str, Any]], ConfigValue],
) -> dict[str, ConfigValue]:
    section = mapping(value, name)
    result: dict[str, ConfigValue] = {}
    for item_id, item_value in section.items():
        if not RESOURCE_ID.fullmatch(item_id):
            raise ConfigError(
                f"{name} keys must start with an alphanumeric character and contain "
                "only alphanumerics, dots, underscores, or hyphens"
            )
        result[item_id] = parser(item_id, mapping(item_value, f"{name}.{item_id}"))
    return result


def mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{context} keys must be strings")
    return value


def string(value: dict[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return result


def optional_string(
    value: dict[str, Any],
    key: str,
    context: str,
) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string when provided")
    return result


def integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer")
    return value


def positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{context} must be positive")
    return float(value)


def boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be a boolean")
    return value


def reject_unknown(
    value: dict[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{context} contains unknown fields: {', '.join(unknown)}")
