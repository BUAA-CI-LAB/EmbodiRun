"""Dependency-light validation shared by navigation value objects."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from embodirun.types import Metadata

EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class NavigationContractError(ValueError):
    """A navigation value violates its public task contract."""


def finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NavigationContractError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise NavigationContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise NavigationContractError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise NavigationContractError(f"{name} must be at most {maximum}")
    return result


def sequence(value: object, name: str = "sequence") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NavigationContractError(f"{name} must be a non-negative integer")
    return value


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NavigationContractError(f"{name} must be a positive integer")
    return value


def encoded_bytes(value: object, name: str, maximum: int, signature: bytes | None = None) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise NavigationContractError(f"{name} must be non-empty bytes")
    if len(value) > maximum:
        raise NavigationContractError(f"{name} exceeds {maximum} bytes")
    if signature is not None and not value.startswith(signature):
        raise NavigationContractError(f"{name} has an invalid file signature")
    return value


def _json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NavigationContractError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NavigationContractError(f"{name} keys must be strings")
            _json_value(nested, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _json_value(nested, f"{name}[{index}]")
        return
    raise NavigationContractError(f"{name} must contain JSON-compatible values")


def metadata(value: Metadata, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NavigationContractError(f"{name} must be a mapping")
    _json_value(value, name)
    return dict(value)


__all__ = [
    "EPISODE_ID_PATTERN",
    "NavigationContractError",
    "encoded_bytes",
    "finite",
    "metadata",
    "positive_int",
    "sequence",
]
