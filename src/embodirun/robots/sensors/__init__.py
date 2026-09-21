"""Robot sensor contracts and hardware implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SensorInput:
    """One configured sensor exposed under a runtime input name."""

    sensor_id: str
    name: str
    kind: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in ("sensor_id", "name", "kind"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"sensor input {field_name} must not be empty")
        if not isinstance(self.options, Mapping) or any(not isinstance(key, str) for key in self.options):
            raise ValueError("sensor input options must be a mapping with string keys")
        object.__setattr__(self, "options", dict(self.options))


__all__ = ["SensorInput"]
