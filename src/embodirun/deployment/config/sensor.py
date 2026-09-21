"""Sensor instance configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import ConfigError, string


@dataclass(frozen=True, slots=True)
class SensorConfig:
    sensor_id: str
    kind: str
    node: str
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    resource: str | None = None
    owner: str | None = None


def parse_sensor(sensor_id: str, value: dict[str, Any]) -> SensorConfig:
    context = f"sensors.{sensor_id}"
    resource = value.get("resource")
    if resource is not None and (not isinstance(resource, str) or not resource.strip()):
        raise ConfigError(f"{context}.resource must be a non-empty string")
    owner = value.get("owner")
    if owner is not None and (not isinstance(owner, str) or not owner.strip()):
        raise ConfigError(f"{context}.owner must be a non-empty string")
    return SensorConfig(
        sensor_id=sensor_id,
        kind=string(value, "type", context),
        node=string(value, "node", context),
        resource=resource,
        owner=owner,
        options={
            name: option
            for name, option in value.items()
            if name
            not in {
                "type",
                "node",
                "resource",
                "owner",
            }
        },
    )


__all__ = ["SensorConfig"]
