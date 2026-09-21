"""Robot instance configuration and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodirun.robots import robot_definition

from .validation import (
    ConfigError,
    optional_string,
    string,
)


@dataclass(frozen=True, slots=True)
class RobotConfig:
    robot_id: str
    kind: str
    node: str
    port: str | None
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    resource: str | None = None
    owner: str | None = None


def parse_robot(robot_id: str, value: dict[str, Any]) -> RobotConfig:
    context = f"robots.{robot_id}"
    kind = string(value, "type", context)
    resource = value.get("resource")
    if resource is not None and (not isinstance(resource, str) or not resource.strip()):
        raise ConfigError(f"{context}.resource must be a non-empty string")
    owner = value.get("owner")
    if owner is not None and (not isinstance(owner, str) or not owner.strip()):
        raise ConfigError(f"{context}.owner must be a non-empty string")
    options = {
        name: option
        for name, option in value.items()
        if name
        not in {
            "type",
            "node",
            "resource",
            "owner",
        }
    }
    try:
        definition = robot_definition(kind)
    except (KeyError, TypeError):
        raise ConfigError(f"{context}.type {kind!r} is not supported") from None
    try:
        definition.config_factory(robot_id, options)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{context}: {error}") from error
    return RobotConfig(
        robot_id=robot_id,
        kind=kind,
        node=string(value, "node", context),
        port=optional_string(value, "port", context),
        resource=resource,
        owner=owner,
        options=options,
    )


def robot_ports(robot: RobotConfig) -> tuple[str, ...]:
    """Return every configured physical port used by a robot.

    Most robot definitions expose one ``port`` field.  The dual SO-101
    definition keeps its two ports in options so that the canonical config
    parser can still validate them; callers that probe or reserve resources
    must therefore use this helper instead of looking only at ``robot.port``.
    """

    if robot.kind == "lerobot.bi_so101":
        values = (robot.options.get("left_port"), robot.options.get("right_port"))
    else:
        values = (robot.port or robot.options.get("port"),)
    return tuple(value for value in values if isinstance(value, str) and value)


def robot_calibration_ids(robot: RobotConfig) -> tuple[str, ...]:
    """Return calibration file stems that must exist for a configured robot."""

    if robot.kind == "lerobot.bi_so101":
        values = (
            robot.options.get("left_calibration_id", f"{robot.robot_id}_left"),
            robot.options.get("right_calibration_id", f"{robot.robot_id}_right"),
        )
    else:
        values = (robot.options.get("calibration_id") or robot.robot_id,)
    return tuple(value for value in values if isinstance(value, str) and value)


__all__ = [
    "RobotConfig",
    "parse_robot",
    "robot_calibration_ids",
    "robot_ports",
]
