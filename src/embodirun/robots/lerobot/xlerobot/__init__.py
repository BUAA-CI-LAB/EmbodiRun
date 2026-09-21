"""XLeRobot external-owner proxy."""

from ... import RobotDefinition
from .adapter import XLeRobotAdapter, XLeRobotAdapterError
from .config import XLeRobotConfig
from .units import (
    ARM_UNITS,
    BASE_UNITS,
    XLEROBOT_ACTION_SPACE,
    canonical_units,
    scope_for_values,
    stamped_metadata,
    validate_action,
)

ROBOT_DEFINITION = RobotDefinition(
    kind="lerobot.xlerobot",
    config_factory=XLeRobotConfig.from_mapping,
    adapter_type=XLeRobotAdapter,
    environment_group="robot-xlerobot-external-owner",
    python=None,
)

__all__ = [
    "ARM_UNITS",
    "BASE_UNITS",
    "ROBOT_DEFINITION",
    "XLEROBOT_ACTION_SPACE",
    "XLeRobotAdapter",
    "XLeRobotAdapterError",
    "XLeRobotConfig",
    "canonical_units",
    "scope_for_values",
    "stamped_metadata",
    "validate_action",
]
