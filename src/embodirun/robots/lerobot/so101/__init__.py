"""SO-101 follower hardware adapter using the Feetech SDK."""

from ... import RobotDefinition
from .adapter import (
    SO101_ACTION_SPACE,
    SO101_JOINTS,
    SO101_MOTORS,
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
)
from .config import SO101Config, StepLimitMode

ROBOT_DEFINITION = RobotDefinition(
    kind="lerobot.so101",
    config_factory=SO101Config.from_mapping,
    adapter_type=SO101Adapter,
    environment_group="robot-so101",
    python="3.12",
)

__all__ = [
    "ROBOT_DEFINITION",
    "SO101_ACTION_SPACE",
    "SO101_JOINTS",
    "SO101_MOTORS",
    "SO101_POSITION_FEATURES",
    "SO101Adapter",
    "SO101AdapterError",
    "SO101Config",
    "StepLimitMode",
]
