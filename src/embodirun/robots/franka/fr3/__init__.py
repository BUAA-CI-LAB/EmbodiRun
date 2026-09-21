"""Franka Research 3 hardware adapter implemented with Franky."""

from ... import RobotDefinition
from .adapter import FR3_ACTION_SPACE, FR3Adapter, FR3AdapterError
from .config import FR3Config

ROBOT_DEFINITION = RobotDefinition(
    kind="franka.fr3",
    config_factory=FR3Config.from_mapping,
    adapter_type=FR3Adapter,
    environment_group="robot-fr3",
)

__all__ = [
    "FR3_ACTION_SPACE",
    "ROBOT_DEFINITION",
    "FR3Adapter",
    "FR3AdapterError",
    "FR3Config",
]
