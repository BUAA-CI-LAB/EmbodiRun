"""ARX5 robot adapter definition."""

from ... import RobotDefinition
from .adapter import ARX5_ACTION_SPACE, ARX5Adapter, ARX5AdapterError
from .config import ARX5Config

ROBOT_DEFINITION = RobotDefinition(
    kind="arx.x5",
    config_factory=ARX5Config.from_mapping,
    adapter_type=ARX5Adapter,
    environment_group="robot-arx5",
)

__all__ = [
    "ARX5_ACTION_SPACE",
    "ROBOT_DEFINITION",
    "ARX5Adapter",
    "ARX5AdapterError",
    "ARX5Config",
]
