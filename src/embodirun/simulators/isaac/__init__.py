"""Isaac Sim Go2 adapter for StreamVLN-style navigation."""

from .. import SimulatorDefinition
from ..navigation import NAVIGATION_IMAGE_FIELD
from .adapter import IsaacAdapter
from .config import IsaacConfig

SIMULATOR_DEFINITION = SimulatorDefinition(
    kind="isaac",
    embodiment_kind="unitree.go2",
    image_fields=(NAVIGATION_IMAGE_FIELD,),
    config_factory=IsaacConfig.from_mapping,
    adapter_type=IsaacAdapter,
    environment_group="sim-isaac",
    python="3.12",
)


__all__ = ["SIMULATOR_DEFINITION", "IsaacAdapter", "IsaacConfig"]
