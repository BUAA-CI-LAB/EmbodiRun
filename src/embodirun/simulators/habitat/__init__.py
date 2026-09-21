"""Habitat-Sim adapter for StreamVLN-style navigation."""

from .. import SimulatorDefinition
from ..navigation import NAVIGATION_IMAGE_FIELD
from .adapter import HabitatAdapter
from .config import HabitatConfig

SIMULATOR_DEFINITION = SimulatorDefinition(
    kind="habitat",
    embodiment_kind="unitree.go2",
    image_fields=(NAVIGATION_IMAGE_FIELD,),
    config_factory=HabitatConfig.from_mapping,
    adapter_type=HabitatAdapter,
    environment_group="sim-habitat",
    python="3.11",
)


__all__ = ["SIMULATOR_DEFINITION", "HabitatAdapter", "HabitatConfig"]
