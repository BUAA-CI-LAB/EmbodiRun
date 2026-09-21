"""VLABench simulator adapter."""

from .. import SimulatorDefinition
from .adapter import VLABENCH_ACTION_DIM, VLABenchAdapter
from .config import VLABenchConfig

SIMULATOR_DEFINITION = SimulatorDefinition(
    kind="vlabench",
    embodiment_kind="franka.panda.eef",
    image_fields=(
        "observation.images.image",
        "observation.images.second_image",
        "observation.images.wrist_image",
    ),
    config_factory=VLABenchConfig.from_mapping,
    adapter_type=VLABenchAdapter,
    environment_group="sim-vlabench",
    python="3.12",
)


__all__ = [
    "SIMULATOR_DEFINITION",
    "VLABENCH_ACTION_DIM",
    "VLABenchAdapter",
    "VLABenchConfig",
]
