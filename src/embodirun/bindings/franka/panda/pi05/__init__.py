"""Pi0.5 mapping for the normalized Franka Panda end-effector space."""

from .... import BindingDefinition
from .mapper import POLICY_ACTION_SPACE, Pi05FrankaPandaMapper

MAXIMUM_CHUNK_STEPS = 50
ADAPTER_CONFIG = {
    "state_fields": ("observation.state",),
}


BINDING_DEFINITION = BindingDefinition(
    kind="franka.panda.pi05",
    robot_kind="franka.panda.eef",
    model_kind="pi05",
    mapper_factory=Pi05FrankaPandaMapper,
    maximum_chunk_steps=MAXIMUM_CHUNK_STEPS,
    adapter_config=ADAPTER_CONFIG,
)


__all__ = [
    "BINDING_DEFINITION",
    "MAXIMUM_CHUNK_STEPS",
    "POLICY_ACTION_SPACE",
    "Pi05FrankaPandaMapper",
]
