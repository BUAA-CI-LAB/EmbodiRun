"""Pi0.5 binding for the LIBERO Franka Panda end-effector task."""

from .... import BindingDefinition
from .mapper import (
    ACTION_FEATURE_NAMES,
    POLICY_ACTION_SPACE,
    LiberoPi05Mapper,
    LiberoPi05MapperError,
)

# The LIBERO checkpoint this binding targets returns a ten-row chunk.
MAXIMUM_CHUNK_STEPS = 10
ADAPTER_CONFIG = {
    "state_fields": ("observation.state",),
    "image_fields": ("image", "wrist_image"),
    "action_feature_names": ACTION_FEATURE_NAMES,
}

BINDING_DEFINITION = BindingDefinition(
    kind="simulated.libero.pi05",
    robot_kind="franka.panda.eef",
    model_kind="pi05",
    mapper_factory=LiberoPi05Mapper,
    maximum_chunk_steps=MAXIMUM_CHUNK_STEPS,
    adapter_config=ADAPTER_CONFIG,
)

__all__ = [
    "ACTION_FEATURE_NAMES",
    "ADAPTER_CONFIG",
    "BINDING_DEFINITION",
    "MAXIMUM_CHUNK_STEPS",
    "POLICY_ACTION_SPACE",
    "LiberoPi05Mapper",
    "LiberoPi05MapperError",
]
