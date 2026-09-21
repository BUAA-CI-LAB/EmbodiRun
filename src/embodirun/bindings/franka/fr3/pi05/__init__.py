"""Pi0.5 value mapping for the Franka Research 3."""

from .... import BindingDefinition
from .mapper import (
    POLICY_ACTION_SPACE,
    Pi05FR3Mapper,
    Pi05FR3MapperConfig,
    Pi05FR3MapperError,
)

BINDING_DEFINITION = BindingDefinition(
    kind="franka.fr3.pi05",
    robot_kind="franka.fr3",
    model_kind="pi05",
    mapper_factory=Pi05FR3Mapper,
    maximum_chunk_steps=50,
)

__all__ = [
    "BINDING_DEFINITION",
    "POLICY_ACTION_SPACE",
    "Pi05FR3Mapper",
    "Pi05FR3MapperConfig",
    "Pi05FR3MapperError",
]
