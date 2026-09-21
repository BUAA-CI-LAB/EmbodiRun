"""Pi0.5 binding for the unit-preserving simulated policy vector."""

from .... import BindingDefinition
from .mapper import (
    POLICY_ACTION_SPACE,
    SO101_POLICY_FEATURE_NAMES,
    SimulatedPolicyVectorPi05Mapper,
    SimulatedPolicyVectorPi05MapperError,
)

MAXIMUM_CHUNK_STEPS = 50
ADAPTER_CONFIG = {
    "state_fields": ("state_native",),
    "image_fields": (
        "observation.images.front",
        "observation.images.wrist",
    ),
    "return_steps": MAXIMUM_CHUNK_STEPS,
    "action_feature_names": SO101_POLICY_FEATURE_NAMES,
}

BINDING_DEFINITION = BindingDefinition(
    kind="simulated.policy_vector.pi05",
    robot_kind="simulated.policy_vector",
    model_kind="pi05",
    mapper_factory=SimulatedPolicyVectorPi05Mapper,
    maximum_chunk_steps=MAXIMUM_CHUNK_STEPS,
    adapter_config=ADAPTER_CONFIG,
)

__all__ = [
    "ADAPTER_CONFIG",
    "BINDING_DEFINITION",
    "MAXIMUM_CHUNK_STEPS",
    "POLICY_ACTION_SPACE",
    "SO101_POLICY_FEATURE_NAMES",
    "SimulatedPolicyVectorPi05Mapper",
    "SimulatedPolicyVectorPi05MapperError",
]
