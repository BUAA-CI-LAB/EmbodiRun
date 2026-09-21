"""Simulated policy-vector robot with no physical-unit assumptions."""

from ... import RobotDefinition
from .adapter import (
    POLICY_VECTOR_ACTION_SPACE,
    POLICY_VECTOR_ACTION_TYPE,
    PolicyVectorAdapter,
    PolicyVectorAdapterError,
)
from .config import (
    POLICY_VECTOR_SIZE,
    STATE_FIELD,
    UNVERIFIED_UNITS,
    PolicyVectorConfig,
)

ROBOT_DEFINITION = RobotDefinition(
    kind="simulated.policy_vector",
    config_factory=PolicyVectorConfig.from_mapping,
    adapter_type=PolicyVectorAdapter,
    environment_group="host",
    python=None,
)

__all__ = [
    "ROBOT_DEFINITION",
    "POLICY_VECTOR_ACTION_SPACE",
    "POLICY_VECTOR_ACTION_TYPE",
    "POLICY_VECTOR_SIZE",
    "STATE_FIELD",
    "UNVERIFIED_UNITS",
    "PolicyVectorAdapter",
    "PolicyVectorAdapterError",
    "PolicyVectorConfig",
]
