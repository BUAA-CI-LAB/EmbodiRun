"""DM0.5 inference-output binding for ARX5."""

from .... import BindingDefinition
from .contract import (
    ACTION_FEATURE_NAMES,
    ACTION_REPRESENTATION,
    POLICY_ACTION_SPACE,
    POLICY_FAMILY,
    ROBOT_ACTION_SPACE,
    ROBOT_MODEL,
    STATE_DESCRIPTION,
    STATE_REPRESENTATION,
    VIEW_ORDER,
)
from .mapper import (
    DM05ARX5Mapper,
    DM05ARX5MapperConfig,
    DM05ARX5MapperError,
)

BINDING_DEFINITION = BindingDefinition(
    kind="arx.x5.dm05",
    robot_kind="arx.x5",
    model_kind="dm05",
    mapper_factory=DM05ARX5Mapper,
    maximum_chunk_steps=25,
)

__all__ = [
    "ACTION_FEATURE_NAMES",
    "ACTION_REPRESENTATION",
    "POLICY_ACTION_SPACE",
    "POLICY_FAMILY",
    "ROBOT_ACTION_SPACE",
    "ROBOT_MODEL",
    "STATE_DESCRIPTION",
    "STATE_REPRESENTATION",
    "VIEW_ORDER",
    "BINDING_DEFINITION",
    "DM05ARX5Mapper",
    "DM05ARX5MapperConfig",
    "DM05ARX5MapperError",
]
