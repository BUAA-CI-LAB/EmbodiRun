"""Pi0.5 value mapping for the SO-101 follower."""

from embodirun.robots.lerobot.so101 import SO101_POSITION_FEATURES

from .... import BindingDefinition
from .mapper import (
    POLICY_ACTION_SPACE,
    Pi05SO101Mapper,
    Pi05SO101MapperError,
)

MAXIMUM_CHUNK_STEPS = 50
ADAPTER_CONFIG = {
    "state_fields": ("joint_positions_deg", "gripper_position"),
    "action_feature_names": SO101_POSITION_FEATURES,
}


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.so101.pi05",
    robot_kind="lerobot.so101",
    model_kind="pi05",
    mapper_factory=Pi05SO101Mapper,
    maximum_chunk_steps=MAXIMUM_CHUNK_STEPS,
    adapter_config=ADAPTER_CONFIG,
)

__all__ = [
    "BINDING_DEFINITION",
    "MAXIMUM_CHUNK_STEPS",
    "POLICY_ACTION_SPACE",
    "Pi05SO101Mapper",
    "Pi05SO101MapperError",
]
