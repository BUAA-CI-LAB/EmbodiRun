"""Pi0.5 binding for left/right SO-101 position actions."""

from embodirun.bindings import BindingDefinition
from embodirun.bindings.lerobot.so101.pi05.mapper import Pi05SO101Mapper
from embodirun.robots.lerobot.bi_so101 import (
    BI_SO101_ACTION_SPACE,
    BI_SO101_POSITION_FEATURES,
)


class Pi05BiSO101Mapper(Pi05SO101Mapper):
    """Map a flat twelve-value Pi0.5 chunk to two arm targets."""

    position_features = BI_SO101_POSITION_FEATURES
    robot_action_space = BI_SO101_ACTION_SPACE

    def _action_values(self, positions: list[float]) -> dict[str, object]:
        return {
            "type": "joint_position",
            "left": {
                "joint_positions_deg": positions[:5],
                "gripper_position": positions[5],
            },
            "right": {
                "joint_positions_deg": positions[6:11],
                "gripper_position": positions[11],
            },
        }


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.bi_so101.pi05",
    robot_kind="lerobot.bi_so101",
    model_kind="pi05",
    mapper_factory=Pi05BiSO101Mapper,
    maximum_chunk_steps=50,
    adapter_config={
        "state_fields": BI_SO101_POSITION_FEATURES,
        "action_feature_names": BI_SO101_POSITION_FEATURES,
    },
)

__all__ = ["BINDING_DEFINITION", "Pi05BiSO101Mapper"]
