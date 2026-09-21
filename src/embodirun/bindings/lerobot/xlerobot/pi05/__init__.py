"""Map explicitly named native-unit Pi0.5 outputs to XLeRobot arm actions."""

from embodirun.bindings import BindingDefinition
from embodirun.bindings.lerobot.so101.pi05.mapper import Pi05SO101Mapper
from embodirun.robots import RobotAction
from embodirun.robots.lerobot.xlerobot.units import ARM_UNITS, XLEROBOT_ACTION_SPACE, stamped_metadata


class Pi05XLeRobotMapper(Pi05SO101Mapper):
    position_features = tuple(ARM_UNITS)
    robot_action_space = XLEROBOT_ACTION_SPACE

    def _action_values(self, positions):
        return dict(zip(self.position_features, positions))

    def map_result(self, result):
        return tuple(
            RobotAction(
                action.timestamp_s, action.values, stamped_metadata(action.values, action.metadata, scope="arms")
            )
            for action in super().map_result(result)
        )


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.xlerobot.pi05",
    robot_kind="lerobot.xlerobot",
    model_kind="pi05",
    mapper_factory=Pi05XLeRobotMapper,
    maximum_chunk_steps=50,
    adapter_config={"state_fields": tuple(ARM_UNITS), "action_feature_names": tuple(ARM_UNITS)},
)
