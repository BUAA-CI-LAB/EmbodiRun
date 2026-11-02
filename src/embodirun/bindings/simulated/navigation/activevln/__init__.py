"""ActiveVLN R2R actions expanded into Habitat's 25 cm / 15 degree steps."""

from embodirun.bindings import BindingDefinition

from .mapper import ActiveVLNSimulatorMapper

BINDING_DEFINITION = BindingDefinition(
    kind="simulated.navigation.activevln",
    robot_kind="unitree.go2",
    model_kind="activevln",
    mapper_factory=ActiveVLNSimulatorMapper,
    maximum_chunk_steps=72,
)

__all__ = ["BINDING_DEFINITION", "ActiveVLNSimulatorMapper"]
