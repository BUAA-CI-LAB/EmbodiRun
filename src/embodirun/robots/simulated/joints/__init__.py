"""Simulated SO-style five-joint arm and gripper."""

from ... import RobotDefinition
from .adapter import (
    FAKE_ACTION_SPACE,
    FAKE_JOINTS,
    FAKE_POSITION_FEATURES,
    FakeJointsAdapter,
    FakeJointsAdapterError,
)
from .config import FakeJointsConfig, FakeStepLimitMode

ROBOT_DEFINITION = RobotDefinition(
    kind="simulated.joints",
    config_factory=FakeJointsConfig.from_mapping,
    adapter_type=FakeJointsAdapter,
    environment_group="host",
    python=None,
)

__all__ = [
    "ROBOT_DEFINITION",
    "FAKE_ACTION_SPACE",
    "FAKE_JOINTS",
    "FAKE_POSITION_FEATURES",
    "FakeJointsAdapter",
    "FakeJointsAdapterError",
    "FakeJointsConfig",
    "FakeStepLimitMode",
]
