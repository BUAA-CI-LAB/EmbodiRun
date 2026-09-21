"""Wire and robot action identities for the DM0.5-to-ARX5 binding."""

POLICY_FAMILY = "dm0.5"
ROBOT_MODEL = "arx5"
POLICY_ACTION_SPACE = "dm05.action_chunk.v1"
ROBOT_ACTION_SPACE = "arx.x5.eef_xyzrpy_gripper.absolute.v1"
ACTION_REPRESENTATION = "absolute_eef_xyzrpy_with_absolute_gripper"
ACTION_FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
)
STATE_REPRESENTATION = "eef_xyzrpy_gripper"
STATE_DESCRIPTION = ("eef", "eef", "eef", "eef", "eef", "eef", "gripper")
VIEW_ORDER = ("cam_global", "cam_side", "cam_arm")

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
]
