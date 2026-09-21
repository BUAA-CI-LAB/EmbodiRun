"""Wired multi-leader SO-101 teleoperation and episode collection.

A leader owns one arm, one UDP port and the followers that obey it. Because the
port is the group key, moving a follower between leaders is a configuration edit
rather than a code change. See ``config.example.yaml`` for the layout schema.
"""

from __future__ import annotations

from .config import (
    ConfigError,
    FollowerConfig,
    LeaderConfig,
    TeleopConfig,
    load_config,
    parse_config,
)
from .follower import FollowerNode, clamp_target
from .leader import LeaderBroadcaster
from .protocol import JOINTS, ProtocolError, decode, encode
from .recorder import EpisodeRecorder

__all__ = [
    "JOINTS",
    "ConfigError",
    "EpisodeRecorder",
    "FollowerConfig",
    "FollowerNode",
    "LeaderBroadcaster",
    "LeaderConfig",
    "ProtocolError",
    "TeleopConfig",
    "clamp_target",
    "decode",
    "encode",
    "load_config",
    "parse_config",
]
