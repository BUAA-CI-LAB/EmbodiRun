"""Configuration for wired SO-101 teleoperation.

The file describes two lists. A *leader* owns one arm, one UDP port and the set
of addresses it broadcasts to. A *follower* names the leader it obeys. The
mapping is therefore pure configuration:

* one leader driving every follower — point every follower at that leader
* two leaders driving two followers each — point two followers at each
* any leader driving any follower — change one ``leader`` field
* a leader driving nothing — simply do not reference it

Nothing in this module opens a device or a socket, so the schema can be checked
on a machine without hardware.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Values in the shipped example use ``<LIKE_THIS>`` so that a real host, serial or
# path is never committed. Substituting them is a prerequisite for a real run,
# and resolving the config is where that is enforced.
PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")

DEFAULT_PORT = 55101
DEFAULT_FPS = 20.0
DEFAULT_CAMERA_FPS = 20.0
DEFAULT_MAX_STEP = 3.0
DEFAULT_MAX_LEAD = 15.0
DEFAULT_WATCHDOG_S = 0.30


class ConfigError(ValueError):
    """The teleoperation configuration is missing, malformed or unresolved."""


@dataclass(frozen=True)
class LeaderConfig:
    """One leader arm broadcasting to the followers that name it."""

    id: str
    serial: str
    arm_id: str
    calibration_dir: str
    advertise: str
    port: int = DEFAULT_PORT
    fps: float = DEFAULT_FPS

    @property
    def destination(self) -> tuple[str, int]:
        return (self.advertise, self.port)


@dataclass(frozen=True)
class FollowerConfig:
    """One follower arm obeying exactly one leader."""

    id: str
    leader: str
    bind: str
    serial: str
    arm_id: str
    calibration_dir: str
    ssh_host: str | None = None
    cameras: tuple[int, ...] = ()
    camera_roles: tuple[str, ...] = ()
    camera_fps: float = DEFAULT_CAMERA_FPS
    record_dir: str | None = None
    max_step: float = DEFAULT_MAX_STEP
    max_lead: float = DEFAULT_MAX_LEAD
    watchdog_s: float = DEFAULT_WATCHDOG_S


@dataclass(frozen=True)
class TeleopConfig:
    """A whole teleoperation layout."""

    leaders: tuple[LeaderConfig, ...]
    followers: tuple[FollowerConfig, ...] = ()
    _by_id: dict[str, LeaderConfig] = field(default_factory=dict, repr=False)

    def leader(self, leader_id: str) -> LeaderConfig:
        try:
            return self._by_id[leader_id]
        except KeyError:
            raise ConfigError(f"unknown leader {leader_id!r}") from None

    def followers_of(self, leader_id: str) -> tuple[FollowerConfig, ...]:
        """Return the followers that name ``leader_id``, in configuration order."""

        return tuple(f for f in self.followers if f.leader == leader_id)

    def destinations(self, leader_id: str) -> tuple[str, ...]:
        """Return the bind addresses a leader must broadcast to.

        A leader with no followers has no destinations, which is a valid layout
        and is why this returns an empty tuple instead of failing.
        """

        return tuple(f.bind for f in self.followers_of(leader_id))

    def unresolved(self) -> tuple[str, ...]:
        """Return every ``path: value`` pair that still holds a placeholder."""

        found: list[str] = []
        for leader in self.leaders:
            for name, value in vars(leader).items():
                if isinstance(value, str) and PLACEHOLDER.search(value):
                    found.append(f"leaders[{leader.id}].{name}={value}")
        for follower in self.followers:
            for name, value in vars(follower).items():
                if isinstance(value, str) and PLACEHOLDER.search(value):
                    found.append(f"followers[{follower.id}].{name}={value}")
        return tuple(found)

    def require_resolved(self) -> None:
        """Raise unless every placeholder has been replaced with a real value."""

        missing = self.unresolved()
        if missing:
            listed = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise ConfigError(
                "configuration still contains placeholders: "
                f"{listed}{more}. Copy the example and substitute the real "
                "hosts, serials, calibration and recording paths."
            )


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"{where} is missing required key {key!r}")
    return mapping[key]


def _positive(value: Any, key: str, where: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ConfigError(f"{where}.{key} must be positive")
    return number


def _leader(raw: Any, index: int) -> LeaderConfig:
    where = f"leaders[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    port = raw.get("port", DEFAULT_PORT)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError(f"{where}.port must be an integer between 1 and 65535")
    return LeaderConfig(
        id=str(_require(raw, "id", where)),
        serial=str(_require(raw, "serial", where)),
        arm_id=str(_require(raw, "arm_id", where)),
        calibration_dir=str(_require(raw, "calibration_dir", where)),
        advertise=str(_require(raw, "advertise", where)),
        port=port,
        fps=_positive(raw.get("fps", DEFAULT_FPS), "fps", where),
    )


def _follower(raw: Any, index: int) -> FollowerConfig:
    where = f"followers[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    cameras = raw.get("cameras", [])
    if not isinstance(cameras, (list, tuple)) or any(
        isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in cameras
    ):
        raise ConfigError(f"{where}.cameras must be a list of device indexes")
    roles = raw.get("camera_roles", [])
    if roles and (not isinstance(roles, (list, tuple)) or len(roles) != len(cameras)):
        raise ConfigError(
            f"{where}.camera_roles must match cameras one for one, got {len(roles)} roles for {len(cameras)} cameras"
        )
    if roles and any(not isinstance(r, str) or not r.strip() for r in roles):
        raise ConfigError(f"{where}.camera_roles must be non-empty strings")
    if roles and (len(set(roles)) != len(roles) or any(r in {".", ".."} or "/" in r or "\\" in r for r in roles)):
        raise ConfigError(f"{where}.camera_roles must be unique directory names")
    record_dir = raw.get("record_dir")
    return FollowerConfig(
        id=str(_require(raw, "id", where)),
        leader=str(_require(raw, "leader", where)),
        bind=str(_require(raw, "bind", where)),
        serial=str(_require(raw, "serial", where)),
        arm_id=str(_require(raw, "arm_id", where)),
        calibration_dir=str(_require(raw, "calibration_dir", where)),
        ssh_host=None if raw.get("ssh_host") is None else str(raw["ssh_host"]),
        cameras=tuple(cameras),
        camera_roles=tuple(str(r) for r in roles),
        camera_fps=_positive(raw.get("camera_fps", DEFAULT_CAMERA_FPS), "camera_fps", where),
        record_dir=None if record_dir is None else str(record_dir),
        max_step=_positive(raw.get("max_step", DEFAULT_MAX_STEP), "max_step", where),
        max_lead=_positive(raw.get("max_lead", DEFAULT_MAX_LEAD), "max_lead", where),
        watchdog_s=_positive(raw.get("watchdog_s", DEFAULT_WATCHDOG_S), "watchdog_s", where),
    )


def parse_config(payload: Any) -> TeleopConfig:
    """Validate an already-loaded mapping and return the typed layout."""

    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a mapping")
    raw_leaders = payload.get("leaders")
    if not isinstance(raw_leaders, list) or not raw_leaders:
        raise ConfigError("configuration needs a non-empty 'leaders' list")
    raw_followers = payload.get("followers", [])
    if not isinstance(raw_followers, list):
        raise ConfigError("'followers' must be a list")

    leaders = tuple(_leader(item, index) for index, item in enumerate(raw_leaders))
    followers = tuple(_follower(item, index) for index, item in enumerate(raw_followers))

    identifiers: set[str] = set()
    for leader in leaders:
        if leader.id in identifiers:
            raise ConfigError(f"duplicate id {leader.id!r}")
        identifiers.add(leader.id)
    for follower in followers:
        if follower.id in identifiers:
            raise ConfigError(f"duplicate id {follower.id!r}")
        identifiers.add(follower.id)

    by_id = {leader.id: leader for leader in leaders}
    for follower in followers:
        if follower.leader not in by_id:
            raise ConfigError(f"follower {follower.id!r} names unknown leader {follower.leader!r}")

    ports: dict[int, str] = {}
    for leader in leaders:
        owner = ports.setdefault(leader.port, leader.id)
        if owner != leader.id:
            raise ConfigError(
                f"leaders {owner!r} and {leader.id!r} share port {leader.port}; "
                "the port is the group key, so each leader needs its own"
            )

    return TeleopConfig(leaders=leaders, followers=followers, _by_id=by_id)


def load_config(path: str | Path) -> TeleopConfig:
    """Read a YAML file and return the validated layout."""

    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"configuration file not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{source} is not valid YAML: {error}") from error
    return parse_config(payload)
