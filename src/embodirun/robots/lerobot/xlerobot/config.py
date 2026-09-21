"""Configuration for an XLeRobot external-owner proxy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

Scope = Literal["arms", "base"]


@dataclass(frozen=True, slots=True)
class XLeRobotConfig:
    url: str
    token: str
    scope: Scope = "arms"
    timeout_s: float = 2.0
    robot_id: str = "xlerobot"

    @classmethod
    def from_mapping(cls, robot_id: str, value: Mapping[str, Any]) -> XLeRobotConfig:
        options = dict(value)
        unknown = sorted(set(options) - {"url", "token", "scope", "timeout_s"})
        if unknown:
            raise ValueError("unknown XLeRobot configuration fields: " + ", ".join(unknown))
        return cls(
            robot_id=robot_id,
            url=options.get("url"),
            token=options.get("token"),
            scope=options.get("scope", "arms"),
            timeout_s=options.get("timeout_s", 2.0),
        )

    def __post_init__(self) -> None:
        parsed = urlparse(self.url) if isinstance(self.url, str) else None
        if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an HTTP(S) URL")
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("token must not be empty")
        if self.scope not in ("arms", "base"):
            raise ValueError("XLeRobot scope must be 'arms' or 'base'")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
