"""Configuration for one LIBERO task environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LIBERO_SUITES = frozenset(
    {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    }
)


@dataclass(frozen=True, slots=True)
class LiberoConfig:
    simulator_id: str
    suite: str
    task_id: int
    max_episode_steps: int = 500
    width: int = 256
    height: int = 256
    viewer: bool = False

    @classmethod
    def from_mapping(
        cls,
        simulator_id: str,
        value: Mapping[str, Any],
    ) -> LiberoConfig:
        options = dict(value)
        allowed = {
            "suite",
            "task_id",
            "max_episode_steps",
            "width",
            "height",
            "viewer",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError("unknown LIBERO configuration fields: " + ", ".join(unknown))
        return cls(
            simulator_id=simulator_id,
            suite=options.get("suite"),
            task_id=options.get("task_id"),
            max_episode_steps=options.get("max_episode_steps", 500),
            width=options.get("width", 256),
            height=options.get("height", 256),
            viewer=options.get("viewer", False),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.simulator_id, str) or not self.simulator_id.strip():
            raise ValueError("LIBERO simulator_id must not be empty")
        object.__setattr__(self, "simulator_id", self.simulator_id.strip())
        if not isinstance(self.suite, str) or self.suite not in LIBERO_SUITES:
            available = ", ".join(sorted(LIBERO_SUITES))
            raise ValueError(f"LIBERO suite must be one of: {available}")
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int) or self.task_id < 0:
            raise ValueError("LIBERO task_id must be a non-negative integer")
        for name in ("max_episode_steps", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"LIBERO {name} must be a positive integer")
        if not isinstance(self.viewer, bool):
            raise TypeError("LIBERO viewer must be a boolean")

    @property
    def task(self) -> str:
        return f"{self.suite}:{self.task_id}"


__all__ = ["LIBERO_SUITES", "LiberoConfig"]
