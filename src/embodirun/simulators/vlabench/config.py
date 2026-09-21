"""Configuration for one VLABench environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VLABenchConfig:
    simulator_id: str
    task: str
    max_episode_steps: int = 500
    width: int = 480
    height: int = 480
    viewer: bool = False

    @classmethod
    def from_mapping(
        cls,
        simulator_id: str,
        value: Mapping[str, Any],
    ) -> VLABenchConfig:
        options = dict(value)
        allowed = {"task", "max_episode_steps", "width", "height", "viewer"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError("unknown VLABench configuration fields: " + ", ".join(unknown))
        return cls(
            simulator_id=simulator_id,
            task=options.get("task"),
            max_episode_steps=options.get("max_episode_steps", 500),
            width=options.get("width", 480),
            height=options.get("height", 480),
            viewer=options.get("viewer", False),
        )

    def __post_init__(self) -> None:
        for name in ("simulator_id", "task"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"VLABench {name} must not be empty")
            object.__setattr__(self, name, value.strip())
        for name in ("max_episode_steps", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"VLABench {name} must be a positive integer")
        if not isinstance(self.viewer, bool):
            raise TypeError("VLABench viewer must be a boolean")


__all__ = ["VLABenchConfig"]
