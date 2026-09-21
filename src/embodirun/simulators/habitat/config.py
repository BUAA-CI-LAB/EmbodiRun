"""Configuration for one R2R-VLNCE episode in Habitat-Sim."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class HabitatConfig:
    simulator_id: str
    dataset: Path
    scenes_dir: Path
    episode_id: str
    max_episode_steps: int = 500
    success_distance_m: float = 3.0
    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0
    camera_height_m: float = 1.25
    gpu_device_id: int = 0
    viewer: bool = False

    @property
    def task(self) -> str:
        return self.episode_id

    @classmethod
    def from_mapping(
        cls,
        simulator_id: str,
        value: Mapping[str, Any],
    ) -> HabitatConfig:
        options = dict(value)
        allowed = {
            "dataset",
            "scenes_dir",
            "episode_id",
            "max_episode_steps",
            "success_distance_m",
            "width",
            "height",
            "hfov_deg",
            "camera_height_m",
            "gpu_device_id",
            "viewer",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError("unknown Habitat configuration fields: " + ", ".join(unknown))
        return cls(
            simulator_id=simulator_id,
            dataset=Path(_text(options.get("dataset"), "dataset")).expanduser(),
            scenes_dir=Path(_text(options.get("scenes_dir"), "scenes_dir")).expanduser(),
            episode_id=_text(options.get("episode_id"), "episode_id"),
            max_episode_steps=_positive_integer(options.get("max_episode_steps", 500), "max_episode_steps"),
            success_distance_m=_positive_number(options.get("success_distance_m", 3.0), "success_distance_m"),
            width=_positive_integer(options.get("width", 640), "width"),
            height=_positive_integer(options.get("height", 480), "height"),
            hfov_deg=_angle(options.get("hfov_deg", 79.0), "hfov_deg"),
            camera_height_m=_positive_number(options.get("camera_height_m", 1.25), "camera_height_m"),
            gpu_device_id=_gpu_device_id(options.get("gpu_device_id", 0), "gpu_device_id"),
            viewer=_boolean(options.get("viewer", False), "viewer"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.simulator_id, str) or not self.simulator_id.strip():
            raise ValueError("Habitat simulator_id must not be empty")
        object.__setattr__(self, "simulator_id", self.simulator_id.strip())
        for name in ("dataset", "scenes_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"Habitat {name} must be a path")
        object.__setattr__(self, "episode_id", _text(self.episode_id, "episode_id"))
        for name in ("max_episode_steps", "width", "height"):
            _positive_integer(getattr(self, name), name)
        _positive_number(self.success_distance_m, "success_distance_m")
        _angle(self.hfov_deg, "hfov_deg")
        _positive_number(self.camera_height_m, "camera_height_m")
        _gpu_device_id(self.gpu_device_id, "gpu_device_id")
        _boolean(self.viewer, "viewer")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Habitat {name} must not be empty")
    return value.strip()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Habitat {name} must be a positive integer")
    return value


def _gpu_device_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError(f"Habitat {name} must be an integer greater than or equal to -1")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Habitat {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Habitat {name} must be finite and positive")
    return result


def _angle(value: object, name: str) -> float:
    result = _positive_number(value, name)
    if result >= 180:
        raise ValueError(f"Habitat {name} must be less than 180 degrees")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Habitat {name} must be a boolean")
    return value


__all__ = ["HabitatConfig"]
