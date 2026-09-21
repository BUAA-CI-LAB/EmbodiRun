"""Configuration for one visual-navigation task in Isaac Sim."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IsaacConfig:
    simulator_id: str
    task: str
    instruction: str
    goal_position: tuple[float, float, float]
    scene: str = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    start_position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    start_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    max_episode_steps: int = 500
    success_distance_m: float = 3.0
    width: int = 640
    height: int = 480
    camera_translation: tuple[float, float, float] = (0.30, 0.0, 0.18)
    forward_speed_mps: float = 0.5
    turn_speed_rad_s: float = 0.6
    action_timeout_s: float = 3.0
    settle_steps: int = 10
    device: str = "cuda"
    viewer: bool = False

    @classmethod
    def from_mapping(
        cls,
        simulator_id: str,
        value: Mapping[str, Any],
    ) -> IsaacConfig:
        options = dict(value)
        allowed = {
            "task",
            "instruction",
            "goal_position",
            "scene",
            "start_position",
            "start_orientation_wxyz",
            "max_episode_steps",
            "success_distance_m",
            "width",
            "height",
            "camera_translation",
            "forward_speed_mps",
            "turn_speed_rad_s",
            "action_timeout_s",
            "settle_steps",
            "device",
            "viewer",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError("unknown Isaac configuration fields: " + ", ".join(unknown))
        return cls(
            simulator_id=simulator_id,
            task=_text(options.get("task"), "task"),
            instruction=_text(options.get("instruction"), "instruction"),
            goal_position=_vector(options.get("goal_position"), 3, "goal_position"),
            scene=_text(
                options.get("scene", "/Isaac/Environments/Simple_Warehouse/warehouse.usd"),
                "scene",
            ),
            start_position=_vector(
                options.get("start_position", (0.0, 0.0, 0.5)),
                3,
                "start_position",
            ),
            start_orientation_wxyz=_normalized_quaternion(options.get("start_orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
            max_episode_steps=_positive_integer(options.get("max_episode_steps", 500), "max_episode_steps"),
            success_distance_m=_positive_number(options.get("success_distance_m", 3.0), "success_distance_m"),
            width=_positive_integer(options.get("width", 640), "width"),
            height=_positive_integer(options.get("height", 480), "height"),
            camera_translation=_vector(
                options.get("camera_translation", (0.30, 0.0, 0.18)),
                3,
                "camera_translation",
            ),
            forward_speed_mps=_positive_number(options.get("forward_speed_mps", 0.5), "forward_speed_mps"),
            turn_speed_rad_s=_positive_number(options.get("turn_speed_rad_s", 0.6), "turn_speed_rad_s"),
            action_timeout_s=_positive_number(options.get("action_timeout_s", 3.0), "action_timeout_s"),
            settle_steps=_non_negative_integer(options.get("settle_steps", 10), "settle_steps"),
            device=_device(options.get("device", "cuda")),
            viewer=_boolean(options.get("viewer", False), "viewer"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.simulator_id, str) or not self.simulator_id.strip():
            raise ValueError("Isaac simulator_id must not be empty")
        object.__setattr__(self, "simulator_id", self.simulator_id.strip())
        for name in ("task", "instruction", "scene"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "goal_position", _vector(self.goal_position, 3, "goal_position"))
        object.__setattr__(self, "start_position", _vector(self.start_position, 3, "start_position"))
        object.__setattr__(
            self,
            "start_orientation_wxyz",
            _normalized_quaternion(self.start_orientation_wxyz),
        )
        object.__setattr__(
            self,
            "camera_translation",
            _vector(self.camera_translation, 3, "camera_translation"),
        )
        for name in ("max_episode_steps", "width", "height"):
            _positive_integer(getattr(self, name), name)
        _positive_number(self.success_distance_m, "success_distance_m")
        _positive_number(self.forward_speed_mps, "forward_speed_mps")
        _positive_number(self.turn_speed_rad_s, "turn_speed_rad_s")
        _positive_number(self.action_timeout_s, "action_timeout_s")
        _non_negative_integer(self.settle_steps, "settle_steps")
        _device(self.device)
        _boolean(self.viewer, "viewer")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Isaac {name} must not be empty")
    return value.strip()


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Isaac {name} must be an array")
    if len(value) != size:
        raise ValueError(f"Isaac {name} must contain {size} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"Isaac {name} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"Isaac {name} must be finite")
        result.append(number)
    return tuple(result)


def _normalized_quaternion(value: object) -> tuple[float, float, float, float]:
    quaternion = _vector(value, 4, "start_orientation_wxyz")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-8:
        raise ValueError("Isaac start_orientation_wxyz must not be zero")
    return tuple(item / norm for item in quaternion)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Isaac {name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Isaac {name} must be a non-negative integer")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Isaac {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Isaac {name} must be finite and positive")
    return result


def _device(value: object) -> str:
    if value not in {"cpu", "cuda"}:
        raise ValueError("Isaac device must be cpu or cuda")
    return str(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Isaac {name} must be a boolean")
    return value


__all__ = ["IsaacConfig"]
