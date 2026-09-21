"""Lazy Habitat-Sim SDK integration for R2R navigation episodes."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodirun.robots.unitree.go2.navigation.discrete import (
    NavigationCommand,
    NavigationCommandKind,
)

from ..navigation import NavigationObservation, NavigationTransition
from .config import HabitatConfig


@dataclass(frozen=True, slots=True)
class HabitatEpisode:
    episode_id: str
    scene: Path
    instruction: str
    start_position: tuple[float, float, float]
    start_rotation_xyzw: tuple[float, float, float, float]
    goal_position: tuple[float, float, float]


class HabitatEnvironment:
    """Own one Habitat-Sim instance and its selected R2R episode."""

    def __init__(
        self,
        config: HabitatConfig,
        episode: HabitatEpisode,
        simulator: Any,
        habitat_sim: Any,
    ) -> None:
        self.config = config
        self.episode = episode
        self._simulator = simulator
        self._habitat_sim = habitat_sim
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Habitat execution requires NumPy") from error
        self._numpy = np
        self._closed = False

    def reset(self, *, seed: int | None) -> NavigationObservation:
        self._ensure_open()
        if seed is not None:
            self._simulator.seed(seed)
        self._simulator.reset()
        state = self._habitat_sim.AgentState()
        state.position = self._numpy.asarray(self.episode.start_position, dtype=self._numpy.float32)
        from habitat_sim.utils.common import quat_from_coeffs

        state.rotation = quat_from_coeffs(self._numpy.asarray(self.episode.start_rotation_xyzw))
        self._simulator.get_agent(0).set_state(state, reset_sensors=True)
        return self._observation(self._simulator.get_sensor_observations())

    def step(self, command: NavigationCommand) -> NavigationTransition:
        self._ensure_open()
        if command.kind is NavigationCommandKind.STOP:
            observation = self._observation(self._simulator.get_sensor_observations())
            success = (
                observation.distance_to_goal_m is not None
                and observation.distance_to_goal_m <= self.config.success_distance_m
            )
            return NavigationTransition(
                observation=observation,
                reward=1.0 if success else 0.0,
                terminated=True,
                info={
                    "success": success,
                    "distance_to_goal_m": observation.distance_to_goal_m,
                    "stop": True,
                },
            )

        action = {
            NavigationCommandKind.MOVE_FORWARD: "move_forward",
            NavigationCommandKind.TURN_LEFT: "turn_left",
            NavigationCommandKind.TURN_RIGHT: "turn_right",
        }[command.kind]
        raw = self._simulator.step(action)
        observation = self._observation(raw)
        return NavigationTransition(
            observation=observation,
            info={
                "collision": bool(getattr(self._simulator, "previous_step_collided", False)),
                "distance_to_goal_m": observation.distance_to_goal_m,
                "stop": False,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._simulator.close()
        self._closed = True

    def _observation(self, values: object) -> NavigationObservation:
        if not isinstance(values, Mapping):
            raise TypeError("Habitat sensor observation must be a mapping")
        if "rgb" not in values:
            raise ValueError("Habitat sensor observation is missing rgb")
        state = self._simulator.get_agent(0).get_state()
        from habitat_sim.utils.common import quat_to_coeffs

        position = _vector(state.position, 3, "agent position")
        rotation = _vector(quat_to_coeffs(state.rotation), 4, "agent rotation")
        return NavigationObservation(
            rgb=values["rgb"],
            position=position,
            rotation=rotation,
            instruction=self.episode.instruction,
            distance_to_goal_m=self._geodesic_distance(position),
            metadata={
                "episode_id": self.episode.episode_id,
                "scene": str(self.episode.scene),
            },
        )

    def _geodesic_distance(self, position: Sequence[float]) -> float | None:
        path = self._habitat_sim.ShortestPath()
        path.requested_start = self._numpy.asarray(position)
        path.requested_end = self._numpy.asarray(self.episode.goal_position)
        if not self._simulator.pathfinder.find_path(path):
            return None
        distance = float(path.geodesic_distance)
        return distance if math.isfinite(distance) and distance >= 0 else None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Habitat environment is closed")


def make_habitat_environment(config: HabitatConfig) -> HabitatEnvironment:
    """Load Habitat only inside its isolated simulator service."""

    try:
        import habitat_sim
    except ImportError as error:
        raise RuntimeError("Habitat execution requires the isolated sim-habitat environment") from error

    episode = load_r2r_episode(config)
    simulator_configuration = habitat_sim.SimulatorConfiguration()
    simulator_configuration.scene_id = str(episode.scene)
    simulator_configuration.gpu_device_id = config.gpu_device_id
    simulator_configuration.enable_physics = False

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [config.height, config.width]
    rgb.position = [0.0, config.camera_height_m, 0.0]
    rgb.hfov = config.hfov_deg

    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [rgb]
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
    }
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(simulator_configuration, [agent]))
    return HabitatEnvironment(config, episode, simulator, habitat_sim)


def load_r2r_episode(config: HabitatConfig) -> HabitatEpisode:
    """Read one R2R-VLNCE episode without importing Habitat-Lab."""

    if not config.dataset.is_file():
        raise FileNotFoundError(f"Habitat dataset does not exist: {config.dataset}")
    try:
        with gzip.open(config.dataset, mode="rt", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Habitat dataset {config.dataset}: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("episodes"), list):
        raise TypeError("Habitat dataset must contain an episodes list")
    raw = next(
        (
            item
            for item in payload["episodes"]
            if isinstance(item, Mapping) and str(item.get("episode_id")) == config.episode_id
        ),
        None,
    )
    if raw is None:
        raise ValueError(f"Habitat dataset has no episode {config.episode_id!r}")

    instruction_value = raw.get("instruction")
    if isinstance(instruction_value, Mapping):
        instruction_value = instruction_value.get("instruction_text")
    instruction = _text(instruction_value, "instruction")
    goals = raw.get("goals")
    if not isinstance(goals, list) or not goals or not isinstance(goals[0], Mapping):
        raise ValueError("Habitat episode must contain at least one goal")
    scene = _resolve_scene(config.scenes_dir, _text(raw.get("scene_id"), "scene_id"))
    return HabitatEpisode(
        episode_id=config.episode_id,
        scene=scene,
        instruction=instruction,
        start_position=_vector(raw.get("start_position"), 3, "start_position"),
        start_rotation_xyzw=_vector(raw.get("start_rotation"), 4, "start_rotation"),
        goal_position=_vector(goals[0].get("position"), 3, "goal.position"),
    )


def _resolve_scene(scenes_dir: Path, scene_id: str) -> Path:
    scene = Path(scene_id)
    candidates = [scene] if scene.is_absolute() else [scenes_dir / scene]
    marker = "scene_datasets/"
    if marker in scene_id:
        candidates.append(scenes_dir / scene_id.split(marker, 1)[1])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Habitat scene {scene_id!r} was not found below {scenes_dir}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Habitat episode {name} must not be empty")
    return value.strip()


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        tolist = getattr(value, "tolist", None)
        value = tolist() if callable(tolist) else value
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Habitat episode {name} must be an array")
    if len(value) != size:
        raise ValueError(f"Habitat episode {name} must contain {size} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"Habitat episode {name} must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError):
            raise TypeError(f"Habitat episode {name} must be numeric") from None
        if not math.isfinite(number):
            raise ValueError(f"Habitat episode {name} must be finite")
        result.append(number)
    return tuple(result)


__all__ = [
    "HabitatEnvironment",
    "HabitatEpisode",
    "load_r2r_episode",
    "make_habitat_environment",
]
