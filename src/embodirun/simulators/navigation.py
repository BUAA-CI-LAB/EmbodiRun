"""Shared adapter logic for visual discrete-navigation simulators."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol

from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.robots.unitree.go2.navigation.discrete import NavigationCommand
from embodirun.types import Metadata

from .adapter import SimulationStep, SimulatorAdapter, SimulatorObservation
from .viewer import CameraViewer

NAVIGATION_IMAGE_FIELD = "observation.images.rgb"


@dataclass(frozen=True, slots=True)
class NavigationObservation:
    """One engine-native RGB frame and metric agent pose."""

    rgb: Any
    position: Sequence[float]
    rotation: Sequence[float]
    instruction: str
    distance_to_goal_m: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(self, "rotation", _vector(self.rotation, 4, "rotation"))
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("navigation instruction must not be empty")
        object.__setattr__(self, "instruction", self.instruction.strip())
        if self.distance_to_goal_m is not None:
            distance = _finite_number(self.distance_to_goal_m, "distance_to_goal_m")
            if distance < 0:
                raise ValueError("distance_to_goal_m must be non-negative")
            object.__setattr__(self, "distance_to_goal_m", distance)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("navigation observation metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class NavigationTransition:
    """One normalized transition returned by a navigation engine."""

    observation: NavigationObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Metadata = field(default_factory=dict)


class NavigationEnvironment(Protocol):
    """Small SDK boundary implemented by Habitat and Isaac backends."""

    def reset(self, *, seed: int | None) -> NavigationObservation: ...

    def step(self, command: NavigationCommand) -> NavigationTransition: ...

    def close(self) -> None: ...


class NavigationSimulatorAdapter(SimulatorAdapter):
    """Translate a visual navigation engine into Deploy's simulator contract."""

    def __init__(
        self,
        config: Any,
        *,
        environment_factory: Callable[[Any], NavigationEnvironment],
        engine_name: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(environment_factory):
            raise TypeError("environment_factory must be callable")
        if not isinstance(engine_name, str) or not engine_name.strip():
            raise ValueError("engine_name must not be empty")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.config = config
        self._environment_factory = environment_factory
        self._engine_name = engine_name
        self._clock = clock
        self._environment: NavigationEnvironment | None = None
        self._viewer = CameraViewer(f"{engine_name} — {config.simulator_id}") if config.viewer else None
        self._step_index = 0
        self._closed = False

    @property
    def simulator_id(self) -> str:
        return self.config.simulator_id

    def reset(
        self,
        *,
        task: str | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SimulatorObservation:
        self._ensure_open()
        if task is not None and task.strip() != self.config.task:
            raise ValueError(f"adapter is bound to {self._engine_name} task {self.config.task!r}, not {task!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError(f"{self._engine_name} seed must be an integer or None")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError(f"{self._engine_name} reset options must be a mapping or None")
        if options:
            raise ValueError(f"{self._engine_name} reset options are not supported")

        if self._environment is None:
            self._environment = self._environment_factory(self.config)
        self._step_index = 0
        return self._normalize(
            self._environment.reset(seed=seed),
            seed=seed,
            step_index=0,
        )

    def step(self, action: RobotAction) -> SimulationStep:
        self._ensure_open()
        if self._environment is None:
            raise RuntimeError(f"{self._engine_name} adapter must be reset before step()")
        command = NavigationCommand.from_robot_action(action)
        transition = self._environment.step(command)
        if not isinstance(transition, NavigationTransition):
            raise TypeError(f"{self._engine_name} environment returned an invalid transition")
        self._step_index += 1
        truncated = transition.truncated or self._step_index >= self.config.max_episode_steps
        return SimulationStep(
            observation=self._normalize(
                transition.observation,
                seed=None,
                step_index=self._step_index,
            ),
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=truncated,
            info=transition.info,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._environment is not None:
                self._environment.close()
        finally:
            if self._viewer is not None:
                self._viewer.close()
            self._environment = None
            self._closed = True

    def _normalize(
        self,
        value: NavigationObservation,
        *,
        seed: int | None,
        step_index: int,
    ) -> SimulatorObservation:
        if not isinstance(value, NavigationObservation):
            raise TypeError(f"{self._engine_name} environment returned an invalid observation")
        rgb = _rgb_array(value.rgb, engine_name=self._engine_name)
        if self._viewer is not None:
            self._viewer.show({"rgb": rgb})
        state: dict[str, Any] = {
            "position": list(value.position),
            "rotation": list(value.rotation),
        }
        if value.distance_to_goal_m is not None:
            state["distance_to_goal_m"] = value.distance_to_goal_m
        metadata = {
            "task": self.config.task,
            "instruction": value.instruction,
            "step_index": step_index,
            **dict(value.metadata),
        }
        if step_index == 0:
            metadata["seed"] = seed
        return SimulatorObservation(
            robot=RobotObservation(
                timestamp_s=float(self._clock()),
                values={"observation.navigation": state},
                metadata=metadata,
            ),
            frames=(_encode_rgb(rgb),),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self._engine_name} adapter is closed")


def _rgb_array(value: Any, *, engine_name: str) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(f"{engine_name} execution requires NumPy") from error
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"{engine_name} RGB observation must be an HWC image")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{engine_name} RGB observation must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{engine_name} RGB observation must be finite")
    return np.clip(array, 0, 255).astype(np.uint8)


def _encode_rgb(value: Any) -> CameraFrame:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("navigation image encoding requires OpenCV") from error
    encoded, payload = cv2.imencode(
        ".jpg",
        cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not encoded:
        raise RuntimeError("navigation RGB frame could not be JPEG encoded")
    return CameraFrame(
        name=NAVIGATION_IMAGE_FIELD,
        mime_type="image/jpeg",
        data=payload.tobytes(),
    )


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        tolist = getattr(value, "tolist", None)
        value = tolist() if callable(tolist) else value
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"navigation {name} must be an array")
    if len(value) != size:
        raise ValueError(f"navigation {name} must contain {size} values")
    return tuple(_finite_number(item, name) for item in value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"navigation {name} must contain numeric values")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"navigation {name} must contain finite values")
    return result


__all__ = [
    "NAVIGATION_IMAGE_FIELD",
    "NavigationEnvironment",
    "NavigationObservation",
    "NavigationSimulatorAdapter",
    "NavigationTransition",
]
