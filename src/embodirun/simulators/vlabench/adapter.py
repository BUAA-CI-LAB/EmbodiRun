"""VLABench implementation of the simulator adapter contract.

The module deliberately keeps LeRobot, VLABench, NumPy, and MuJoCo behind the
runtime boundary.  Importing :mod:`embodirun.simulators` therefore does
not require the dedicated simulation environment.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from typing import Any

from embodirun.robots.adapter import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

from ..adapter import SimulationStep, SimulatorAdapter, SimulatorObservation
from ..viewer import CameraViewer
from .camera import make_semantic_vlabench_environment
from .config import VLABenchConfig

VLABENCH_ACTION_DIM = 7


class VLABenchAdapter(SimulatorAdapter):
    """Expose one LeRobot VLABench environment through Deploy's contract."""

    def __init__(
        self,
        config: VLABenchConfig,
        *,
        environment_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, VLABenchConfig):
            raise TypeError("config must be a VLABenchConfig")
        if environment_factory is not None and not callable(environment_factory):
            raise TypeError("environment_factory must be callable or None")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.config = config
        self._environment_factory = environment_factory or make_semantic_vlabench_environment
        self._clock = clock
        self._environment: Any | None = None
        self._viewer = CameraViewer(f"VLABench — {config.simulator_id}") if config.viewer else None
        self._last_observation: SimulatorObservation | None = None
        self._instruction: str | None = None
        self._closed = False
        self._step_index = 0

    @property
    def simulator_id(self) -> str:
        return self.config.simulator_id

    @property
    def instruction(self) -> str:
        if self._instruction is None:
            raise RuntimeError("VLABench adapter must be reset before reading its instruction")
        return self._instruction

    def reset(
        self,
        *,
        task: str | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SimulatorObservation:
        self._ensure_open()
        if task is not None and task.strip() != self.config.task:
            raise ValueError(f"adapter is bound to VLABench task {self.config.task!r}, not {task!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("VLABench seed must be an integer or None")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("VLABench reset options must be a mapping or None")

        self._seed_process(seed)
        environment = self._get_or_create_environment()
        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options:
            reset_kwargs["options"] = dict(options)
        raw_observation, reset_info = _unpack_reset(environment.reset(**reset_kwargs))

        instruction = _extract_instruction(environment, fallback=self.config.task)
        if hasattr(environment, "task_description"):
            environment.task_description = instruction
        self._instruction = instruction
        self._step_index = 0
        observation = self._normalize_observation(
            raw_observation,
            metadata={
                "task": self.config.task,
                "instruction": instruction,
                "seed": seed,
                "reset_info": dict(reset_info),
                "step_index": 0,
            },
        )
        self._last_observation = observation
        return observation

    def step(self, action: RobotAction) -> SimulationStep:
        self._ensure_open()
        if self._environment is None or self._last_observation is None:
            raise RuntimeError("VLABench endpoint must be reset before step()")
        if not isinstance(action, RobotAction):
            raise TypeError("VLABench action must be a RobotAction")

        action_array = _action_array(action.values)
        raw_observation, reward, terminated, truncated, info = _unpack_step(self._environment.step(action_array))
        self._step_index += 1
        observation = self._normalize_observation(
            raw_observation,
            metadata={
                "task": self.config.task,
                "instruction": self.instruction,
                "step_index": self._step_index,
            },
        )
        self._last_observation = observation
        return SimulationStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=(truncated or self._step_index >= self.config.max_episode_steps),
            info=info,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._environment is not None:
                close = getattr(self._environment, "close", None)
                if callable(close):
                    close()
        finally:
            if self._viewer is not None:
                self._viewer.close()
            self._environment = None
            self._closed = True

    def _get_or_create_environment(self) -> Any:
        if self._environment is None:
            self._environment = self._environment_factory(
                task=self.config.task,
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                render_resolution=(self.config.height, self.config.width),
                robot="franka",
                max_episode_steps=self.config.max_episode_steps,
                action_mode="eef",
            )
        return self._environment

    def _seed_process(self, seed: int | None) -> None:
        if seed is None:
            return
        # VLABench samples layouts through module-level RNGs before the inner
        # dm-control reset receives the seed.
        random.seed(seed)
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("VLABench execution requires NumPy in the isolated simulator environment") from error
        np.random.seed(seed)

    def _normalize_observation(
        self,
        values: Any,
        *,
        metadata: Mapping[str, Any],
    ) -> SimulatorObservation:
        if not isinstance(values, Mapping):
            raise TypeError("VLABench observation must be a mapping")
        normalized = dict(values)
        pixels = normalized.pop("pixels", None)
        if not isinstance(pixels, Mapping) or not pixels:
            raise TypeError("VLABench observation pixels must be a non-empty mapping")
        if self._viewer is not None:
            self._viewer.show(pixels)
        if "agent_pos" not in normalized:
            raise TypeError("VLABench observation must contain agent_pos")
        state = _json_values(normalized.pop("agent_pos"), name="agent_pos")
        if normalized:
            metadata = {**dict(metadata), "environment_values": normalized}
        return SimulatorObservation(
            robot=RobotObservation(
                timestamp_s=float(self._clock()),
                values={"observation.state": state},
                metadata=dict(metadata),
            ),
            frames=tuple(_encode_frame(f"observation.images.{name}", value) for name, value in pixels.items()),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VLABench endpoint is closed")


def _encode_frame(name: str, value: Any) -> CameraFrame:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("VLABench image encoding requires NumPy and OpenCV") from error
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"VLABench camera {name!r} must return an HWC RGB image")
    bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    encoded, payload = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not encoded:
        raise RuntimeError(f"VLABench camera {name!r} could not be JPEG encoded")
    return CameraFrame(name=name, mime_type="image/jpeg", data=payload.tobytes())


def _json_values(value: Any, *, name: str) -> Any:
    tolist = getattr(value, "tolist", None)
    result = tolist() if callable(tolist) else value
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"VLABench {name} must be an array")
    return list(result)


def _extract_instruction(environment: Any, *, fallback: str) -> str:
    task_object = getattr(getattr(environment, "_env", None), "task", None)
    get_instruction = getattr(task_object, "get_instruction", None)
    if callable(get_instruction):
        instruction = get_instruction()
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
        if isinstance(instruction, (tuple, list)):
            for candidate in instruction:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    return fallback


def _action_array(values: Any) -> Any:
    raw = values.get("action") if isinstance(values, Mapping) and "action" in values else values
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("VLABench execution requires NumPy") from error
    action = np.asarray(raw, dtype=np.float32)
    if action.shape != (VLABENCH_ACTION_DIM,):
        raise ValueError(f"VLABench action must have shape ({VLABENCH_ACTION_DIM},), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("VLABench action must contain only finite values")
    return action


def _unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        if not isinstance(info, Mapping):
            raise TypeError("VLABench reset info must be a mapping")
        return observation, info
    return result, {}


def _unpack_step(
    result: Any,
) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError("VLABench step must return five values")
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, Mapping):
        raise TypeError("VLABench step info must be a mapping")
    return observation, float(reward), bool(terminated), bool(truncated), dict(info)


__all__ = ["VLABENCH_ACTION_DIM", "VLABenchAdapter"]
