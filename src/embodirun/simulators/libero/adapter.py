"""LIBERO implementation of the simulator adapter contract."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from embodirun.robots import RobotAction, RobotObservation
from embodirun.robots.sensors.cameras import CameraFrame

from ..adapter import SimulationStep, SimulatorAdapter, SimulatorObservation
from ..viewer import CameraViewer
from .config import LiberoConfig

LIBERO_ACTION_DIM = 7
_CAMERA_MAPPING = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "wrist_image",
}


class LiberoAdapter(SimulatorAdapter):
    """Expose one deterministic LIBERO suite task through Deploy's contract."""

    def __init__(
        self,
        config: LiberoConfig,
        *,
        environment_factory: Callable[[LiberoConfig], Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, LiberoConfig):
            raise TypeError("config must be a LiberoConfig")
        if environment_factory is not None and not callable(environment_factory):
            raise TypeError("environment_factory must be callable or None")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.config = config
        self._environment_factory = environment_factory or make_libero_environment
        self._clock = clock
        self._environment: Any | None = None
        self._viewer = CameraViewer(f"LIBERO — {config.simulator_id}") if config.viewer else None
        self._instruction: str | None = None
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
            raise ValueError(f"adapter is bound to LIBERO task {self.config.task!r}, not {task!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("LIBERO seed must be an integer or None")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("LIBERO reset options must be a mapping or None")
        if options:
            raise ValueError("LIBERO reset options are not supported")

        environment = self._get_or_create_environment()
        raw_observation, reset_info = _unpack_reset(environment.reset(seed=seed))
        instruction = getattr(environment, "task_description", None)
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError("LIBERO environment did not provide a task description")
        self._instruction = instruction.strip()
        self._step_index = 0
        return self._normalize_observation(
            raw_observation,
            metadata={
                "suite": self.config.suite,
                "task_id": self.config.task_id,
                "task": self.config.task,
                "instruction": self._instruction,
                "seed": seed,
                "reset_info": dict(reset_info),
                "step_index": 0,
            },
        )

    def step(self, action: RobotAction) -> SimulationStep:
        self._ensure_open()
        if self._environment is None or self._instruction is None:
            raise RuntimeError("LIBERO adapter must be reset before step()")
        if not isinstance(action, RobotAction):
            raise TypeError("LIBERO action must be a RobotAction")

        action_array = _action_array(action.values)
        raw_observation, reward, terminated, truncated, info = _unpack_step(self._environment.step(action_array))
        self._step_index += 1
        observation = self._normalize_observation(
            raw_observation,
            metadata={
                "suite": self.config.suite,
                "task_id": self.config.task_id,
                "task": self.config.task,
                "instruction": self._instruction,
                "step_index": self._step_index,
            },
        )
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
            self._environment = self._environment_factory(self.config)
        return self._environment

    def _normalize_observation(
        self,
        values: Any,
        *,
        metadata: Mapping[str, Any],
    ) -> SimulatorObservation:
        root = _mapping(values, "observation")
        raw_pixels = _mapping(root.get("pixels"), "observation.pixels")
        pixels = {name: _policy_image(value, name=name) for name, value in raw_pixels.items()}
        if set(pixels) != {"image", "wrist_image"}:
            raise ValueError("LIBERO observation must contain image and wrist_image cameras")
        if self._viewer is not None:
            self._viewer.show(pixels)

        robot = _mapping(root.get("robot_state"), "observation.robot_state")
        eef = _mapping(robot.get("eef"), "observation.robot_state.eef")
        gripper = _mapping(robot.get("gripper"), "observation.robot_state.gripper")
        position = _vector(eef.get("pos"), size=3, name="eef.pos")
        quaternion = _vector(eef.get("quat"), size=4, name="eef.quat")
        gripper_position = _vector(gripper.get("qpos"), size=2, name="gripper.qpos")
        state = [
            *position,
            *_quaternion_xyzw_to_axis_angle(quaternion),
            *gripper_position,
        ]
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
            raise RuntimeError("LIBERO adapter is closed")


def make_libero_environment(config: LiberoConfig) -> Any:
    """Construct the LeRobot wrapper without importing it in the host process."""

    _prepare_libero_import()
    try:
        from lerobot.envs.libero import LiberoEnv
        from libero.libero import benchmark
    except ImportError as error:
        raise RuntimeError("LIBERO execution requires the isolated sim-libero environment") from error

    suite_type = benchmark.get_benchmark_dict().get(config.suite)
    if suite_type is None:
        raise RuntimeError(f"LIBERO suite {config.suite!r} is not installed")
    suite = suite_type()
    task_count = len(suite.tasks)
    if config.task_id >= task_count:
        raise ValueError(f"LIBERO task_id {config.task_id} is outside suite {config.suite!r} with {task_count} tasks")
    return LiberoEnv(
        task_suite=suite,
        task_id=config.task_id,
        task_suite_name=config.suite,
        episode_length=config.max_episode_steps,
        camera_name=tuple(_CAMERA_MAPPING),
        camera_name_mapping=dict(_CAMERA_MAPPING),
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=config.width,
        observation_height=config.height,
        init_states=True,
        episode_index=0,
        n_envs=1,
        num_steps_wait=10,
        control_freq=20,
        control_mode="relative",
        hard_reset=True,
    )


def _prepare_libero_import() -> None:
    """Isolate hf-libero's path file and keep package import non-interactive."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    configured_root = os.environ.get("LIBERO_CONFIG_PATH")
    config_root = Path(configured_root or Path.home() / ".cache" / "embodirun" / "libero").expanduser()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    config_file = config_root / "config.yaml"
    if configured_root is not None and config_file.is_file():
        return

    package = find_spec("libero")
    locations = () if package is None else package.submodule_search_locations
    if not locations:
        raise RuntimeError("LIBERO execution requires the isolated sim-libero environment")
    benchmark_root = Path(next(iter(locations))) / "libero"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(config_root / "datasets"),
        "assets": str(Path.home() / ".cache" / "libero" / "assets"),
    }
    config_root.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_root,
            prefix=".config.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(json.dumps(paths, indent=2))
            temporary.write("\n")
            temporary_name = temporary.name
        os.replace(temporary_name, config_file)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _action_array(values: Any) -> Any:
    raw = values.get("action") if isinstance(values, Mapping) and "action" in values else values
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("LIBERO execution requires NumPy") from error
    action = np.asarray(raw, dtype=np.float32)
    if action.shape != (LIBERO_ACTION_DIM,):
        raise ValueError(f"LIBERO action must have shape ({LIBERO_ACTION_DIM},), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("LIBERO action must contain only finite values")
    return np.clip(action, -1.0, 1.0)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"LIBERO {name} must be an object with string keys")
    return dict(value)


def _vector(value: Any, *, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        tolist = getattr(value, "tolist", None)
        value = tolist() if callable(tolist) else value
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"LIBERO {name} must be an array")
    if len(value) != size:
        raise ValueError(f"LIBERO {name} must contain {size} values")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"LIBERO {name} must contain numeric values")
        try:
            number = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"LIBERO {name} must contain numeric values") from None
        if not math.isfinite(number):
            raise ValueError(f"LIBERO {name} must contain finite values")
        result.append(number)
    return result


def _quaternion_xyzw_to_axis_angle(quaternion: Sequence[float]) -> list[float]:
    x, y, z, w = quaternion
    w = max(-1.0, min(1.0, w))
    denominator = math.sqrt(max(0.0, 1.0 - w * w))
    if denominator <= 1e-10:
        return [0.0, 0.0, 0.0]
    scale = 2.0 * math.acos(w) / denominator
    return [x * scale, y * scale, z * scale]


def _policy_image(value: Any, *, name: str) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("LIBERO execution requires NumPy") from error
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"LIBERO camera {name!r} must return an HWC RGB image")
    return np.ascontiguousarray(image[::-1, ::-1], dtype=np.uint8)


def _encode_frame(name: str, value: Any) -> CameraFrame:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("LIBERO image encoding requires OpenCV") from error
    encoded, payload = cv2.imencode(
        ".jpg",
        cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not encoded:
        raise RuntimeError(f"LIBERO camera {name!r} could not be JPEG encoded")
    return CameraFrame(name=name, mime_type="image/jpeg", data=payload.tobytes())


def _unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("LIBERO reset must return observation and info")
    observation, info = result
    if not isinstance(info, Mapping):
        raise TypeError("LIBERO reset info must be a mapping")
    return observation, info


def _unpack_step(
    result: Any,
) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError("LIBERO step must return five values")
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, Mapping):
        raise TypeError("LIBERO step info must be a mapping")
    return observation, float(reward), bool(terminated), bool(truncated), dict(info)


__all__ = ["LIBERO_ACTION_DIM", "LiberoAdapter", "make_libero_environment"]
