"""Dataset-aligned camera observations for VLABench.

LeRobot 0.6 assigns the first three rendered VLABench cameras to ``image``,
``second_image``, and ``wrist_image`` by position.  The compiled VLABench model
currently orders them as right, left, forward, and wrist, so that positional
mapping silently feeds a forward view where the checkpoint expects the wrist
view.  This module resolves the three dataset views from compiled camera names
instead.

Heavy simulator dependencies stay behind :func:`make_semantic_vlabench_environment`.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

VLABENCH_DATASET_CAMERA_KEYS = ("image", "second_image", "wrist_image")


class VLABenchCameraMappingError(RuntimeError):
    """Raised when the compiled model cannot provide the three dataset cameras."""


def resolve_vlabench_camera_indices(camera_names: list[str | None]) -> dict[str, int]:
    """Map dataset image keys to compiled camera indices by semantic name.

    ``image`` is the left scene camera, ``second_image`` is the right scene
    camera, and ``wrist_image`` is the camera whose compiled name ends with
    ``Franka_wrist_cam``.
    """

    matches: dict[str, list[int]] = {
        "image": [],
        "second_image": [],
        "wrist_image": [],
    }
    for index, raw_name in enumerate(camera_names):
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        name = raw_name.strip().casefold()
        if _has_component_suffix(name, "left"):
            matches["image"].append(index)
        if _has_component_suffix(name, "right"):
            matches["second_image"].append(index)
        if name.endswith("franka_wrist_cam"):
            matches["wrist_image"].append(index)

    invalid = {key: indices for key, indices in matches.items() if len(indices) != 1}
    if invalid:
        details = ", ".join(f"{key}={indices}" for key, indices in invalid.items())
        raise VLABenchCameraMappingError(
            "expected exactly one compiled camera for each dataset view "
            f"({details}); available cameras={camera_names!r}"
        )
    return {key: indices[0] for key, indices in matches.items()}


def compiled_camera_names(model: Any) -> list[str | None]:
    """Read camera names in render-index order from a compiled MuJoCo model."""

    try:
        camera_count = int(model.ncam)
    except (AttributeError, TypeError, ValueError) as error:
        raise VLABenchCameraMappingError("compiled VLABench model does not expose a valid ncam") from error
    id2name = getattr(model, "id2name", None)
    if not callable(id2name):
        raise VLABenchCameraMappingError("compiled VLABench model does not expose id2name()")
    try:
        return [id2name(index, "camera") for index in range(camera_count)]
    except (TypeError, ValueError) as error:
        raise VLABenchCameraMappingError("failed to read camera names from the compiled VLABench model") from error


class VLABenchSemanticCameraMixin:
    """Override LeRobot VLABench observation capture with dataset camera semantics."""

    _env: Any
    _robot_base_xyz: Any
    obs_type: str
    render_resolution: tuple[int, int]

    def _get_obs(self) -> dict[str, Any]:
        """Render only the left, right, and Franka wrist cameras by name."""

        if self._env is None:
            raise RuntimeError("VLABench environment must be initialized before observation")

        import numpy as np

        physics = self._env.physics
        camera_names = compiled_camera_names(physics.model)
        camera_indices = resolve_vlabench_camera_indices(camera_names)
        height, width = self.render_resolution
        images = {
            key: _to_hwc3(
                self._env.render(
                    camera_id=camera_index,
                    height=height,
                    width=width,
                ),
                height=height,
                width=width,
            )
            for key, camera_index in camera_indices.items()
        }

        raw = np.asarray(
            self._env.robot.get_ee_state(physics),
            dtype=np.float64,
        ).ravel()
        pos_world = raw[:3] if raw.size >= 3 else np.zeros(3, dtype=np.float64)
        quat_wxyz = raw[3:7] if raw.size >= 7 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        gripper = float(raw[7]) if raw.size >= 8 else 0.0
        base = self._robot_base_xyz if self._robot_base_xyz is not None else np.zeros(3, dtype=np.float64)
        pos_robot = pos_world - base
        euler_xyz = _quaternion_wxyz_to_euler_xyz(quat_wxyz)
        agent_pos = np.concatenate([pos_robot, euler_xyz, [gripper]]).astype(np.float64)

        if self.obs_type == "pixels":
            return {"pixels": images}
        if self.obs_type == "pixels_agent_pos":
            return {"pixels": images, "agent_pos": agent_pos}
        raise ValueError(f"Unknown VLABench observation type: {self.obs_type}")


@lru_cache(maxsize=1)
def _semantic_vlabench_environment_type() -> type[Any]:
    try:
        from lerobot.envs.vlabench import VLABenchEnv
    except ImportError as error:
        raise RuntimeError("semantic VLABench observations require the isolated LeRobot 0.6 environment") from error

    class SemanticVLABenchEnv(VLABenchSemanticCameraMixin, VLABenchEnv):
        """LeRobot VLABench environment with dataset-aligned camera semantics."""

    return SemanticVLABenchEnv


def make_semantic_vlabench_environment(**kwargs: Any) -> Any:
    """Construct the project-local semantic-camera subclass of LeRobot VLABench."""

    return _semantic_vlabench_environment_type()(**kwargs)


def _has_component_suffix(name: str, component: str) -> bool:
    return name == component or any(name.endswith(f"{separator}{component}") for separator in ("/", "\\", ":"))


def _quaternion_wxyz_to_euler_xyz(quaternion: Any) -> Any:
    """Convert one finite quaternion without requiring SciPy at import or runtime."""

    import numpy as np

    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise VLABenchCameraMappingError("VLABench end-effector quaternion must be finite wxyz")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        raise VLABenchCameraMappingError("VLABench end-effector quaternion must be non-zero")
    w, x, y, z = (float(value) / norm for value in values)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sine)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _to_hwc3(frame: Any, *, height: int, width: int) -> Any:
    import numpy as np

    image = np.asarray(frame)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3:
        raise VLABenchCameraMappingError(f"VLABench camera returned an unsupported image shape {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise VLABenchCameraMappingError(f"VLABench camera returned an unsupported channel count {image.shape[-1]}")
    if image.shape[:2] != (height, width):
        import cv2

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8)


__all__ = [
    "VLABENCH_DATASET_CAMERA_KEYS",
    "VLABenchCameraMappingError",
    "VLABenchSemanticCameraMixin",
    "compiled_camera_names",
    "make_semantic_vlabench_environment",
    "resolve_vlabench_camera_indices",
]
