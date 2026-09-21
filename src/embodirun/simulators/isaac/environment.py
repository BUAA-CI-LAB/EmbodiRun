"""Lazy Isaac Sim SDK integration using its built-in Go2 controller."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

from embodirun.robots.unitree.go2.navigation.discrete import (
    NavigationCommand,
    NavigationCommandKind,
)

from ..navigation import NavigationObservation, NavigationTransition
from .config import IsaacConfig

_PHYSICS_DT = 1.0 / 200.0


class IsaacEnvironment:
    """Own one Isaac application, Go2 articulation, controller, and RGB sensor."""

    def __init__(self, config: IsaacConfig, application: Any) -> None:
        self.config = config
        self._application = application
        self._closed = False
        self._torch = import_module("torch")
        self._numpy = import_module("numpy")

        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import isaacsim.core.experimental.utils.transform as transform_utils
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from isaacsim.storage.native import get_assets_root_path

        self._app_utils = app_utils
        stage_utils.create_new_stage()
        stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
        RenderingManager.set_dt(1.0 / 50.0)
        SimulationManager.set_backend("torch")
        SimulationManager.set_physics_sim_device(config.device)
        SimulationManager.set_physics_dt(_PHYSICS_DT)

        scene = config.scene
        if scene.startswith("/Isaac/"):
            assets_root = get_assets_root_path()
            if not assets_root:
                raise RuntimeError("Isaac Sim assets root is unavailable")
            scene = assets_root + scene
        stage_utils.add_reference_to_stage(
            usd_path=scene,
            path="/World/Environment",
        )

        self._go2 = Go2FlatTerrainPolicy(
            prim_path="/World/Go2",
            position=list(config.start_position),
            orientation=list(config.start_orientation_wxyz),
        )
        app_utils.play(commit=True)
        self._application.update()
        self._go2.initialize()
        self._go2.post_reset()

        root_path = str(self._go2.robot.paths[0])
        camera_path = f"{root_path}/streamvln_camera"
        camera_orientation = transform_utils.euler_angles_to_quaternion(
            self._numpy.array([90.0, 0.0, -90.0]),
            degrees=True,
            extrinsic=False,
        ).numpy()
        camera = RtxCamera(
            camera_path,
            tick_rate=50.0,
            translations=self._numpy.asarray(config.camera_translation),
            orientations=camera_orientation,
        )
        self._camera = CameraSensor(
            camera,
            resolution=(config.height, config.width),
            annotators=["rgb"],
        )
        self._warm_up_camera()

    def reset(self, *, seed: int | None) -> NavigationObservation:
        self._ensure_open()
        if seed is not None:
            self._numpy.random.seed(seed)
            self._torch.manual_seed(seed)
        self._go2.post_reset()
        self._advance(self._zero_command(), self.config.settle_steps)
        return self._observation()

    def step(self, command: NavigationCommand) -> NavigationTransition:
        self._ensure_open()
        if command.kind is NavigationCommandKind.STOP:
            self._advance(self._zero_command(), self.config.settle_steps)
            observation = self._observation()
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

        start_position, start_rotation = self._pose()
        if command.kind is NavigationCommandKind.MOVE_FORWARD:
            velocity = (self.config.forward_speed_mps, 0.0, 0.0)
        else:
            sign = 1.0 if command.kind is NavigationCommandKind.TURN_LEFT else -1.0
            velocity = (0.0, 0.0, sign * self.config.turn_speed_rad_s)
        control = self._torch.tensor(
            velocity,
            dtype=self._torch.float32,
            device=self.config.device,
        )

        reached = False
        maximum_steps = max(1, math.ceil(self.config.action_timeout_s / _PHYSICS_DT))
        for _ in range(maximum_steps):
            self._advance(control, 1)
            position, rotation = self._pose()
            if command.kind is NavigationCommandKind.MOVE_FORWARD:
                reached = _planar_distance(position, start_position) >= command.distance_m
            else:
                reached = abs(_angle_delta(_yaw(rotation), _yaw(start_rotation))) >= math.radians(command.angle_deg)
            if reached:
                break
        self._advance(self._zero_command(), self.config.settle_steps)
        observation = self._observation()
        fallen = observation.position[2] < 0.15
        return NavigationTransition(
            observation=observation,
            terminated=fallen,
            info={
                "action_target_reached": reached,
                "distance_to_goal_m": observation.distance_to_goal_m,
                "fallen": fallen,
                "stop": False,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._app_utils.stop(commit=True)
        finally:
            self._application.close()
            self._closed = True

    def _advance(self, command: Any, steps: int) -> None:
        for _ in range(steps):
            self._go2.forward(_PHYSICS_DT, command)
            self._application.update()

    def _zero_command(self) -> Any:
        return self._torch.zeros(
            3,
            dtype=self._torch.float32,
            device=self.config.device,
        )

    def _pose(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        positions, rotations = self._go2.robot.get_world_poses()
        position = tuple(float(item) for item in positions.numpy()[0])
        rotation = tuple(float(item) for item in rotations.numpy()[0])
        return position, rotation

    def _observation(self) -> NavigationObservation:
        position, rotation = self._pose()
        image, _ = self._camera.get_data("rgb")
        if image is None:
            raise RuntimeError("Isaac RGB camera did not produce a frame")
        pixels = image.numpy() if hasattr(image, "numpy") else image
        return NavigationObservation(
            rgb=pixels,
            position=position,
            rotation=rotation,
            instruction=self.config.instruction,
            distance_to_goal_m=_planar_distance(position, self.config.goal_position),
            metadata={"scene": self.config.scene, "task": self.config.task},
        )

    def _warm_up_camera(self) -> None:
        for _ in range(30):
            self._application.update()
            image, _ = self._camera.get_data("rgb")
            if image is not None:
                return
        raise RuntimeError("Isaac RGB camera did not initialize")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Isaac environment is closed")


def make_isaac_environment(config: IsaacConfig) -> IsaacEnvironment:
    """Launch Isaac only inside its isolated simulator service."""

    try:
        from isaacsim import SimulationApp
    except ImportError as error:
        raise RuntimeError("Isaac execution requires the isolated sim-isaac environment") from error
    application = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        return IsaacEnvironment(config, application)
    except Exception:
        application.close()
        raise


def _planar_distance(left: Any, right: Any) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _yaw(rotation_wxyz: Any) -> float:
    w, x, y, z = (float(value) for value in rotation_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_delta(current: float, initial: float) -> float:
    return math.atan2(math.sin(current - initial), math.cos(current - initial))


__all__ = ["IsaacEnvironment", "make_isaac_environment"]
