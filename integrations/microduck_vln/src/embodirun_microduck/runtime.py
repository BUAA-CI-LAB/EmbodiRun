"""Demo-only MicroDuck/MPC and bounded streaming video adapters."""

from __future__ import annotations

import math
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DemoSimulation:
    def __init__(self, root: Path, *, max_physics_steps: int, fps: int):
        import mujoco

        self._mujoco = mujoco
        import onnxruntime as ort
        from vln_mujoco.mpc import Q_WEIGHTS, R_WEIGHTS, MPCController
        from vln_mujoco.robots.microduck import MicroDuckBackend

        walking_policy = root / "src/robots_assets/alpha_walking.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        walking_session = ort.InferenceSession(
            str(walking_policy), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.robot = MicroDuckBackend(
            robot_model=root / "src/robots_assets/mjcf_assets/robot_allcollisions.xml",
            walking_policy=walking_policy,
            policy_session=walking_session,
        )
        self.renderer = mujoco.Renderer(self.robot.model, height=270, width=480)
        self.camera = mujoco.MjvCamera()
        # Use the existing CasADi solver with the MicroDuck's physical speed limits.
        self.controller = MPCController(
            horizon=5, dt_s=0.1, w_max=1.5, a_max_v=2.0, a_max_w=5.0, q_weights=Q_WEIGHTS, r_weights=R_WEIGHTS
        )
        self.max_physics_steps = max_physics_steps
        self.dt = float(self.robot.model.opt.timestep)
        self.frame_stride = max(1, round(1 / (fps * self.dt)))
        self.control_stride = max(1, round(0.1 / self.dt))
        self.callback = None
        self.physics_steps = 0
        self.path_length = 0.0
        self.previous_command = (0.0, 0.0)
        self.mpc_ms = []

    def reset(self, start):
        self.robot.reset()
        index = self.robot._base_qpos
        x, y, yaw = start
        self.robot.data.qpos[index : index + 2] = [x, y]
        self.robot.data.qpos[index + 3 : index + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self._mujoco.mj_forward(self.robot.model, self.robot.data)
        self.physics_steps = 0
        self.path_length = 0.0
        self.previous_command = (0.0, 0.0)
        self.mpc_ms = []

    def pose(self):
        x, y, z, yaw = self.robot.state().pose
        return (x, y, yaw)

    def distance(self, goal):
        x, y, _ = self.pose()
        return math.hypot(x - goal[0], y - goal[1])

    def render(self, third=False):
        camera = self.robot.third_person_camera(self.camera) if third else self.robot.camera_name
        self.renderer.update_scene(self.robot.data, camera=camera)
        return self.renderer.render().copy()

    def _step(self, command):
        before = self.pose()
        self.robot.step(command)
        after = self.pose()
        if not np.isfinite(self.robot.data.qpos).all() or not np.isfinite(self.robot.data.qvel).all():
            raise RuntimeError("Non-finite MuJoCo state")
        self.path_length += math.hypot(after[0] - before[0], after[1] - before[1])
        self.physics_steps += 1
        if self.callback is not None and self.physics_steps % self.frame_stride == 0:
            self.callback()

    def execute(self, action):
        from vln_mujoco.mpc import (
            Q_WEIGHTS,
            build_pose_aligned_reference,
            project_body_to_world,
            project_world_to_local,
            wrap_angle,
        )
        from vln_mujoco.robots.microduck import shape_microduck_command

        kind, magnitude = action
        start_s = time.perf_counter()
        before_steps = self.physics_steps
        reached = kind == 0
        if kind == 0:
            for _ in range(self.control_stride):
                self._step((0.0, 0.0))
            self.previous_command = (0.0, 0.0)
        else:
            distance = magnitude / 100 if kind == 1 else 0.0
            angle = math.radians(magnitude) * (1 if kind == 2 else -1) if kind != 1 else 0.0
            # Each discrete R2R command specifies an endpoint. An interpolated
            # slow path can keep the angular command below this gait's stepping
            # threshold indefinitely; track the endpoint over the MPC horizon.
            body = [(distance, 0, angle)]
            trajectory = project_body_to_world(body, self.pose())
            target = trajectory[-1]
            command = self.previous_command
            for index in range(self.max_physics_steps):
                pose = np.asarray(self.pose())
                angular_error = abs(wrap_angle(target[2] - pose[2]))
                position_error = np.linalg.norm(target[:2] - pose[:2])
                if (kind == 1 and position_error <= 0.03) or (kind != 1 and angular_error <= math.radians(2)):
                    reached = True
                    break
                if index % self.control_stride == 0:
                    reference = build_pose_aligned_reference(trajectory, pose, horizon=5, weights=Q_WEIGHTS)
                    reference = project_world_to_local(reference, pose)
                    solve_start = time.perf_counter()
                    command, _ = self.controller.solve(np.zeros(3), reference, self.previous_command, 0.30)
                    self.mpc_ms.append((time.perf_counter() - solve_start) * 1000)
                    command = shape_microduck_command(*command)
                    self.previous_command = command
                self._step(command)
            # Bound angular coast with the same 150-step brake budget as the existing env.
            if kind != 1:
                for _ in range(150):
                    if abs(self.robot.state().velocity[1]) < 0.1:
                        break
                    self._step((0.0, 0.0))
                self.previous_command = (0.0, 0.0)
        if self.callback is not None:
            self.callback()
        pose = np.asarray(self.pose())
        position_error = 0.0 if kind == 0 else float(np.linalg.norm(target[:2] - pose[:2]))
        yaw_error = 0.0 if kind == 0 else abs(math.degrees(wrap_angle(target[2] - pose[2])))
        reached = kind == 0 or (position_error <= 0.03 if kind == 1 else yaw_error <= 2.0)
        return {
            "target_reached": reached,
            "physics_steps": self.physics_steps - before_steps,
            "position_error_m": position_error,
            "yaw_error_degrees": yaw_error,
            "wall_s": time.perf_counter() - start_s,
            "pose": list(self.pose()),
        }

    def close(self):
        self.callback = None
        self.renderer.close()


class EpisodeVideo:
    """Encode on the GPU node via a pipe; keep no raw-frame archive."""

    def __init__(self, path: Path, fps: int, ffmpeg: str):
        self.path = path
        self.fps = fps
        self.frames = 0
        self.closed = False
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x400",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-threads",
                "2",
                "-maxrate",
                "1200k",
                "-bufsize",
                "2400k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
        )
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        except OSError:
            self.font = ImageFont.load_default()

    def compose(
        self, first, third, *, instruction, episode_id, action, step, distance, sim_time, inference_ms, banner=None
    ):
        canvas = Image.new("RGB", (960, 400), "#111827")
        canvas.paste(Image.fromarray(first), (0, 30))
        canvas.paste(Image.fromarray(third), (480, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), "MICRODUCK / first person", font=self.font, fill="white")
        draw.text((490, 6), "Third person / ActiveVLN SFT-v3 / embodiinfer EngineCore", font=self.font, fill="white")
        for index, line in enumerate(textwrap.wrap(instruction, width=108)[:2]):
            draw.text((10, 306 + 18 * index), line, font=self.font, fill="#fef08a")
        draw.text(
            (10, 347),
            f"{episode_id} | action {step}: {action} | distance {distance:.2f} m",
            font=self.font,
            fill="white",
        )
        draw.text(
            (10, 371),
            f"sim {sim_time:.1f}s | inference {inference_ms:.0f}ms",
            font=self.font,
            fill="#a5b4fc",
        )
        if banner:
            draw.rectangle((40, 90, 920, 220), fill="#111827")
            for index, line in enumerate(banner):
                draw.text((64, 107 + 27 * index), line, font=self.font, fill="white")
        return canvas

    def write(self, image, repeats=1):
        data = np.asarray(image, dtype=np.uint8).tobytes()
        for _ in range(repeats):
            self.process.stdin.write(data)
            self.frames += 1
        if self.path.exists() and self.path.stat().st_size > 90_000_000:
            raise RuntimeError("Video reached the 90 MB quota guard")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=30)
            if code:
                raise RuntimeError(f"ffmpeg failed ({code}); see {self.path.with_suffix('.ffmpeg.log')}")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()
