"""Isolated Habitat-Sim process: JSON commands, PNG observations, and metric evidence.

The script deliberately imports no model package or EmbodiRun runtime, allowing
an existing Habitat Python environment to serve the benchmark controller.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
from pathlib import Path


def main() -> None:
    """Execute reset/step commands with pinned sensors and explicit primitive turns."""
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs
    from PIL import Image

    simulator = None
    scene = None
    turn_degrees = None
    goal = None
    stopped = False
    steps = 0
    length = 0.0
    shortest = None

    def position():
        return simulator.get_agent(0).get_state().position.copy()

    def distance():
        path = habitat_sim.ShortestPath()
        path.requested_start = position()
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if not simulator.pathfinder.find_path(path) or not math.isfinite(path.geodesic_distance):
            return None
        return float(path.geodesic_distance)

    def observation(trajectory):
        raw = simulator.get_sensor_observations()["rgb"][..., :3]
        buffer = io.BytesIO()
        Image.fromarray(raw).save(buffer, format="PNG")
        state = simulator.get_agent(0).get_state()
        return {
            "png": base64.b64encode(buffer.getvalue()).decode(),
            "position": state.position.tolist(),
            "rotation": quat_to_coeffs(state.rotation).tolist(),
            "distance_to_goal_m": distance(),
            "shortest_path_m": shortest,
            "path_length_m": length,
            "steps": steps,
            "stopped": stopped,
            "trajectory": trajectory,
            "habitat_sim_version": habitat_sim.__version__,
            "turn_degrees": turn_degrees,
        }

    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "close":
                break
            if command["op"] == "reset":
                episode = command["episode"]
                target = str(Path(command["scenes_dir"]) / episode["scene_id"])
                requested_turn = command.get("turn_degrees", 15)
                if type(requested_turn) is not int or requested_turn not in (15, 30):
                    raise ValueError("primitive turns must be 15 or 30 degrees")
                if scene != target or turn_degrees != requested_turn:
                    if simulator is not None:
                        simulator.close()
                    cfg = habitat_sim.SimulatorConfiguration()
                    cfg.scene_id = target
                    cfg.gpu_device_id = 0  # CUDA_VISIBLE_DEVICES selects the physical renderer GPU.
                    cfg.enable_physics = False
                    cfg.allow_sliding = True
                    sensor = habitat_sim.CameraSensorSpec()
                    sensor.uuid = "rgb"
                    sensor.sensor_type = habitat_sim.SensorType.COLOR
                    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
                    sensor.resolution = [480, 640]
                    sensor.position = [0.0, 1.25, 0.0]
                    sensor.hfov = 90.0
                    agent = habitat_sim.agent.AgentConfiguration()
                    agent.height = 1.5
                    agent.radius = 0.1
                    agent.sensor_specifications = [sensor]
                    agent.action_space = {
                        name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=value))
                        for name, value in (
                            ("move_forward", 0.25),
                            ("turn_left", requested_turn),
                            ("turn_right", requested_turn),
                        )
                    }
                    simulator = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
                    scene = target
                    turn_degrees = requested_turn
                simulator.seed(command["seed"])
                simulator.reset()
                state = habitat_sim.AgentState()
                state.position = np.asarray(episode["start_position"], dtype=np.float32)
                state.rotation = quat_from_coeffs(np.asarray(episode["start_rotation"]))
                simulator.get_agent(0).set_state(state, reset_sensors=True)
                goal = episode["goals"][0]["position"]
                stopped, steps, length = False, 0, 0.0
                shortest = distance()
                if shortest is None:
                    raise RuntimeError("episode start has no geodesic path to its goal")
                result = observation([])
            elif command["op"] == "step":
                if simulator is None or stopped:
                    raise RuntimeError("step requires an active episode")
                trajectory = []
                for next_step, action in enumerate(command["actions"], start=steps + 1):
                    if type(action) is not int or action not in (0, 1, 2, 3):
                        raise ValueError("invalid primitive action")
                    if next_step > 500:
                        break
                    before = position()
                    steps = next_step
                    collided = False
                    if action == 0:
                        stopped = True
                    else:
                        transition = simulator.step({1: "move_forward", 2: "turn_left", 3: "turn_right"}[action])
                        collided = bool(transition["collided"])
                    after = position()
                    length += float(np.linalg.norm(after - before))
                    trajectory.append(
                        {
                            "action": action,
                            "position": after.tolist(),
                            "distance_to_goal_m": distance(),
                            "collision": collided,
                        }
                    )
                    if stopped:
                        break
                result = observation(trajectory)
            else:
                raise ValueError("unknown simulator command")
            print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        if simulator is not None:
            simulator.close()


if __name__ == "__main__":
    main()
