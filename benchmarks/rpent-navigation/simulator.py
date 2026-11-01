"""Habitat transport and public observation boundary for paired navigation."""

from __future__ import annotations

import base64
import json
import os
import select
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from agents.rpent.navigation.skill import Feedback

from embodirun.robots.mobile_base.discrete import NavigationCommandKind
from embodirun.robots.mobile_base.frames import EncodedRGBFrame
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.robots.mobile_base.request import NavigationRequest


class SimulatorProcess:
    """The evaluator owns the worker; the planner never receives its private state."""

    def __init__(self, command: list[str], output: Path):
        self.log = (output / "habitat.log").open("a")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            env=dict(os.environ, MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet"),
        )

    def request(self, command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.process.stdout], [], [], 5)
            if ready:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"Habitat exited {self.process.poll()}; inspect habitat.log")
                if line.startswith("{"):
                    return json.loads(line)
                self.log.write(line)
                self.log.flush()
        raise TimeoutError("Habitat observation timed out")

    def close(self):
        if self.process.poll() is None:
            with suppress(BrokenPipeError):
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=15)
        self.log.close()


def primitive_actions(chunk):
    actions = []
    for command in chunk.commands:
        if command.kind is NavigationCommandKind.STOP:
            actions.append(0)
        elif command.kind is NavigationCommandKind.MOVE_FORWARD:
            actions.extend([1] * round(command.distance_m / 0.25))
        else:
            action = 2 if command.kind is NavigationCommandKind.TURN_LEFT else 3
            actions.extend([action] * round(command.angle_deg / 15))
    return actions


class HabitatBackend:
    def __init__(self, simulator: SimulatorProcess, episode: dict, scenes: str, *, seed=42):
        self.simulator = simulator
        self.private = simulator.request(
            {"op": "reset", "episode": episode, "scenes_dir": scenes, "seed": seed, "turn_degrees": 15}
        )
        self.sequence = 0
        self.trajectory = []

    def observe(self):
        return self._public()

    def _public(self, *, reason="observation", trajectory=()):
        # Allowlist, never forward the worker dictionary (which contains goals,
        # geodesics and absolute positions) into an agent tool response.
        return Feedback(
            image=base64.b64decode(self.private["png"]),
            reason=reason,
            actions=tuple(step["action"] for step in trajectory),
            collisions=sum(bool(step["collision"]) for step in trajectory),
            primitive_steps=self.private["steps"],
            exhausted=self.private["steps"] >= 500,
        )

    def run(self, policy, instruction, episode_id, max_chunks, *, terminal_stop):
        trajectory = []
        reason = "chunk_limit"
        for _ in range(max_chunks):
            if self.private["steps"] >= 500 or self.private["stopped"]:
                reason = "budget" if not self.private["stopped"] else "local_stop"
                break
            frame = EncodedRGBFrame(
                self.sequence, time.monotonic(), base64.b64decode(self.private["png"]), "image/png", 640, 480
            )
            observation = NavigationObservation(episode_id, self.sequence, True, (frame,))
            chunk = policy.plan(NavigationRequest(instruction, observation))
            self.sequence += 1
            self.private = self.simulator.request(
                {"op": "step", "actions": primitive_actions(chunk), "terminal_stop": terminal_stop}
            )
            actual = self.private["trajectory"]
            trajectory.extend(actual)
            self.trajectory.extend(actual)
            if actual and actual[-1]["action"] == 0:
                reason = "local_stop"
                break
        return self._public(reason=reason, trajectory=trajectory)

    def finish(self, *, success):
        if success and not self.private["stopped"] and self.private["steps"] < 500:
            self.private = self.simulator.request({"op": "step", "actions": [0], "terminal_stop": True})
            self.trajectory.extend(self.private["trajectory"])

    def metrics(self):
        state = self.private
        success = bool(state["stopped"] and state["distance_to_goal_m"] is not None and state["distance_to_goal_m"] < 3)
        shortest = state["shortest_path_m"]
        return {
            "success": success,
            "spl": float(success) * shortest / max(shortest, state["path_length_m"], 1e-9),
            "navigation_error_m": state["distance_to_goal_m"],
            "path_length_m": state["path_length_m"],
            "shortest_path_m": shortest,
            "primitive_steps": state["steps"],
            "stopped": state["stopped"],
            "collisions": sum(bool(row["collision"]) for row in self.trajectory),
            "habitat_sim_version": state["habitat_sim_version"],
        }
