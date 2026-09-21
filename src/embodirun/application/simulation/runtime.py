"""Policy loop for one simulator episode."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from embodirun.bindings import BindingMapper
from embodirun.model_services import InferenceClient
from embodirun.robots import RobotAction
from embodirun.simulators import SimulatorAdapter, SimulatorObservation


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Summary returned after one bounded simulator episode."""

    policy_steps: int
    environment_steps: int
    total_reward: float
    terminated: bool
    truncated: bool


class SimulationRuntime:
    """Own one policy session and advance one simulator without wall-clock sleeps."""

    def __init__(
        self,
        simulator: SimulatorAdapter,
        client: InferenceClient,
        *,
        mapper: BindingMapper,
        chunk_steps: int,
    ) -> None:
        if isinstance(chunk_steps, bool) or not isinstance(chunk_steps, int) or chunk_steps <= 0:
            raise ValueError("chunk_steps must be a positive integer")
        self.simulator = simulator
        self.client = client
        self.mapper = mapper
        self.chunk_steps = chunk_steps
        self.session = None

    def run(
        self,
        *,
        instruction: str | None,
        max_policy_steps: int,
        task: str | None = None,
        seed: int | None = None,
    ) -> EpisodeOutcome:
        if instruction is not None and not instruction.strip():
            raise ValueError("instruction must be non-empty when provided")
        if isinstance(max_policy_steps, bool) or not isinstance(max_policy_steps, int) or max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be a positive integer")

        current = self.simulator.reset(task=task, seed=seed)
        prompt = instruction.strip() if instruction is not None else _instruction(current)
        self.session = self.client.open_session(
            robot_id=self.simulator.simulator_id,
            action_space=self.mapper.policy_action_space,
            metadata={"source": "simulator"},
        )
        total_reward = 0.0
        environment_steps = 0
        policy_steps = 0
        terminated = False
        truncated = False
        for step_id in range(max_policy_steps):
            request = self.mapper.map_observation(
                current.robot,
                session_id=self.session.session_id,
                request_id=f"step-{step_id}-{uuid.uuid4().hex}",
                step_id=step_id,
                instruction=prompt,
                frames=current.frames,
            )
            result = self.client.step(request)
            actions = tuple(self.mapper.map_result(result))
            if not actions:
                raise RuntimeError("binding returned an empty action chunk")
            if any(not isinstance(action, RobotAction) for action in actions):
                raise TypeError("binding action chunk must contain RobotAction values")
            policy_steps += 1
            for action in actions[: self.chunk_steps]:
                transition = self.simulator.step(action)
                current = transition.observation
                total_reward += transition.reward
                environment_steps += 1
                terminated = transition.terminated
                truncated = transition.truncated
                if transition.done:
                    break
            if terminated or truncated:
                break

        return EpisodeOutcome(
            policy_steps=policy_steps,
            environment_steps=environment_steps,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self.session is not None:
            self.client.close(self.session.session_id)
            self.session = None


def _instruction(observation: SimulatorObservation) -> str:
    instruction = observation.robot.metadata.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("simulator reset did not provide an instruction; pass one explicitly")
    return instruction.strip()


__all__ = ["EpisodeOutcome", "SimulationRuntime"]
