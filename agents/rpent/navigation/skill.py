"""Bound recurrent navigation skills without importing a planner or model runtime."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from embodirun.application.navigation.chunk_session import ChunkPolicy
from embodirun.robots.mobile_base.request import NavigationRequest

from .events import EventMonitor, EventPolicy


@dataclass(frozen=True)
class Feedback:
    """Agent-visible evidence. Simulator metrics deliberately have no field here."""

    image: bytes
    mime_type: str = "image/png"
    reason: str = "observation"
    actions: tuple[int, ...] = ()
    collisions: int | None = None
    primitive_steps: int | None = None
    exhausted: bool = False


class NavigationBackend(Protocol):
    def observe(self) -> Feedback: ...

    def run(
        self, policy: ChunkPolicy, instruction: str, episode_id: str, max_chunks: int, *, terminal_stop: bool
    ) -> Feedback: ...

    def finish(self, *, success: bool) -> None: ...


class RecurrentPolicy:
    """Keep model history across bounded calls; only a new subgoal resets it."""

    def __init__(self, policy, check_cancelled: Callable[[], None]):
        self.policy = policy
        self.check_cancelled = check_cancelled
        self.calls = 0

    def plan(self, request: NavigationRequest):
        self.check_cancelled()
        observation = replace(request.observation, reset=self.calls == 0)
        chunk = self.policy.plan(NavigationRequest(request.instruction, observation))
        self.calls += 1
        return chunk

    def close(self) -> None:
        self.policy.close()


class NavigationSkill:
    """Own one episode's model sessions; execution remains in the backend."""

    def __init__(
        self,
        backend: NavigationBackend,
        policy_factory: Callable,
        *,
        max_chunks: int = 500,
        max_runtime_s: float = 3600,
        check_cancelled: Callable[[], None] = lambda: None,
    ):
        if type(max_chunks) is not int or max_chunks < 1 or max_runtime_s <= 0:
            raise ValueError("skill budgets must be positive")
        self.backend, self.policy_factory = backend, policy_factory
        self.max_chunks, self.max_runtime_s = max_chunks, max_runtime_s
        self.check_cancelled = check_cancelled
        self.started = time.monotonic()
        self.policy: RecurrentPolicy | None = None
        self.instruction = ""
        self.episode_id = ""
        self.subgoals = self.model_calls = 0
        self.finished = self.failed = False
        self._local_stopped = False
        self.feedback: Feedback | None = None

    def navigate_until_event(self, instruction: str, *, restart: bool = False, event_policy=None) -> Feedback:
        """Keep each policy chunk closed-loop locally; return only at an event or checkpoint."""
        policy = event_policy or EventPolicy()
        actions = []
        collisions = None
        try:
            monitor = EventMonitor(self.backend.observe().image, policy)
            for index in range(policy.max_chunks):
                feedback = self.navigate(instruction, max_chunks=1, restart=restart and index == 0)
                actions.extend(feedback.actions)
                if feedback.collisions is not None:
                    collisions = (collisions or 0) + feedback.collisions
                if feedback.reason != "chunk_limit" or feedback.exhausted:
                    break
                event = monitor.update(feedback)
                if event:
                    feedback = replace(feedback, reason=event)
                    break
            else:
                feedback = replace(feedback, reason="checkpoint")
            self.feedback = replace(feedback, actions=tuple(actions), collisions=collisions)
            return self.feedback
        except BaseException:
            self.failed = True
            raise

    def observe(self) -> Feedback:
        self.feedback = self.backend.observe()
        return self.feedback

    def navigate(
        self, instruction: str, *, max_chunks: int = 4, restart: bool = False, terminal_stop: bool = False
    ) -> Feedback:
        if self.finished or self.failed:
            raise RuntimeError("navigation is closed; failed execution cannot be retried")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 10000:
            raise ValueError("subgoal must contain 1-10000 characters")
        if type(max_chunks) is not int or not 1 <= max_chunks <= self.max_chunks:
            raise ValueError("invalid chunk budget")
        if type(restart) is not bool or type(terminal_stop) is not bool:
            raise ValueError("restart and terminal_stop must be booleans")
        remaining = self.max_chunks - self.model_calls
        if remaining < 1 or time.monotonic() - self.started >= self.max_runtime_s:
            self.feedback = replace(self.backend.observe(), reason="budget", exhausted=True)
            return self.feedback
        instruction = instruction.strip()
        # STOP closes this local skill, not the overall task. Starting it again
        # needs new model history even if its text is unchanged.
        reset = restart or instruction != self.instruction or self._local_stopped
        if self.policy is None or reset:
            self.close()
            self.policy = RecurrentPolicy(self.policy_factory(), self.check_cancelled)
            self.instruction = instruction
            self.episode_id = "subgoal-" + uuid.uuid4().hex
            self.subgoals += 1
            self._local_stopped = False
        before = self.policy.calls
        try:
            self.feedback = self.backend.run(
                self.policy,
                instruction,
                self.episode_id,
                min(max_chunks, remaining),
                terminal_stop=terminal_stop,
            )
            self._local_stopped = self.feedback.reason == "local_stop"
            if terminal_stop and self.feedback.reason == "local_stop":
                self.finished = True
            return self.feedback
        except BaseException:
            self.failed = True
            raise
        finally:
            self.model_calls += self.policy.calls - before

    def finish(self, *, success: bool) -> None:
        if not self.finished:
            self.backend.finish(success=success)
            self.finished = True
        self.close()

    def close(self) -> None:
        if self.policy is not None:
            self.policy.close()
            self.policy = None
