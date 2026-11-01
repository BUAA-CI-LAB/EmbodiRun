"""Public RPent skill proxy; private evaluation results stay outside tool responses."""

from __future__ import annotations

import base64
import time
from dataclasses import asdict

from agents.rpent.navigation.skill import Feedback


class RemoteSkill:
    def __init__(self, executor, episode, scenes, *, max_runtime_s, expected_sources, profile="event"):
        self.executor = executor
        self.check_cancelled = lambda: None
        self.model_calls = self.subgoals = 0
        self.finished = False
        self.trajectory = self.timings = []
        self.transport_timings = []
        reset = self._request("reset", episode=episode, scenes=scenes, max_runtime_s=max_runtime_s, profile=profile)
        if reset.get("source_sha256") != expected_sources:
            raise ValueError("remote execution source differs from the evaluator's frozen protocol")

    def _request(self, op, **values):
        started = time.perf_counter()
        response = self.executor.request({"op": op, **values})
        self.transport_timings.append(
            {
                "op": op,
                "round_trip_ms": (time.perf_counter() - started) * 1000,
                **response.get("transport", {}),
            }
        )
        if "error" in response:
            raise RuntimeError(response["error"])
        self.model_calls = response.get("model_calls", self.model_calls)
        self.subgoals = response.get("subgoals", self.subgoals)
        self.finished = response.get("finished", self.finished)
        return response

    @staticmethod
    def _feedback(response):
        return Feedback(
            image=base64.b64decode(response["image"]),
            mime_type=response["mime_type"],
            reason=response["reason"],
            actions=tuple(response["actions"]),
            collisions=response["collisions"],
            primitive_steps=response["primitive_steps"],
            exhausted=response["exhausted"],
        )

    def observe(self):
        return self._feedback(self._request("observe"))

    def navigate(self, instruction, *, max_chunks=4, restart=False, terminal_stop=False):
        self.check_cancelled()
        return self._feedback(
            self._request(
                "navigate", instruction=instruction, max_chunks=max_chunks, restart=restart, terminal_stop=terminal_stop
            )
        )

    def navigate_until_event(self, instruction, *, restart=False, event_policy):
        self.check_cancelled()
        return self._feedback(
            self._request(
                "navigate_event",
                instruction=instruction,
                restart=restart,
                event_policy=asdict(event_policy),
            )
        )

    def finish(self, *, success):
        self._request("finish", success=success)

    def metrics(self):
        result = self._request("result")
        self.trajectory, self.timings = result["trajectory"], result["timings"]
        return result["metrics"]

    def close(self):
        self._request("close_episode")
