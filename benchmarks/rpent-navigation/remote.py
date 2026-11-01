"""Public RPent skill proxy; private evaluation results stay outside tool responses."""

from __future__ import annotations

import base64

from agents.rpent.navigation.skill import Feedback


class RemoteSkill:
    def __init__(self, executor, episode, scenes, *, max_runtime_s, expected_sources):
        self.executor = executor
        self.check_cancelled = lambda: None
        self.model_calls = self.subgoals = 0
        self.finished = False
        self.trajectory = self.timings = []
        reset = self._request("reset", episode=episode, scenes=scenes, max_runtime_s=max_runtime_s)
        if reset.get("source_sha256") != expected_sources:
            raise ValueError("remote execution source differs from the evaluator's frozen protocol")

    def _request(self, op, **values):
        response = self.executor.request({"op": op, **values})
        if "error" in response:
            raise RuntimeError(response["error"])
        self.model_calls = response.get("model_calls", self.model_calls)
        self.subgoals = response.get("subgoals", self.subgoals)
        self.finished = response.get("finished", self.finished)
        return response

    @staticmethod
    def _feedback(response):
        return Feedback(
            png=base64.b64decode(response["png"]),
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

    def finish(self, *, success):
        self._request("finish", success=success)

    def metrics(self):
        result = self._request("result")
        self.trajectory, self.timings = result["trajectory"], result["timings"]
        return result["metrics"]

    def close(self):
        self._request("close_episode")
