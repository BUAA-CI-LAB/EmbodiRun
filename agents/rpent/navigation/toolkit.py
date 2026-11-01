"""RPent tools with visual evidence and episode-local working memory."""

from __future__ import annotations

import json
from pathlib import Path

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly

from .skill import Feedback, NavigationSkill

SYSTEM_PROMPT = """You control an embodied navigator through RPent tools.
Solve the complete user instruction in an unfamiliar environment. Begin by
observing. Track ordered landmarks and what remains to do in update_plan.
Use navigate to give ActiveVLN a concrete, visually grounded local instruction.
It returns fresh RGB after at most the requested number of complete action
chunks. Use larger bounds for clear uninterrupted travel and smaller bounds
near a decision or destination. Reusing exactly the same instruction continues
its visual history; changing it starts a new local skill. Never rewrite an
unchanged goal merely to continue. A local_stop is only the skill's claim: inspect
the returned view and verify the task's required landmarks before finishing.
Keep unfulfilled route constraints; do not turn the task into a shorter task.
If blocked or revisiting the same view, revise the plan using observed evidence
and restart the skill when needed. Do not invent visible objects or destinations.
You may use the full user instruction as the local instruction when decomposition
would remove necessary route context. Use only the supplied tools and images.
There is no privileged map, goal position or success detector available.
finish(success) requests the final STOP; it is your judgement, not measured
success. Stop as stuck/failure if the budget is exhausted. Do not return a prose
answer instead of calling finish. Tool errors are failures, not permission to
assume motion happened. Never execute parallel navigation tools.
"""


def spec(name, description, properties=None, required=()):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": list(required),
            "additionalProperties": False,
        },
    }


class NavigationToolkit(Toolkit):
    def __init__(self, skill: NavigationSkill, output: Path, *, max_tool_calls: int = 80):
        self.skill, self.output = skill, output
        self.max_tool_calls, self.tool_calls = max_tool_calls, 0
        self.plan = {}
        self.finish_result = None
        output.mkdir(parents=True, exist_ok=True)
        super().__init__(
            dashboard_events=NullDashboardEventSink(), state=EnvState(output), memory=MemoryManager(output / "memory")
        )
        self.skill.check_cancelled = self.raise_if_cancelled
        self.add_tool("observe", spec("observe", "Read current RGB and task progress without moving."), self.observe)
        self.add_tool(
            "navigate",
            spec(
                "navigate",
                "Run a bounded ActiveVLN local skill, then inspect fresh RGB.",
                {
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 10000},
                    "max_chunks": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
                    "restart": {"type": "boolean", "default": False},
                },
                ("instruction",),
            ),
            self.navigate,
        )
        self.add_tool(
            "update_plan",
            spec(
                "update_plan",
                "Save episode-local progress based only on observed evidence.",
                {
                    "completed": {"type": "array", "items": {"type": "string"}},
                    "remaining": {"type": "array", "items": {"type": "string"}},
                    "observations": {"type": "string"},
                },
                ("completed", "remaining", "observations"),
            ),
            self.update_plan,
        )
        self.add_tool(
            "finish",
            spec(
                "finish",
                "End the task; success requests final STOP, subject to independent evaluation.",
                {
                    "status": {"type": "string", "enum": ["success", "failure", "stuck"]},
                    "summary": {"type": "string", "minLength": 1},
                },
                ("status", "summary"),
            ),
            self.finish,
        )

    def _register_common_tools(self):
        # Generic filesystem tools expose evaluator artifacts. The navigation
        # toolkit instead provides a structured, episode-scoped memory tool.
        pass

    def get_env_state(self, *, command, result, elapsed_s):
        return result

    def get_tools_spec(self):
        # These schemas have no global RPent path placeholders. Keeping them
        # literal avoids coupling an episode toolkit to the CLI's global output.
        return [schema for schema, _handler in self._tools.values()]

    def execute_tool(self, name, input_dict):
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls and name != "finish":
            name, input_dict = "finish", {"status": "stuck", "summary": "Tool budget exhausted"}
        result = super().execute_tool(name, input_dict)
        public = {key: value for key, value in result.result.items() if key != "_image_bytes"}
        with (self.output / "tools.jsonl").open("a") as stream:
            stream.write(json.dumps({"tool": name, "input": input_dict, "result": public}) + "\n")
        return result

    def _feedback(self, feedback: Feedback):
        return {
            "_image_bytes": feedback.png,
            "reason": feedback.reason,
            "executed_actions": list(feedback.actions),
            "collisions": feedback.collisions,
            "primitive_steps": feedback.primitive_steps,
            "budget_exhausted": feedback.exhausted,
            "model_calls": self.skill.model_calls,
            "subgoals": self.skill.subgoals,
            "remaining_tool_calls": max(0, self.max_tool_calls - self.tool_calls),
            "plan": self.plan,
        }

    @readonly
    def observe(self):
        return self._feedback(self.skill.observe())

    def navigate(self, instruction, max_chunks=4, restart=False):
        if type(max_chunks) is not int or not 1 <= max_chunks <= 8:
            raise ValueError("max_chunks must be in [1,8]")
        return self._feedback(self.skill.navigate(instruction, max_chunks=max_chunks, restart=restart))

    @readonly
    def update_plan(self, completed, remaining, observations):
        if any(
            not isinstance(value, list) or any(not isinstance(v, str) for v in value)
            for value in (completed, remaining)
        ) or not isinstance(observations, str):
            raise ValueError("plan requires string lists and an observation summary")
        plan = {"completed": completed, "remaining": remaining, "observations": observations}
        encoded = json.dumps(plan, ensure_ascii=False)
        if len(encoded) > 16000:
            raise ValueError("working memory exceeds 16000 characters")
        self.memory.root.mkdir(parents=True, exist_ok=True)
        path = self.memory.root / "working.json"
        path.with_suffix(".tmp").write_text(encoded + "\n")
        path.with_suffix(".tmp").replace(path)
        self.plan = plan
        return {"plan": plan}

    def finish(self, status, summary):
        if status not in ("success", "failure", "stuck") or not isinstance(summary, str) or not summary.strip():
            raise ValueError("finish requires a supported status and evidence summary")
        self.skill.finish(success=status == "success")
        self.finish_result = {"_finish": True, "status": status, "summary": summary, "task_success": "unverified"}
        return self.finish_result

    def solved(self):
        # Planner claims are never simulator ground truth.
        return False

    def close(self):
        self.skill.close()
