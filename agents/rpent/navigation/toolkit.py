"""RPent tools with visual evidence and episode-local working memory."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.session import EnvState
from rpent.tools.toolkit import ToolCancelled, Toolkit, readonly

from .events import EventPolicy
from .images import planner_image
from .skill import Feedback, NavigationSkill
from .task_source import indexed_task, select_task_spans

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
Calling finish with status="success" requests the final STOP; it is your judgement, not measured
success. Stop as stuck/failure if the budget is exhausted. Do not return a prose
answer instead of calling finish. Tool errors are failures, not permission to
assume motion happened. Never execute parallel navigation tools.
"""

EVENT_PROMPT = """You plan a navigation task through RPent tools and ActiveVLN.
The initial RGB and public feedback are attached. Plan directly from them.
Every navigate result includes a fresh RGB observation.
Preserve every ordered landmark and route constraint in the
user instruction. Give navigate a meaningful, visually grounded route segment,
or the whole original instruction when splitting it would lose context.
ActiveVLN then runs a local visual feedback loop: it sees a new image before
every action chunk. You do not need to supervise individual turns or chunks.
Control returns on local_stop, blocked, stalled, loop, budget, or a bounded
checkpoint. Inspect the returned RGB and event before deciding what to do next.
A local_stop is only ActiveVLN's claim, so verify the visible destination and
remaining route constraints. A checkpoint merely asks for a periodic inspection;
continue the same instruction verbatim unless the evidence requires a new goal.
Changing instruction or restart=true resets the local model's visual history.
After blocked/stalled/loop, choose an evidence-based recovery or revised subgoal.
Keep concise working memory in the optional plan argument of navigate or finish;
it is saved in that call, without a separate planning tool round.
Do not invent objects, use hidden maps, or assume a tool error means motion ran.
Call finish with status="success" only when the full task is visually verified; it requests
the final STOP, not a ground-truth success judgement. Finish stuck/failure when
the budget is exhausted. Use only the supplied tools and images. Do not execute
parallel navigation tools or return prose in place of finish.
"""

PLAN_PROPERTIES = {
    "completed": {"type": "array", "items": {"type": "string"}},
    "remaining": {"type": "array", "items": {"type": "string"}},
    "observations": {"type": "string"},
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": PLAN_PROPERTIES,
    "required": list(PLAN_PROPERTIES),
    "additionalProperties": False,
}

# Optional format demonstrations for a small shared model. These are invented
# tasks, not observations or achievements from the current episode.
SHARED_TOOL_EXAMPLES = (
    (
        json.dumps(
            {
                "task": "Visit the lounge, then the office.",
                "working_memory": {"completed": ["lounge"], "remaining": ["office"]},
                "latest_feedback": {
                    "reason": "local_stop",
                    "observations": "At the lounge exit; office not visited yet.",
                },
            },
            separators=(",", ":"),
        ),
        json.dumps(
            {
                "name": "navigate",
                "arguments": {
                    "instruction": "Go to the office.",
                    "plan": {"completed": ["lounge"], "remaining": ["office"], "observations": "At the lounge exit."},
                },
            },
            separators=(",", ":"),
        ),
    ),
    (
        json.dumps(
            {
                "task": "Go to the balcony and stop.",
                "working_memory": {"completed": ["balcony"], "remaining": []},
                "latest_feedback": {
                    "reason": "local_stop",
                    "observations": "At the balcony railing; all goals verified.",
                },
            },
            separators=(",", ":"),
        ),
        json.dumps(
            {
                "name": "finish",
                "arguments": {"status": "success", "summary": "At the balcony railing; all goals verified."},
            },
            separators=(",", ":"),
        ),
    ),
)


def shared_tool_examples(*, plan_tools: bool = False) -> tuple[tuple[str, str], ...]:
    """Return fixed format demonstrations for the selected tool capabilities."""
    examples = list(SHARED_TOOL_EXAMPLES)
    if plan_tools:
        for remaining, completed in ((["kitchen", "study"], []), (["study"], ["kitchen"])):
            place = remaining[0]
            state = {
                "task": "Visit the kitchen, then the study.",
                "working_memory": {"completed": completed, "remaining": remaining, "observations": ""},
                "previous_tool_call": {"name": "navigate", "arguments": {"instruction": "Go to the " + place + "."}},
                "latest_feedback": {"reason": "local_stop", "observations": "Inside the " + place + "."},
            }
            arguments = {"evidence": "Inside the " + place + "."}
            if len(remaining) > 1:
                arguments["instruction"] = "Go to the study."
            examples.append(
                (
                    json.dumps(state, separators=(",", ":")),
                    json.dumps({"name": "advance_goal", "arguments": arguments}, separators=(",", ":")),
                )
            )
    return tuple(examples)


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
    def __init__(
        self,
        skill: NavigationSkill,
        output: Path,
        *,
        max_tool_calls: int = 80,
        profile="event",
        event_policy=None,
        initial_task: str | None = None,
        plan_tools: bool = False,
        source_tools: bool = False,
    ):
        if profile not in ("event", "legacy"):
            raise ValueError("unsupported navigation profile")
        if initial_task is not None and (not isinstance(initial_task, str) or not initial_task.strip()):
            raise ValueError("initial_task must contain a task")
        if type(plan_tools) is not bool or (plan_tools and profile != "event"):
            raise ValueError("plan_tools requires the event profile")
        if type(source_tools) is not bool or (source_tools and (profile != "event" or initial_task is None)):
            raise ValueError("source_tools requires the event profile and an initial task")
        self.source_tools = source_tools
        self.profile = profile
        self.event_policy = event_policy or EventPolicy()
        self.system_prompt = EVENT_PROMPT if profile == "event" else SYSTEM_PROMPT
        self.skill, self.output = skill, output
        self.max_tool_calls, self.tool_calls = max_tool_calls, 0
        self.plan = {}
        self.task: str | None = None
        self.finish_result = None
        self._episode_cancelled = threading.Event()
        self._tool_thread = threading.local()
        output.mkdir(parents=True, exist_ok=True)
        super().__init__(
            dashboard_events=NullDashboardEventSink(), state=EnvState(output), memory=MemoryManager(output / "memory")
        )
        if initial_task is not None:
            # Record the given task as unfulfilled without inventing a route,
            # visited landmarks or a successful tool execution.
            self.update_plan(completed=[], remaining=[initial_task], observations="")
            self.set_task(initial_task)
        self.skill.check_cancelled = self.raise_if_cancelled
        self.add_tool("observe", spec("observe", "Read current RGB and task progress without moving."), self.observe)
        self.add_tool(
            "navigate",
            spec(
                "navigate",
                "Run ActiveVLN's local visual loop until an event or bounded checkpoint; returns fresh RGB."
                if profile == "event"
                else "Run a bounded ActiveVLN local skill, then inspect fresh RGB.",
                {
                    "instruction": {"type": "string", "minLength": 1, "maxLength": 10000},
                    "restart": {"type": "boolean", "default": False},
                    **(
                        {"plan": PLAN_SCHEMA}
                        if profile == "event"
                        else {
                            "max_chunks": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
                        }
                    ),
                },
                ("instruction",),
            ),
            self.navigate_event if profile == "event" else self.navigate,
        )
        if profile == "event" and initial_task is not None:
            self.add_tool(
                "navigate_task",
                spec(
                    "navigate_task",
                    "Run the local navigation loop for the complete current user task verbatim. "
                    "Use this to avoid copying or shortening the task. Returns the same event and fresh RGB as navigate. "
                    "Choose navigate with an explicit instruction for a different local goal or recovery.",
                    {"restart": {"type": "boolean", "default": False}, "plan": PLAN_SCHEMA},
                ),
                self.navigate_task,
            )
        if profile == "legacy":
            self.add_tool(
                "update_plan",
                spec(
                    "update_plan",
                    "Save episode-local progress based only on observed evidence.",
                    PLAN_PROPERTIES,
                    tuple(PLAN_PROPERTIES),
                ),
                self.update_plan,
            )
        if plan_tools:
            self.add_tool(
                "advance_goal",
                spec(
                    "advance_goal",
                    "Use only when your observation confirms the first remaining goal is complete. "
                    "Record that goal as completed without rewriting the history. If goals remain, "
                    "run the next goal using instruction when supplied, otherwise its exact saved text. "
                    "If no goals remain, request final STOP. Evidence is your judgement, not a success detector. "
                    "Use navigate with a revised plan when correction or replanning is needed.",
                    {
                        "evidence": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "instruction": {"type": "string", "minLength": 1, "maxLength": 10000},
                    },
                    ("evidence",),
                ),
                self.advance_goal,
            )
        self.add_tool(
            "finish",
            spec(
                "finish",
                "End the task; success requests final STOP, subject to independent evaluation.",
                {
                    "status": {"type": "string", "enum": ["success", "failure", "stuck"]},
                    "summary": {"type": "string", "minLength": 1},
                    **({"plan": PLAN_SCHEMA} if profile == "event" else {}),
                },
                ("status", "summary"),
            ),
            self.finish,
        )
        if source_tools:
            self.add_tool(
                "plan_task",
                spec(
                    "plan_task",
                    "Select ordered local goals from indexed_task using inclusive one-based word ranges. "
                    "Copy those exact spans into the remaining plan, retain completed history, and navigate "
                    "the first goal until an event. Choose boundaries that preserve all route constraints.",
                    {
                        "spans": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "integer", "minimum": 1},
                            },
                        }
                    },
                    ("spans",),
                ),
                self.plan_task,
            )
            self.add_tool(
                "continue_goal",
                spec(
                    "continue_goal",
                    "Continue the first saved remaining goal verbatim until an event. "
                    "Preserve its navigation history unless restart is requested or the local skill stopped.",
                    {"restart": {"type": "boolean", "default": False}},
                ),
                self.continue_goal,
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
        """Track inline tool ownership so a signal never waits on its own thread."""
        previous = self.is_executing_on_current_thread()
        self._tool_thread.active = True
        try:
            return self._execute_navigation_tool(name, input_dict)
        finally:
            self._tool_thread.active = previous

    def is_executing_on_current_thread(self) -> bool:
        """Whether this caller must unwind a tool before waiting for cancellation."""
        return getattr(self._tool_thread, "active", False)

    def request_cancel(self) -> None:
        """Cancel this episode without blocking the thread executing its current tool."""
        self._episode_cancelled.set()

    def raise_if_cancelled(self) -> None:
        """Check episode cancellation and RPent's active-operation cancellation."""
        if self._episode_cancelled.is_set():
            raise ToolCancelled("navigation episode interrupted")
        super().raise_if_cancelled()

    def _execute_navigation_tool(self, name, input_dict):
        started = time.perf_counter()
        self.tool_calls += 1
        limit = self.max_tool_calls if self.profile == "event" else self.max_tool_calls + 1
        if self.tool_calls >= limit and name != "finish":
            name, input_dict = "finish", {"status": "stuck", "summary": "Tool budget exhausted"}
        result = super().execute_tool(name, input_dict)
        public = {key: value for key, value in result.result.items() if key not in ("_image_bytes", "_image_mime_type")}
        # RPent's generic ToolResult defaults to PNG. Keep the real JPEG MIME
        # on the wire, without requiring a fork of the pinned RPent dependency.
        for block in result.content_blocks:
            if block["type"] == "image":
                block["source"]["media_type"] = result.result.get("_image_mime_type", "image/png")
            elif block["type"] == "text":
                block["text"] = json.dumps(public, ensure_ascii=False)
        image = result.result.get("_image_bytes", b"")
        image_path = None
        if image:
            extension = "jpg" if result.result.get("_image_mime_type") == "image/jpeg" else "png"
            image_path = f"workspace/tool-observation-{self.tool_calls:04d}.{extension}"
            target = self.output / image_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image)
        with (self.output / "tools.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "tool": name,
                        "input": input_dict,
                        "result": public,
                        "elapsed_s": time.perf_counter() - started,
                        "image_bytes": len(image),
                        "image_path": image_path,
                        "image_sha256": hashlib.sha256(image).hexdigest() if image else None,
                    }
                )
                + "\n"
            )
        return result

    def _feedback(self, feedback: Feedback):
        if self.profile == "event":
            feedback = planner_image(feedback)
        return {
            "_image_bytes": feedback.image,
            "_image_mime_type": feedback.mime_type,
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

    def initial_observation(self):
        """Attach a public image to the initial task without an LLM tool round."""
        started = time.perf_counter()
        feedback = self.observe()
        workspace = self.output / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / "initial-observation.jpg"
        path.write_bytes(feedback.pop("_image_bytes"))
        feedback.pop("_image_mime_type")
        (self.output / "initial-observation.json").write_text(
            json.dumps(
                {
                    "feedback": feedback,
                    "elapsed_s": time.perf_counter() - started,
                    "image_bytes": path.stat().st_size,
                }
            )
            + "\n"
        )
        return path, feedback

    @readonly
    def observe(self):
        return self._feedback(self.skill.observe())

    def navigate(self, instruction, max_chunks=4, restart=False):
        if type(max_chunks) is not int or not 1 <= max_chunks <= 8:
            raise ValueError("max_chunks must be in [1,8]")
        return self._feedback(self.skill.navigate(instruction, max_chunks=max_chunks, restart=restart))

    def set_task(self, task: str) -> None:
        """Update the literal task reference without inventing progress or executing motion."""
        if not isinstance(task, str) or not task.strip() or len(task) > 10000:
            raise ValueError("task reference requires 1-10000 characters")
        self.task = task

    def decision_context(self) -> dict:
        """Expose literal word references only when explicitly enabled."""
        if not self.source_tools:
            return {}
        if self.task is None:
            raise ValueError("source references require an initialized task")
        return {"indexed_task": indexed_task(self.task)}

    def plan_task(self, spans: list[list[int]]) -> dict:
        """Save model-selected task spans and start their first local goal."""
        if self.task is None:
            raise ValueError("plan_task requires an initialized task")
        goals = select_task_spans(self.task, spans)
        plan = {
            "completed": list(self.plan.get("completed", [])),
            "remaining": goals,
            "observations": self.plan.get("observations", ""),
        }
        return self.navigate_event(goals[0], plan=plan)

    def continue_goal(self, restart: bool = False) -> dict:
        """Continue the model's current saved goal without rewriting it."""
        remaining = self.plan.get("remaining", [])
        if not remaining:
            raise ValueError("continue_goal requires a saved remaining goal")
        return self.navigate_event(remaining[0], restart=restart)

    def navigate_task(self, restart: bool = False, plan: dict | None = None) -> dict:
        """Execute the model-selected complete-task tool using the current task verbatim."""
        if self.task is None:
            raise ValueError("navigate_task requires an initialized task")
        return self.navigate_event(self.task, restart=restart, plan=plan)

    def navigate_event(self, instruction, restart=False, plan=None):
        if plan is not None:
            self.update_plan(**plan)
        return self._feedback(
            self.skill.navigate_until_event(
                instruction,
                restart=restart,
                event_policy=self.event_policy,
            )
        )

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

    def finish(self, status, summary, plan=None):
        if status not in ("success", "failure", "stuck") or not isinstance(summary, str) or not summary.strip():
            raise ValueError("finish requires a supported status and evidence summary")
        effective_plan = self.plan if plan is None else plan
        if status == "success" and isinstance(effective_plan, dict) and effective_plan.get("remaining"):
            raise ValueError("success contradicts working memory: update the plan or advance the remaining goal")
        if plan is not None:
            self.update_plan(**plan)
        self.skill.finish(success=status == "success")
        self.finish_result = {"_finish": True, "status": status, "summary": summary, "task_success": "unverified"}
        return self.finish_result

    def advance_goal(self, evidence: str, instruction: str | None = None) -> dict:
        """Apply the model's completion judgement to one saved goal, then continue.

        This moves exactly the first remaining entry into completed history. It
        performs no visual judgement and never fills in an absent plan. Explicit
        replanning remains available through ``navigate`` with a replacement plan.
        """
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 2000:
            raise ValueError("advance_goal requires 1-2000 characters of observed evidence")
        if instruction is not None and (
            not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 10000
        ):
            raise ValueError("instruction must contain 1-10000 characters")
        remaining = self.plan.get("remaining", [])
        if not remaining:
            raise ValueError("advance_goal requires a saved remaining goal")
        if len(remaining) == 1 and instruction is not None:
            raise ValueError("no next goal remains; omit instruction to finish")
        plan = {
            "completed": [*self.plan.get("completed", []), remaining[0]],
            "remaining": remaining[1:],
            "observations": evidence,
        }
        if not plan["remaining"]:
            return self.finish("success", evidence, plan=plan)
        return self.navigate_event(instruction or plan["remaining"][0], plan=plan)

    def solved(self):
        # Planner claims are never simulator ground truth.
        return False

    def close(self):
        self.skill.close()
