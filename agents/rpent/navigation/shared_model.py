"""Experimental RPent planner borrowing the navigator's resident model service.

The transport must attest the same model instance for text and navigation. This
module owns no weights and cannot fall back to another model or provider.
"""

from __future__ import annotations

import base64
import json
import math
import queue
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from jsonschema import Draft202012Validator, ValidationError
from rpent.planner.base import PlannerResult
from rpent.tools.toolkit import Toolkit


@dataclass(frozen=True)
class TextReply:
    """Raw output, identity, tokens and distinct client/server wall times.

    ``elapsed_s`` measures the complete client request. ``service_elapsed_s``
    optionally measures server handling, including scheduling and preprocessing;
    neither value is pure GPU execution time.
    """

    text: str
    model_instance: str
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    service_elapsed_s: float | None = None


class TextTransport(Protocol):
    """A text view of the exact model service used by navigation."""

    model_instance: str

    def complete(self, messages: list[dict], image: bytes | None, *, timeout_s: float) -> TextReply:
        """Generate from bounded context; never execute a tool inside inference."""
        ...


class SharedModelHttpClient:
    """Use an opt-in text capability on the existing navigation HTTP endpoint."""

    def __init__(
        self,
        endpoint: str,
        token: str | None,
        *,
        max_new_tokens: int = 192,
        expected_model_instance: str | None = None,
    ) -> None:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be in [1,512]")
        self.max_new_tokens = max_new_tokens
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = httpx.Client(base_url=endpoint.rstrip("/"), headers=headers, trust_env=False)
        try:
            response = self.client.get("/v1/capabilities", timeout=15)
            response.raise_for_status()
            capability = response.json()["adapter"]["shared_text"]
            if capability.get("schema") != "vvla.shared_text.v1" or capability.get("shared_parameters") is not True:
                raise ValueError("navigation service does not attest shared model parameters")
            self.model_instance = capability["model_instance"]
            if not isinstance(self.model_instance, str) or not self.model_instance:
                raise ValueError("model instance identity is required")
            if expected_model_instance is not None and self.model_instance != expected_model_instance:
                raise ValueError("text endpoint differs from the navigation model instance")
        except BaseException:
            self.client.close()
            raise

    def complete(self, messages: list[dict], image: bytes | None, *, timeout_s: float) -> TextReply:
        """Request raw text from the attested model; mismatched identity fails closed."""
        started = time.perf_counter()
        response = self.client.post(
            "/v1/text",
            json={
                "schema": "vvla.text.request.v1",
                "request_id": uuid.uuid4().hex,
                "model_instance": self.model_instance,
                "messages": messages,
                "image_base64": base64.b64encode(image).decode() if image else None,
                "max_new_tokens": self.max_new_tokens,
                "timeout_s": timeout_s,
            },
            timeout=timeout_s + 5,
        )
        response.raise_for_status()
        value = response.json()
        if value.get("schema") != "vvla.text.response.v1" or value.get("model_instance") != self.model_instance:
            raise RuntimeError("text response came from a different model instance")
        service_elapsed = value.get("elapsed_s")
        if service_elapsed is not None and (
            type(service_elapsed) not in (int, float) or not math.isfinite(service_elapsed) or service_elapsed < 0
        ):
            raise RuntimeError("text response has invalid service timing")
        return TextReply(
            value["text"],
            self.model_instance,
            value["input_tokens"],
            len(value["token_ids"]),
            time.perf_counter() - started,
            service_elapsed,
        )

    def close(self) -> None:
        """Close only this borrowed transport, never the model service."""
        self.client.close()


def parse_tool_call(text: str, schemas: dict[str, dict]) -> tuple[str, dict]:
    """Validate exact model JSON against RPent schemas without filling arguments."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise ValueError("return one JSON object with name and arguments")
    name, arguments = value["name"], value["arguments"]
    if not isinstance(name, str) or name not in schemas:
        raise ValueError("unknown tool name")
    Draft202012Validator(schemas[name]).validate(arguments)
    return name, arguments


def decision_feedback(feedback: dict, working_memory: dict) -> dict:
    """Keep task evidence and budgets while removing duplicate memory/telemetry.

    Complete tool output remains in the execution log. This view preserves
    action order, collisions, consumed steps, errors and all unknown fields.
    """
    value = {key: item for key, item in feedback.items() if key not in {"model_calls", "subgoals"}}
    if "plan" in value and value["plan"] == working_memory:
        del value["plan"]
    return value


class SharedModelPlanner:
    """RPent's Planner contract with bounded context and one model identity.

    Tool execution is serial and uses RPent's cancellation/memory/logging path.
    Only the latest image is sent; the original task and model-authored plan
    remain in every context. A local STOP is feedback, never automatic success.
    """

    def __init__(
        self,
        transport: TextTransport,
        *,
        initial_image: bytes | None = None,
        initial_feedback: dict | None = None,
        timeout_s: float = 600,
        check_cancelled: Callable[[], None] = lambda: None,
        allowed_tools: tuple[str, ...] | None = None,
        examples: Sequence[tuple[str, str]] = (),
    ) -> None:
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.transport = transport
        self.initial_image = initial_image
        self.initial_feedback = initial_feedback or {}
        self.timeout_s = timeout_s
        self.check_cancelled = check_cancelled
        self.allowed_tools = allowed_tools
        self.examples = tuple(examples)

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: Any = None,
    ) -> PlannerResult:
        """Decide and dispatch actual RPent tools; errors never become successful motion."""
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be positive")
        if dashboard_interaction is not None:
            raise NotImplementedError("shared-model planner does not yet support Dashboard steering")
        specs = toolkit.get_tools_spec()
        if self.allowed_tools is not None:
            available = {item["name"] for item in specs}
            if not self.allowed_tools or not set(self.allowed_tools) <= available:
                raise ValueError("allowed_tools must select registered RPent tools")
            specs = [item for item in specs if item["name"] in self.allowed_tools]
        schemas = {item["name"]: item["input_schema"] for item in specs}
        example_messages: list[dict[str, str]] = []
        for prompt, response in self.examples:
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError("examples must contain text prompts and raw JSON tool responses")
            parse_tool_call(response, schemas)
            example_messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
        system = (
            system_prompt + '\nReturn only one JSON tool call: {"name":"tool_name","arguments":{...}}. '
            "Use the exact argument names below. Do not emit movement commands or explanations. "
            "When previous_invalid_response is present, correct it using latest_feedback.format_error. Tools:\n"
            + json.dumps(specs, separators=(",", ":"))
        )
        deadline = time.monotonic() + self.timeout_s
        image, feedback = self.initial_image, dict(self.initial_feedback)
        transcript: list[dict] = []
        stats = {
            "turns_used": 0,
            "tool_calls": 0,
            "invalid_responses": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "request_elapsed_s": 0.0,
            "service_elapsed_s": 0.0,
            "service_timing_samples": 0,
            "model_instance": self.transport.model_instance,
            "stale_responses": 0,
        }
        previous_call: dict | None = None
        previous_invalid_response: str | None = None
        phase = "decision"
        task_revision = 0

        def drain_updates() -> bool:
            nonlocal user_message, task_revision, previous_invalid_response
            changed = False
            if input_queue is None:
                return changed
            while True:
                try:
                    item = input_queue.get_nowait()
                except queue.Empty:
                    return changed
                if item is None:
                    raise InterruptedError("planner cancelled by user input")
                user_message += "\nUser update: " + item
                task_revision += 1
                previous_invalid_response = None
                changed = True

        try:
            for _ in range(max_turns):
                self.check_cancelled()
                drain_updates()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("planner episode deadline expired")
                set_task = getattr(toolkit, "set_task", None)
                if callable(set_task):
                    set_task(user_message)
                working_memory = getattr(toolkit, "plan", {})
                state = {
                    "task": user_message,
                    "task_revision": task_revision,
                    "working_memory": working_memory,
                    "previous_tool_call": previous_call,
                    "latest_feedback": decision_feedback(feedback, working_memory),
                }
                context_provider = getattr(toolkit, "decision_context", None)
                if callable(context_provider):
                    extra = context_provider()
                    if not isinstance(extra, dict) or set(extra) & set(state):
                        raise ValueError("toolkit decision context must not replace reserved task state")
                    state.update(extra)
                if previous_invalid_response is not None:
                    state["previous_invalid_response"] = previous_invalid_response
                content = ([{"type": "image"}] if image else []) + [
                    {"type": "text", "text": json.dumps(state, separators=(",", ":"))}
                ]
                messages = [
                    {"role": "system", "content": system},
                    *example_messages,
                    {"role": "user", "content": content},
                ]
                phase = "transport"
                reply = self.transport.complete(messages, image, timeout_s=min(remaining, 60))
                phase = "decision"
                if reply.model_instance != self.transport.model_instance:
                    raise RuntimeError("model instance changed during the episode")
                stats["turns_used"] += 1
                stats["total_input_tokens"] += reply.input_tokens
                stats["total_output_tokens"] += reply.output_tokens
                stats["request_elapsed_s"] += reply.elapsed_s
                if reply.service_elapsed_s is not None:
                    stats["service_elapsed_s"] += reply.service_elapsed_s
                    stats["service_timing_samples"] += 1
                transcript.append({"role": "assistant", "content": reply.text, "state": state})
                self.check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("planner response arrived after the deadline")
                if drain_updates():
                    stats["stale_responses"] += 1
                    transcript[-1]["discarded"] = "task_revision_changed"
                    continue
                try:
                    name, arguments = parse_tool_call(reply.text, schemas)
                except (ValueError, ValidationError) as error:
                    stats["invalid_responses"] += 1
                    previous_invalid_response = reply.text[:4096]
                    feedback = {**feedback, "format_error": str(error)[:400]}
                    if stats["invalid_responses"] >= 2:
                        stats["failure_kind"] = "invalid_tool_call"
                        raise ValueError("model failed the actual tool schema twice") from error
                    continue
                previous_invalid_response = None
                previous_call = {"name": name, "arguments": arguments}
                phase = "tool"
                result = toolkit.execute_tool(name, arguments)
                phase = "decision"
                stats["tool_calls"] += 1
                # The navigation toolkit owns the public-feedback boundary.
                image = result.result.get("_image_bytes", image)
                feedback = {key: value for key, value in result.result.items() if not key.startswith("_")}
                transcript.append({"role": "tool", "name": name, "content": feedback})
                # Preserve what the tool actually did, but do not accept a late
                # finish after cancellation or the planner's task deadline.
                self.check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("tool response arrived after the episode deadline")
                if result.is_finish:
                    return PlannerResult(finish_result=feedback, messages=transcript, stats=stats)
            stats["failure_kind"] = "turn_budget"
            return PlannerResult(messages=transcript, stats=stats, error="planner turn budget exhausted")
        except Exception as error:
            stats.setdefault(
                "failure_kind", "episode_deadline" if isinstance(error, TimeoutError) and phase == "decision" else phase
            )
            return PlannerResult(messages=transcript, stats=stats, error=f"{type(error).__name__}: {error}")
