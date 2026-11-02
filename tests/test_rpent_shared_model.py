"""Actual RPent dispatch, schema rejection, cancellation and identity isolation."""

import json
import queue

import pytest

pytest.importorskip("rpent")
pytest.importorskip("httpx")
pytest.importorskip("jsonschema")

import httpx
from agents.rpent.navigation.shared_model import (
    SharedModelHttpClient,
    SharedModelPlanner,
    TextReply,
    decision_feedback,
    parse_tool_call,
)
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit


class Tools(Toolkit):
    def __init__(self, root):
        self.calls = []
        self.plan = {"remaining": ["kitchen", "study"]}
        super().__init__(dashboard_events=NullDashboardEventSink(), memory=MemoryManager(root), state=EnvState(root))
        self.add_tool(
            "navigate",
            {
                "name": "navigate",
                "input_schema": {
                    "type": "object",
                    "properties": {"instruction": {"type": "string"}},
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
            },
            self.navigate,
        )
        self.add_tool(
            "finish",
            {
                "name": "finish",
                "input_schema": {
                    "type": "object",
                    "properties": {"status": {"enum": ["success", "failure"]}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
            self.finish,
        )

    def _register_common_tools(self):
        pass

    def get_tools_spec(self):
        return [schema for schema, _ in self._tools.values()]

    def get_env_state(self, *, command, result, elapsed_s):
        return result

    def navigate(self, instruction):
        self.calls.append(instruction)
        return {"reason": "local_stop", "_image_bytes": b"second-image"}

    def finish(self, status):
        return {"_finish": True, "status": status, "task_success": "unverified"}


class Transport:
    model_instance = "one-model"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, image, *, timeout_s):
        self.requests.append((messages, image))
        return TextReply(next(self.replies), self.model_instance, 200, 20, 0.1)


@pytest.mark.parametrize("interruption", ["deadline", "cancel"])
def test_late_tool_finish_retains_execution_log_without_accepting_completion(tmp_path, monkeypatch, interruption):
    clock = [100.0]
    cancelled = False
    monkeypatch.setattr("agents.rpent.navigation.shared_model.time.monotonic", lambda: clock[0])

    class LateTools(Tools):
        def finish(self, status):
            nonlocal cancelled
            if interruption == "deadline":
                clock[0] += 2.0
            else:
                cancelled = True
            return super().finish(status)

    def check_cancelled():
        if cancelled:
            raise InterruptedError("cancelled during tool execution")

    tools = LateTools(tmp_path)
    transport = Transport(['{"name":"finish","arguments":{"status":"success"}}'])
    result = SharedModelPlanner(transport, timeout_s=1.0, check_cancelled=check_cancelled).solve(
        system_prompt="Navigate", user_message="Go to the door", toolkit=tools, max_turns=3
    )
    assert result.finish_result is None and result.error
    assert len(transport.requests) == result.stats["tool_calls"] == 1
    assert result.messages[-1]["role"] == "tool"
    assert result.messages[-1]["content"]["status"] == "success"
    if interruption == "deadline":
        assert result.stats["failure_kind"] == "episode_deadline"
    else:
        assert "cancelled during tool execution" in result.error


def test_context_compaction_preserves_execution_evidence_and_does_not_mutate_logs():
    plan = {"completed": [], "remaining": ["task"], "observations": ""}
    feedback = {
        "reason": "budget",
        "budget_exhausted": True,
        "primitive_steps": 500,
        "executed_actions": [1, 2, 1, 0],
        "collisions": 2,
        "remaining_tool_calls": 3,
        "plan": plan,
        "model_calls": 20,
        "subgoals": 2,
        "format_error": "missing instruction",
        "extra_observation": "closed door",
    }
    original = json.dumps(feedback, sort_keys=True)
    compact = decision_feedback(feedback, plan)
    assert compact == {key: value for key, value in feedback.items() if key not in {"plan", "model_calls", "subgoals"}}
    assert json.dumps(feedback, sort_keys=True) == original
    assert decision_feedback(feedback, {})["plan"] == plan


def test_rpent_tools_receive_model_arguments_and_local_stop_is_not_completion(tmp_path):
    tools = Tools(tmp_path)
    transport = Transport(
        [
            '{"name":"navigate","arguments":{"instruction":"enter kitchen"}}',
            '{"name":"navigate","arguments":{"instruction":"go to study"}}',
            '{"name":"finish","arguments":{"status":"success"}}',
        ]
    )
    result = SharedModelPlanner(transport, initial_image=b"first-image").solve(
        system_prompt="use tools",
        user_message="Visit kitchen then study",
        toolkit=tools,
        max_turns=3,
    )
    assert result.error is None and tools.calls == ["enter kitchen", "go to study"]
    assert result.finish_result["task_success"] == "unverified"
    assert result.stats["tool_calls"] == 3
    assert [image for _, image in transport.requests] == [b"first-image", b"second-image", b"second-image"]
    for messages, _ in transport.requests:
        state = json.loads(messages[-1]["content"][-1]["text"])
        assert state["task"] == "Visit kitchen then study" and state["working_memory"] == tools.plan
        assert "_image_bytes" not in state["latest_feedback"]


def test_invalid_model_schema_never_becomes_a_repaired_navigation_call(tmp_path):
    tools = Tools(tmp_path)
    transport = Transport(
        [
            '{"name":"navigate","arguments":{"direction":"left"}}',
            '{"name":"finish","arguments":{"success":true}}',
        ]
    )
    result = SharedModelPlanner(transport).solve(
        system_prompt="use tools",
        user_message="task",
        toolkit=tools,
        max_turns=5,
    )
    assert result.error and result.stats["invalid_responses"] == 2
    assert not tools.calls and not result.finish_result
    assert len(transport.requests) == 2
    retry_state = json.loads(transport.requests[1][0][-1]["content"][-1]["text"])
    assert retry_state["previous_invalid_response"] == '{"name":"navigate","arguments":{"direction":"left"}}'
    assert "format_error" in retry_state["latest_feedback"]


def test_planner_timing_keeps_client_wall_time_separate_from_partial_server_samples(tmp_path):
    tools = Tools(tmp_path)
    replies = iter(
        [
            TextReply('{"name":"navigate","arguments":{"instruction":"kitchen"}}', "one-model", 10, 5, 0.5, 0.2),
            TextReply('{"name":"finish","arguments":{"status":"success"}}', "one-model", 10, 5, 0.4),
        ]
    )
    transport = Transport([])
    transport.complete = lambda *args, **kwargs: next(replies)
    result = SharedModelPlanner(transport).solve(
        system_prompt="use tools", user_message="kitchen", toolkit=tools, max_turns=2
    )
    assert result.error is None
    assert result.stats["request_elapsed_s"] == pytest.approx(0.9)
    assert result.stats["service_elapsed_s"] == 0.2
    assert result.stats["service_timing_samples"] == 1
    assert "model_elapsed_s" not in result.stats


def test_format_examples_are_context_only_and_never_execute_or_update_memory(tmp_path):
    tools = Tools(tmp_path)
    transport = Transport(['{"name":"finish","arguments":{"status":"failure"}}'])
    demonstration = '{"name":"navigate","arguments":{"instruction":"visit the example office"}}'
    result = SharedModelPlanner(transport, examples=[("A different task", demonstration)]).solve(
        system_prompt="use tools", user_message="Current task", toolkit=tools, max_turns=1
    )
    assert result.error is None and not tools.calls
    assert tools.plan == {"remaining": ["kitchen", "study"]}
    messages = transport.requests[0][0]
    assert messages[1:3] == [
        {"role": "user", "content": "A different task"},
        {"role": "assistant", "content": demonstration},
    ]
    state = json.loads(messages[-1]["content"][-1]["text"])
    assert state["task"] == "Current task" and state["working_memory"] == tools.plan
    assert result.stats["tool_calls"] == 1


def test_navigation_examples_obey_the_current_registered_tool_contract(tmp_path):
    from types import SimpleNamespace

    from agents.rpent.navigation.toolkit import SHARED_TOOL_EXAMPLES, NavigationToolkit

    tools = NavigationToolkit(SimpleNamespace(), tmp_path)
    schemas = {item["name"]: item["input_schema"] for item in tools.get_tools_spec()}
    for _, response in SHARED_TOOL_EXAMPLES:
        parse_tool_call(response, schemas)


def test_initial_memory_records_only_the_given_unfulfilled_task(tmp_path):
    from types import SimpleNamespace

    from agents.rpent.navigation.toolkit import NavigationToolkit

    task = "Go past the kitchen and stop at the study door."
    toolkit = NavigationToolkit(SimpleNamespace(), tmp_path, initial_task=task)
    expected = {"completed": [], "remaining": [task], "observations": ""}
    assert toolkit.plan == expected
    assert json.loads((tmp_path / "memory" / "working.json").read_text()) == expected
    assert toolkit.tool_calls == 0 and toolkit.finish_result is None


def test_task_reference_is_model_selected_and_follows_steering_after_stale_response(tmp_path):
    from types import SimpleNamespace

    from agents.rpent.navigation.toolkit import NavigationToolkit

    updates = queue.Queue()
    toolkit = NavigationToolkit(SimpleNamespace(), tmp_path, initial_task="First task")
    executed = []
    toolkit.navigate_event = lambda instruction, **kwargs: executed.append(instruction) or {"reason": "checkpoint"}
    transport = Transport(['{"name":"navigate_task","arguments":{}}'] * 2)
    original_complete = transport.complete

    def complete(*args, **kwargs):
        reply = original_complete(*args, **kwargs)
        if len(transport.requests) == 1:
            updates.put("Continue to the office after the first task.")
        return reply

    transport.complete = complete
    result = SharedModelPlanner(transport, allowed_tools=("navigate_task",)).solve(
        system_prompt="use tools",
        user_message="First task",
        toolkit=toolkit,
        max_turns=2,
        input_queue=updates,
    )
    assert result.stats["stale_responses"] == 1 and result.stats["tool_calls"] == 1
    assert executed == ["First task\nUser update: Continue to the office after the first task."]
    assert toolkit.plan["completed"] == []


def test_invalid_demonstrations_fail_before_model_or_tool_work(tmp_path):
    tools = Tools(tmp_path)
    transport = Transport([])
    with pytest.raises(ValueError):
        SharedModelPlanner(transport, examples=[("task", "not JSON")]).solve(
            system_prompt="use tools", user_message="task", toolkit=tools, max_turns=1
        )
    assert not transport.requests and not tools.calls


def test_cancelled_planner_never_dispatches_a_late_model_response(tmp_path):
    tools = Tools(tmp_path)
    calls = 0

    def cancel_after_reply():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("cancelled")

    transport = Transport(['{"name":"navigate","arguments":{"instruction":"move"}}'])
    result = SharedModelPlanner(transport, check_cancelled=cancel_after_reply).solve(
        system_prompt="use tools",
        user_message="task",
        toolkit=tools,
        max_turns=1,
    )
    assert "cancelled" in result.error and not tools.calls


def test_foreign_model_identity_cannot_execute_a_tool(tmp_path):
    tools = Tools(tmp_path)
    transport = Transport([])
    transport.complete = lambda *args, **kwargs: TextReply(
        '{"name":"navigate","arguments":{"instruction":"move"}}',
        "other-model",
        1,
        1,
        0.1,
    )
    result = SharedModelPlanner(transport).solve(
        system_prompt="use tools",
        user_message="task",
        toolkit=tools,
        max_turns=1,
    )
    assert "model instance changed" in result.error and not tools.calls


def test_input_queue_cancel_prevents_inference(tmp_path):
    inputs = queue.Queue()
    inputs.put(None)
    transport = Transport([])
    result = SharedModelPlanner(transport).solve(
        system_prompt="use tools",
        user_message="task",
        toolkit=Tools(tmp_path),
        max_turns=1,
        input_queue=inputs,
    )
    assert "cancelled" in result.error and not transport.requests


def test_user_steering_during_inference_discards_the_old_decision(tmp_path):
    tools = Tools(tmp_path)
    inputs = queue.Queue()
    transport = Transport(
        [
            '{"name":"navigate","arguments":{"instruction":"old goal"}}',
            '{"name":"finish","arguments":{"status":"failure"}}',
        ]
    )
    complete = transport.complete

    def updated(*args, **kwargs):
        reply = complete(*args, **kwargs)
        if len(transport.requests) == 1:
            inputs.put("Cancel that route and finish as failure.")
        return reply

    transport.complete = updated
    result = SharedModelPlanner(transport).solve(
        system_prompt="use tools",
        user_message="old task",
        toolkit=tools,
        max_turns=2,
        input_queue=inputs,
    )
    assert result.error is None and not tools.calls
    assert result.stats["stale_responses"] == 1 and result.stats["total_output_tokens"] == 40
    assert result.messages[0]["discarded"] == "task_revision_changed"
    state = json.loads(transport.requests[1][0][-1]["content"][-1]["text"])
    assert state["task_revision"] == 1 and "Cancel that route" in state["task"]


def test_tool_subset_is_used_for_both_prompt_and_validation(tmp_path):
    tools = Tools(tmp_path)
    transport = Transport(['{"name":"navigate","arguments":{"instruction":"move"}}'] * 2)
    result = SharedModelPlanner(transport, allowed_tools=("finish",)).solve(
        system_prompt="use tools",
        user_message="task",
        toolkit=tools,
        max_turns=2,
    )
    assert not tools.calls and result.stats["failure_kind"] == "invalid_tool_call"
    assert '"name":"navigate"' not in transport.requests[0][0][0]["content"]


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, 0])
def test_nonfinite_or_invalid_episode_deadline_is_rejected(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        SharedModelPlanner(Transport([]), timeout_s=timeout)


def test_http_identity_is_checked_against_the_actual_navigation_executor(monkeypatch):
    def response(request):
        return httpx.Response(
            200,
            json={
                "adapter": {
                    "shared_text": {
                        "schema": "vvla.shared_text.v1",
                        "shared_parameters": True,
                        "model_instance": "different",
                    }
                }
            },
            request=request,
        )

    client_type = httpx.Client
    wires = []

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        wire = client_type(**kwargs, transport=httpx.MockTransport(response))
        wires.append(wire)
        return wire

    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(ValueError, match="differs from the navigation"):
        SharedModelHttpClient("http://model.test", None, expected_model_instance="navigation-instance")
    assert wires[0].is_closed


def test_http_text_keeps_raw_output_and_rejects_a_restarted_model(monkeypatch):
    calls = []

    def response(request):
        if request.method == "GET":
            value = {
                "adapter": {
                    "shared_text": {
                        "schema": "vvla.shared_text.v1",
                        "shared_parameters": True,
                        "model_instance": "one-model",
                    }
                }
            }
        else:
            calls.append(json.loads(request.content))
            value = {
                "schema": "vvla.text.response.v1",
                "model_instance": "one-model" if len(calls) == 1 else "restarted",
                "text": "not JSON",
                "token_ids": [1, 2],
                "input_tokens": 5,
                "elapsed_s": 0.25,
            }
        return httpx.Response(200, json=value, request=request)

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(response))
    )
    client = SharedModelHttpClient("http://model.test", None, expected_model_instance="one-model")
    try:
        reply = client.complete([{"role": "user", "content": "task"}], b"frame", timeout_s=1)
        assert reply.text == "not JSON" and reply.output_tokens == 2
        assert reply.service_elapsed_s == 0.25 and reply.elapsed_s >= 0
        assert calls[0]["model_instance"] == "one-model"
        with pytest.raises(RuntimeError, match="different model instance"):
            client.complete([{"role": "user", "content": "task"}], None, timeout_s=1)
    finally:
        client.close()


@pytest.mark.parametrize("raw", ["[]", '{"name":"unknown","arguments":{}}', "```json\n{}\n```"])
def test_tool_parser_rejects_invalid_or_unknown_calls(raw):
    with pytest.raises(ValueError):
        parse_tool_call(raw, {})
