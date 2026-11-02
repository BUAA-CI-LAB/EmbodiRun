"""Literal references must not invent progress or corrupt a saved route."""

import json
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("rpent")
pytest.importorskip("httpx")
pytest.importorskip("jsonschema")

from agents.rpent.navigation.events import EventPolicy
from agents.rpent.navigation.shared_model import SharedModelPlanner
from agents.rpent.navigation.task_source import indexed_task, select_task_spans
from agents.rpent.navigation.toolkit import NavigationToolkit

from tests.test_rpent_navigation_events import skill_for
from tests.test_rpent_shared_model import Tools, Transport


def test_task_spans_copy_punctuation_whitespace_and_repeated_visits():
    task = "  Go   to kitchen.\nThen return to office.  "
    assert indexed_task(task) == "1: Go\n2: to\n3: kitchen.\n4: Then\n5: return\n6: to\n7: office."
    assert select_task_spans(task, [[1, 3], [5, 7], [1, 3]]) == [
        "Go   to kitchen.",
        "return to office.",
        "Go   to kitchen.",
    ]


@pytest.mark.parametrize("spans", [[], [[0, 2]], [[2, 1]], [[1, 100]], [[True, 2]], [[1]], [[1, 2]] * 17])
def test_bad_task_spans_cannot_save_memory_or_execute_motion(tmp_path, spans):
    skill, backend, _ = skill_for([])
    toolkit = NavigationToolkit(skill, tmp_path, initial_task="Go to the kitchen.", source_tools=True)
    memory = (tmp_path / "memory/working.json").read_bytes()
    result = toolkit.execute_tool("plan_task", {"spans": spans})
    assert "error" in result.result and not result.is_finish
    assert backend.executions == 0 and not backend.finish_calls
    assert (tmp_path / "memory/working.json").read_bytes() == memory


def test_selected_plan_and_current_goal_preserve_route_and_policy_history(tmp_path):
    skill, backend, policies = skill_for([20, 40])
    toolkit = NavigationToolkit(
        skill,
        tmp_path,
        initial_task="Go   to kitchen. Then return to office.",
        source_tools=True,
        event_policy=EventPolicy(max_chunks=1),
    )
    toolkit.update_plan(completed=["foyer"], remaining=["old goal"], observations="Left the foyer.")
    first = toolkit.execute_tool("plan_task", {"spans": [[1, 3], [5, 7]]})
    assert first.result["reason"] == "checkpoint" and not first.is_finish
    assert toolkit.plan == {
        "completed": ["foyer"],
        "remaining": ["Go   to kitchen.", "return to office."],
        "observations": "Left the foyer.",
    }
    second = toolkit.execute_tool("continue_goal", {})
    assert second.result["reason"] == "checkpoint" and not second.is_finish
    assert backend.executions == 2 and len(policies) == 1
    assert [r.instruction for r in policies[0].requests] == ["Go   to kitchen."] * 2
    assert [r.observation.reset for r in policies[0].requests] == [True, False]
    assert toolkit.plan["completed"] == ["foyer"] and not backend.finish_calls


def test_current_goal_missing_is_an_error_without_motion(tmp_path):
    skill, backend, _ = skill_for([])
    toolkit = NavigationToolkit(skill, tmp_path, initial_task="Go to the kitchen.", source_tools=True)
    toolkit.update_plan(completed=["kitchen"], remaining=[], observations="At the counter.")
    result = toolkit.execute_tool("continue_goal", {})
    assert "error" in result.result and backend.executions == 0 and not backend.finish_calls


def test_task_source_follows_steering_and_old_span_response_is_discarded(tmp_path):
    updates = queue.Queue()
    toolkit = NavigationToolkit(SimpleNamespace(), tmp_path, initial_task="Go left.", source_tools=True)
    executed = []
    toolkit.navigate_event = lambda instruction, **kwargs: executed.append(instruction) or {"reason": "checkpoint"}
    transport = Transport(
        [
            '{"name":"plan_task","arguments":{"spans":[[1,2]]}}',
            '{"name":"plan_task","arguments":{"spans":[[5,7]]}}',
        ]
    )
    original = transport.complete

    def complete(*args, **kwargs):
        response = original(*args, **kwargs)
        if len(transport.requests) == 1:
            updates.put("Visit the office.")
        return response

    transport.complete = complete
    result = SharedModelPlanner(transport, allowed_tools=("plan_task",)).solve(
        system_prompt="use tools", user_message="Go left.", toolkit=toolkit, max_turns=2, input_queue=updates
    )
    assert result.stats["stale_responses"] == 1 and executed == ["Visit the office."]
    state = json.loads(transport.requests[-1][0][-1]["content"][-1]["text"])
    assert state["indexed_task"].endswith("5: Visit\n6: the\n7: office.")


def test_toolkit_extra_context_cannot_replace_task_state(tmp_path):
    tools = Tools(tmp_path)
    tools.decision_context = lambda: {"task": "replacement"}
    transport = Transport([])
    result = SharedModelPlanner(transport).solve(
        system_prompt="use tools", user_message="original task", toolkit=tools, max_turns=1
    )
    assert "reserved task state" in result.error and not transport.requests
