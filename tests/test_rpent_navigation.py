"""Session lifecycle and evaluator isolation at the optional navigation boundary."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from agents.rpent.navigation.skill import Feedback, NavigationSkill

from embodirun.application.navigation.chunk_session import NavigationChunk
from embodirun.robots.mobile_base.discrete import NavigationCommand, NavigationCommandKind
from embodirun.robots.mobile_base.frames import EncodedRGBFrame
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.robots.mobile_base.request import NavigationRequest


class Policy:
    def __init__(self):
        self.requests = []
        self.closed = False

    def plan(self, request):
        self.requests.append(request)
        return NavigationChunk(request.observation.sequence, (NavigationCommand(NavigationCommandKind.STOP),))

    def close(self):
        self.closed = True


class Backend:
    def __init__(self):
        self.feedback = Feedback(b"png")
        self.stops = []
        self.executions = 0

    def observe(self):
        return self.feedback

    def run(self, policy, instruction, episode_id, max_chunks, *, terminal_stop):
        for index in range(max_chunks):
            frame = EncodedRGBFrame(index, float(index), b"png", "image/png")
            policy.plan(NavigationRequest(instruction, NavigationObservation(episode_id, index, True, (frame,))))
            self.executions += 1
        return self.feedback

    def finish(self, *, success):
        self.stops.append(success)


def make_skill(**kwargs):
    policies = []

    def factory():
        policy = Policy()
        policies.append(policy)
        return policy

    backend = Backend()
    return NavigationSkill(backend, factory, **kwargs), backend, policies


def test_continue_preserves_history_but_new_subgoal_resets():
    skill, backend, policies = make_skill()
    skill.navigate("Go to the doorway", max_chunks=2)
    skill.navigate("Go to the doorway", max_chunks=1)
    assert [r.observation.reset for r in policies[0].requests] == [True, False, False]
    assert len({r.observation.episode_id for r in policies[0].requests}) == 1
    skill.navigate("Turn right into the hall", max_chunks=1)
    assert policies[0].closed
    assert policies[1].requests[0].observation.reset
    assert policies[0].requests[0].observation.episode_id != policies[1].requests[0].observation.episode_id
    assert skill.model_calls == 4
    skill.close()
    assert policies[1].closed


def test_local_stop_needs_explicit_global_finish():
    skill, backend, policies = make_skill()
    backend.feedback = replace(backend.feedback, reason="local_stop")
    skill.navigate("Find the door", max_chunks=1)
    assert not backend.stops and not skill.finished
    backend.feedback = replace(backend.feedback, reason="observation")
    skill.observe()
    skill.navigate("Find the door", max_chunks=1)
    assert policies[0].closed and len(policies) == 2
    skill.finish(success=True)
    assert backend.stops == [True]
    with pytest.raises(RuntimeError, match="closed"):
        skill.navigate("Keep going")


def test_cumulative_budget_cannot_be_reset_by_changing_goal():
    skill, backend, policies = make_skill(max_chunks=3)
    skill.navigate("door", max_chunks=2)
    skill.navigate("hall", max_chunks=2)
    feedback = skill.navigate("room", max_chunks=2)
    assert feedback.exhausted and backend.executions == 3
    assert skill.model_calls == 3 and len(policies) == 2


def test_cancel_before_inference_prevents_motion_and_poisoned_retry():
    def cancelled():
        raise InterruptedError("cancelled")

    skill, backend, policies = make_skill(check_cancelled=cancelled)
    with pytest.raises(InterruptedError):
        skill.navigate("door", max_chunks=1)
    assert backend.executions == 0 and not policies[0].requests
    with pytest.raises(RuntimeError, match="failed execution"):
        skill.navigate("door", max_chunks=1)
    skill.close()


def test_simulator_metrics_are_not_agent_observations():
    path = Path(__file__).resolve().parents[1] / "benchmarks/rpent-navigation/simulator.py"
    spec = importlib.util.spec_from_file_location("rpent_habitat_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class PrivateSimulator:
        def request(self, _):
            return {
                "png": "aW1hZ2U=",
                "steps": 4,
                "position": [10, 20, 30],
                "goals": [{"position": [12, 20, 30]}],
                "distance_to_goal_m": 2,
                "shortest_path_m": 15,
                "success": True,
                "stopped": False,
            }

    backend = module.HabitatBackend(PrivateSimulator(), {}, "unused")
    public = backend.observe()
    assert public.png == b"image" and public.primitive_steps == 4
    for key in ("position", "goals", "distance_to_goal_m", "shortest_path_m", "success"):
        assert not hasattr(public, key)


def test_toolkit_records_finish_without_claiming_measured_success(tmp_path):
    pytest.importorskip("rpent")
    from agents.rpent.navigation.toolkit import NavigationToolkit

    skill, backend, _ = make_skill()
    toolkit = NavigationToolkit(skill, tmp_path)
    assert {spec["name"] for spec in toolkit.get_tools_spec()} == {"observe", "navigate", "update_plan", "finish"}
    result = toolkit.execute_tool("finish", {"status": "success", "summary": "door visible"})
    assert result.is_finish and result.result["task_success"] == "unverified"
    assert backend.stops == [True] and not toolkit.solved()


def test_tool_budget_aborts_without_false_stop_success(tmp_path):
    pytest.importorskip("rpent")
    from agents.rpent.navigation.toolkit import NavigationToolkit

    skill, backend, _ = make_skill()
    toolkit = NavigationToolkit(skill, tmp_path, max_tool_calls=1)
    toolkit.execute_tool("observe", {})
    result = toolkit.execute_tool("navigate", {"instruction": "door"})
    assert result.is_finish and result.result["status"] == "stuck"
    assert backend.stops == [False] and backend.executions == 0
