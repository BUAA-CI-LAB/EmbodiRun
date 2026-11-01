"""Watchdog events, preserved policy history, and compressed planner-only feedback."""

import base64
import io
import json

import pytest
from agents.rpent.navigation.events import EventPolicy
from agents.rpent.navigation.images import planner_image
from agents.rpent.navigation.skill import Feedback, NavigationSkill
from PIL import Image

from embodirun.application.navigation.chunk_session import NavigationChunk
from embodirun.robots.mobile_base.discrete import NavigationCommand, NavigationCommandKind
from embodirun.robots.mobile_base.frames import EncodedRGBFrame
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.robots.mobile_base.request import NavigationRequest


def picture(value):
    output = io.BytesIO()
    Image.new("RGB", (640, 480), (value, value, value)).save(output, format="PNG")
    return output.getvalue()


class Policy:
    def __init__(self):
        self.requests = []
        self.closed = False

    def plan(self, request):
        self.requests.append(request)
        return NavigationChunk(
            request.observation.sequence, (NavigationCommand(NavigationCommandKind.MOVE_FORWARD, distance_m=0.25),)
        )

    def close(self):
        self.closed = True


class Backend:
    def __init__(self, frames, *, stop_at=None, collisions=0):
        self.feedback = Feedback(picture(0), primitive_steps=0)
        self.frames = iter(frames)
        self.stop_at, self.collisions = stop_at, collisions
        self.executions = 0
        self.finish_calls = []

    def observe(self):
        return self.feedback

    def run(self, policy, instruction, episode_id, max_chunks, *, terminal_stop):
        for _ in range(max_chunks):
            frame = EncodedRGBFrame(self.executions, float(self.executions), self.feedback.image, "image/png")
            policy.plan(
                NavigationRequest(instruction, NavigationObservation(episode_id, self.executions, True, (frame,)))
            )
            self.executions += 1
            stopped = self.executions == self.stop_at
            self.feedback = Feedback(
                picture(next(self.frames)),
                reason="local_stop" if stopped else "chunk_limit",
                actions=(0,) if stopped else (1, 1, 1, 1),
                collisions=self.collisions,
                primitive_steps=self.executions * 4,
            )
            if stopped:
                break
        return self.feedback

    def finish(self, *, success):
        self.finish_calls.append(success)


def skill_for(frames, *, stop_at=None, collisions=0, **kwargs):
    backend = Backend(frames, stop_at=stop_at, collisions=collisions)
    policies = []

    def factory():
        policy = Policy()
        policies.append(policy)
        return policy

    return NavigationSkill(backend, factory, **kwargs), backend, policies


def test_checkpoint_keeps_visual_feedback_and_recurrent_session():
    skill, backend, policies = skill_for([20, 40, 60, 80, 100])
    feedback = skill.navigate_until_event("Go through the hall", event_policy=EventPolicy(max_chunks=3))
    assert feedback.reason == "checkpoint" and len(feedback.actions) == 12
    skill.navigate_until_event("Go through the hall", event_policy=EventPolicy(max_chunks=2))
    assert backend.executions == skill.model_calls == 5 and len(policies) == 1
    assert [r.observation.reset for r in policies[0].requests] == [True, False, False, False, False]
    assert len({r.observation.latest_rgb.data for r in policies[0].requests}) == 5


@pytest.mark.parametrize(
    ("frames", "collisions", "reason", "calls"),
    [([0] * 8, 0, "stalled", 4), ([20] * 8, 4, "blocked", 1), ([20, 40, 60, 0, 80], 0, "loop", 4)],
)
def test_watchdog_returns_before_remaining_chunks(frames, collisions, reason, calls):
    skill, backend, _ = skill_for(frames, collisions=collisions)
    feedback = skill.navigate_until_event("Reach the doorway")
    assert feedback.reason == reason and backend.executions == calls
    assert not backend.finish_calls and not skill.finished


def test_local_stop_returns_immediately_without_global_success():
    skill, backend, policies = skill_for([20, 40, 60], stop_at=2)
    feedback = skill.navigate_until_event("Reach the door")
    assert feedback.reason == "local_stop" and backend.executions == 2
    assert not skill.finished and not backend.finish_calls
    skill.navigate_until_event("Reach the door", event_policy=EventPolicy(max_chunks=1))
    assert len(policies) == 2 and policies[0].closed
    assert policies[1].requests[0].observation.reset


def test_event_loop_respects_global_budget_and_cancellation():
    skill, backend, _ = skill_for([20, 40, 60, 80], max_chunks=3)
    result = skill.navigate_until_event("Reach the door")
    assert result.exhausted and result.reason == "budget" and backend.executions == 3
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("cancelled")

    skill, backend, _ = skill_for([20, 40, 60, 80], check_cancelled=cancel)
    with pytest.raises(InterruptedError, match="cancelled"):
        skill.navigate_until_event("Reach the door")
    assert backend.executions == 2 and skill.failed
    with pytest.raises(RuntimeError, match="failed execution"):
        skill.navigate("Do not retry")


def test_compression_preserves_policy_image_and_resolution():
    source = Feedback(picture(80))
    compressed = planner_image(source)
    assert source.mime_type == "image/png" and source.image.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(compressed.image)) as decoded:
        assert decoded.format == "JPEG" and decoded.size == (640, 480)
    assert planner_image(compressed) is compressed


def test_event_tool_merges_plan_and_emits_actual_jpeg(tmp_path):
    pytest.importorskip("rpent")
    from agents.rpent.navigation.toolkit import NavigationToolkit

    skill, backend, _ = skill_for([20], stop_at=1)
    toolkit = NavigationToolkit(skill, tmp_path)
    assert {spec["name"] for spec in toolkit.get_tools_spec()} == {"observe", "navigate", "finish"}
    plan = {"completed": [], "remaining": ["Reach the doorway"], "observations": "Door visible"}
    result = toolkit.execute_tool("navigate", {"instruction": "Reach the doorway", "plan": plan})
    assert result.result["plan"] == plan and toolkit.tool_calls == 1
    assert json.loads((tmp_path / "memory/working.json").read_text()) == plan
    block = next(b for b in result.content_blocks if b["type"] == "image")
    assert block["source"]["media_type"] == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as image:
        assert image.format == "JPEG"
    assert backend.feedback.mime_type == "image/png"
    public = json.loads(next(b["text"] for b in result.content_blocks if b["type"] == "text"))
    assert not any(k.startswith("_image") for k in public)
    assert "distance_to_goal_m" not in public and "position" not in public


def test_invalid_combined_plan_cannot_start_motion(tmp_path):
    pytest.importorskip("rpent")
    from agents.rpent.navigation.toolkit import NavigationToolkit

    skill, backend, _ = skill_for([20])
    toolkit = NavigationToolkit(skill, tmp_path)
    result = toolkit.execute_tool(
        "navigate",
        {
            "instruction": "Reach the door",
            "plan": {"completed": "wrong", "remaining": [], "observations": ""},
        },
    )
    assert result.result["error"] and backend.executions == 0
    assert toolkit.plan == {} and not (tmp_path / "memory/working.json").exists()


def test_event_tool_budget_reserves_the_terminal_call(tmp_path):
    pytest.importorskip("rpent")
    from agents.rpent.navigation.toolkit import NavigationToolkit

    skill, backend, _ = skill_for([20])
    toolkit = NavigationToolkit(skill, tmp_path, max_tool_calls=2)
    toolkit.execute_tool("observe", {})
    result = toolkit.execute_tool("navigate", {"instruction": "Reach the door"})
    assert result.is_finish and result.result["status"] == "stuck"
    assert toolkit.tool_calls == 2 and backend.executions == 0
    assert backend.finish_calls == [False]


def test_initial_frame_uses_native_image_input_without_a_tool_round(tmp_path, monkeypatch):
    pytest.importorskip("rpent")
    from agents.rpent.navigation.planner import NavigationPlanner
    from agents.rpent.navigation.toolkit import NavigationToolkit
    from openai_codex import LocalImageInput, TextInput
    from rpent.planner.codex import CodexPlanner

    skill, backend, _ = skill_for([20])
    toolkit = NavigationToolkit(skill, tmp_path)
    path, feedback = toolkit.initial_observation()
    assert toolkit.tool_calls == backend.executions == 0
    assert feedback["primitive_steps"] == 0 and not any(k.startswith("_image") for k in feedback)
    received = []
    monkeypatch.setattr(CodexPlanner, "_run_session", lambda _self, prompt: received.extend(prompt))
    NavigationPlanner(tmp_path, model="unused", initial_image=path)._run_session("Reach the door")
    assert received == [TextInput("Reach the door"), LocalImageInput(str(path))]
    with Image.open(path) as image:
        assert image.format == "JPEG" and image.size == (640, 480)
