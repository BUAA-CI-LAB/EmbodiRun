"""Go2 planner wiring with fake transports and no physical-device operations."""

import builtins
import io
import json
import signal
from types import SimpleNamespace

import pytest

pytest.importorskip("rpent")
pytest.importorskip("httpx")
pytest.importorskip("jsonschema")

from agents.rpent.navigation import go2, shared_model
from agents.rpent.navigation.skill import Feedback
from PIL import Image

from embodirun.application.navigation.chunk_session import NavigationChunk
from embodirun.robots.mobile_base.discrete import NavigationCommand, NavigationCommandKind
from embodirun.robots.mobile_base.frames import EncodedRGBFrame
from embodirun.robots.mobile_base.observation import NavigationObservation
from embodirun.robots.mobile_base.request import NavigationRequest


@pytest.fixture
def entrypoint(monkeypatch, tmp_path):
    config = SimpleNamespace(
        inference_url="http://navigation.invalid",
        inference_token_env="TEST_SHARED_MODEL_TOKEN",
        inference_timeout_s=1,
        camera_url="http://camera.invalid",
        camera_token_env="TEST_CAMERA_TOKEN",
        camera_timeout_s=1,
        control_url="http://control.invalid",
        control_token_env="TEST_CONTROL_TOKEN",
        control_timeout_s=1,
        robot_id="fake-go2",
        session=SimpleNamespace(max_chunks=32),
        require_endpoints=lambda: None,
    )
    client = object()
    observed = {
        "hardware_clients": 0,
        "runs": 0,
        "policy_calls": 0,
        "transport_closed": False,
        "navigation_client": None,
    }
    output = tmp_path / "result"
    args = ["--config", str(tmp_path / "config.json"), "--instruction", "Reach the study.", "--output", str(output)]
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "gray").save(buffer, format="PNG")
    frame_bytes = buffer.getvalue()
    replies = [{"name": "finish", "arguments": {"status": "failure", "summary": "Unexecuted."}}]
    monkeypatch.setattr(go2.NavigationConfig, "load", lambda _: config)
    monkeypatch.setattr(go2, "connect_client", lambda *a, **k: client)
    monkeypatch.setattr(go2, "check_capabilities", lambda _: {"shared_text": {"model_instance": "resident-model"}})

    def hardware_client(*args, **kwargs):
        observed["hardware_clients"] += 1
        return object()

    monkeypatch.setattr(go2, "Go2CameraClient", hardware_client)
    monkeypatch.setattr(go2, "Go2ControlClient", hardware_client)

    class Backend:
        def __init__(self, camera, base, config, *, execute, **kwargs):
            self.execute = execute

        def observe(self):
            return Feedback(frame_bytes)

        def run(self, policy, instruction, episode_id, max_chunks, *, terminal_stop):
            assert self.execute, "Read-only planning dispatched motion"
            observed["runs"] += 1
            if observed.get("on_run") is not None:
                observed["on_run"]()
            frame = EncodedRGBFrame(0, 0.0, frame_bytes, "image/png")
            request = NavigationRequest(instruction, NavigationObservation(episode_id, 0, True, (frame,)))
            policy.plan(request)
            return Feedback(frame_bytes, reason="local_stop", actions=(0,))

        finish = go2.Go2Backend.finish

    class Policy:
        def __init__(self, navigator_client, **kwargs):
            observed["navigation_client"] = navigator_client

        def plan(self, request):
            observed["policy_calls"] += 1
            return NavigationChunk(0, (NavigationCommand(NavigationCommandKind.STOP),))

        def close(self):
            pass

    class Transport:
        model_instance = "resident-model"

        def __init__(self, endpoint, token, *, expected_model_instance, max_new_tokens):
            assert endpoint == config.inference_url and expected_model_instance == self.model_instance
            assert max_new_tokens == 512

        def complete(self, messages, image, *, timeout_s):
            assert image
            tools = json.loads(messages[0]["content"].split("Tools:\n", 1)[1])
            observed["allowed_tools"] = {item["name"] for item in tools}
            return shared_model.TextReply(json.dumps(replies.pop(0)), self.model_instance, 10, 10, 0.001)

        def close(self):
            observed["transport_closed"] = True

    monkeypatch.setattr(go2, "Go2Backend", Backend)
    monkeypatch.setattr(go2, "ActiveVLNPolicy", Policy)
    monkeypatch.setattr(shared_model, "SharedModelHttpClient", Transport)
    original_import = builtins.__import__

    def no_cloud_import(name, *args, **kwargs):
        if name in {"planner", "openai_codex", "rpent.planner.codex", "agents.rpent.navigation.planner"}:
            raise AssertionError("Shared-model entrypoint imported the cloud planner")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cloud_import)
    return args, output, observed, replies, client


def test_shared_go2_is_default_and_read_only_without_execute(entrypoint):
    args, output, observed, _, _ = entrypoint
    assert go2.main(args) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["planner_backend"] == "shared-model" and result["motion_enabled"] is False
    assert result["stats"]["model_instance"] == "resident-model"
    assert observed["runs"] == 0 and observed["transport_closed"]
    assert observed["allowed_tools"] == {"observe", "finish"}


def test_shared_go2_execution_uses_attested_navigation_client(entrypoint):
    args, output, observed, replies, client = entrypoint
    replies.insert(0, {"name": "navigate_task", "arguments": {}})
    assert go2.main([*args, "--execute"]) == 0
    assert observed["navigation_client"] is client and observed["runs"] == 1
    assert observed["allowed_tools"] == {"navigate_task", "navigate", "advance_goal", "finish"}
    result = json.loads((output / "result.json").read_text())
    assert result["model_calls"] == 1 and result["motion_enabled"] is True
    assert observed["transport_closed"]


def test_missing_shared_capability_fails_before_hardware_clients(entrypoint, monkeypatch):
    args, _, observed, _, _ = entrypoint
    monkeypatch.setattr(go2, "check_capabilities", lambda _: {"model": "activevln"})
    with pytest.raises(RuntimeError, match="model-instance identity"):
        go2.main(args)
    assert observed["hardware_clients"] == 0


def test_read_only_model_cannot_dispatch_navigation(entrypoint):
    args, output, observed, replies, _ = entrypoint
    replies[:] = [{"name": "navigate_task", "arguments": {}}] * 2
    assert go2.main(args) == 1
    assert observed["runs"] == 0 and observed["transport_closed"]
    assert json.loads((output / "result.json").read_text())["stats"]["invalid_responses"] == 2


def test_read_only_finish_cannot_claim_success_by_clearing_the_plan(entrypoint):
    args, output, observed, replies, _ = entrypoint
    replies.insert(
        0,
        {
            "name": "finish",
            "arguments": {
                "status": "success",
                "summary": "Claimed without execution.",
                "plan": {"completed": ["Reach the study."], "remaining": [], "observations": ""},
            },
        },
    )
    go2.main(args)
    result = json.loads((output / "result.json").read_text())
    assert not result["finish"] or result["finish"]["status"] != "success"
    assert observed["runs"] == 0 and observed["transport_closed"]


def test_signal_inside_inline_tool_unwinds_before_waiting(entrypoint, monkeypatch):
    from agents.rpent.navigation.toolkit import NavigationToolkit

    args, _, observed, replies, _ = entrypoint
    replies.insert(0, {"name": "navigate_task", "arguments": {}})
    original_wait = NavigationToolkit.cancel_active_and_wait

    def checked_wait(toolkit):
        assert not toolkit.is_executing_on_current_thread(), "Would deadlock waiting on the caller's own tool"
        return original_wait(toolkit)

    monkeypatch.setattr(NavigationToolkit, "cancel_active_and_wait", checked_wait)
    previous = signal.getsignal(signal.SIGTERM)
    observed["on_run"] = lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    with pytest.raises(SystemExit) as error:
        go2.main([*args, "--execute"])
    assert error.value.code == 128 + signal.SIGTERM
    assert observed["policy_calls"] == 0 and observed["transport_closed"]
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.parametrize(
    "options",
    [["--planner-model", "second-model"], ["--profile", "legacy"], ["--planner-backend", "codex"]],
)
def test_invalid_backend_options_fail_before_service_access(entrypoint, monkeypatch, options):
    args, _, observed, _, _ = entrypoint
    monkeypatch.setattr(go2, "connect_client", lambda *a, **k: pytest.fail("Service access before argument validation"))
    with pytest.raises(SystemExit) as error:
        go2.main([*args, *options])
    assert error.value.code == 2 and observed["hardware_clients"] == 0
