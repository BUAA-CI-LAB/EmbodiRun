"""Real recipe/client/Control/mapper/adapter over HTTP, with one fake owner.

Only the external owner, model inference and Astra decision are fixtures.
No camera, robot SDK, model service or authenticated Astra process is opened.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.xlerobot_snack_delivery.run import ControlRuntimes, RecipeError, SnackDelivery
from examples.xlerobot_snack_delivery.services import prepare

from embodirun.application.contracts import ControlServiceConfig
from embodirun.model_services import PolicyAction, PolicyResult
from embodirun.robots.lerobot.xlerobot.units import ARM_UNITS, stamped_metadata
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.server import ControlHttpServer, ControlService
from test_xlerobot_external_owner import _close_control, _OwnerHandler, _OwnerState, _port


class ScopeState(_OwnerState):
    def observation(self, owner):
        reply = super().observation(owner)
        reply["observation"].update(
            {
                "state_age_ns": 1_000_000,
                "camera_ages_ns": {"front": 1_000_000},
                "camera_timestamps_ns": {"front": time.time_ns() - 1_000_000},
                "safety": {
                    "base_control_ready": True,
                    "stopped": True,
                    "stop_confirmed": True,
                    "arms_stop_confirmed": self.owner is None and getattr(self, "confirm_arms", True),
                },
                "navigation": {"arrived": True, "zero_velocity": True},
                "task_evidence": {"grasp_confirmed": bool(self.commands)},
            }
        )
        reply["images"] = {"front": base64.b64encode(b"fake-jpeg").decode()}
        return reply


class Model:
    def open_session(self, **kwargs):
        return SimpleNamespace(session_id="fixture-model-session")

    def step(self, observation):
        assert observation.images
        return PolicyResult(
            request_id=observation.request_id,
            session_id=observation.session_id,
            step_id=observation.step_id,
            session_revision=1,
            action_space="pi05.action_chunk.v1",
            actions=(PolicyAction("action_chunk", {"data": [[0.0] * 12] * 10, "feature_names": list(ARM_UNITS)}),),
        )

    def close(self, session_id):
        pass


@pytest.mark.parametrize("confirm_arms", [True, False])
def test_recipe_through_real_http_control_and_shared_external_owner(tmp_path, monkeypatch, confirm_arms):
    states = {scope: ScopeState(scope=scope) for scope in ("arms", "base")}
    states["arms"].confirm_arms = confirm_arms

    class Handler(_OwnerHandler):
        @property
        def state(self):
            return states[self.headers["X-Teleop-Scope"]]

    owner = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    owner_thread = threading.Thread(target=owner.serve_forever, daemon=True)
    owner_thread.start()
    running = []
    try:
        (tmp_path / "hardware.json").write_text("{}")
        deployment = {
            "owner": {"port": owner.server_port, "hardware_config": "hardware.json"},
            "control": {"base_port": _port(), "manipulation_port": _port()},
            "model": {"endpoint": "http://127.0.0.1:1"},
            "cameras": {"observation.images.front": "front"},
        }
        prepare(deployment, tmp_path / "private", base_dir=tmp_path)
        configs = []
        for scope in ("base", "arms"):
            config = ControlServiceConfig.from_json((tmp_path / f"private/{scope}-control.json").read_text())
            options = {**config.robot_options, "token": "t"}
            inputs = tuple(replace(item, options={**item.options, "token": "t"}) for item in config.inputs)
            config = replace(config, robot_options=options, inputs=inputs)
            configs.append(config)
            manager = DeviceManager(
                "local",
                owner_id=f"test-{scope}",
                lock_dir=tmp_path / f"locks-{scope}",
                state_path=tmp_path / f"device-state-{scope}.json",
            )
            service = ControlService(config, device_manager=manager)
            monkeypatch.setattr(service, "_inference_client", lambda *_: Model())
            monkeypatch.setattr(service, "_release_inference_client", lambda *_: None)
            server = ControlHttpServer(service, state_dir=tmp_path / f"jobs-{scope}")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            running.append((service, server, thread))
        config = json.loads(Path("examples/xlerobot_snack_delivery/config.example.json").read_text())
        for role, service in zip(("base", "manipulation"), configs):
            config["control"]["endpoints"][role]["endpoint"] = f"http://127.0.0.1:{service.port}"
        action = {
            "timestamp_s": 0,
            "values": {"x.vel": 0.03, "theta.vel": 0},
            "metadata": stamped_metadata({"x.vel": 0.03, "theta.vel": 0}, scope="base"),
        }
        config["navigation"]["routes"] = {
            name: {"fixture": False, "source": "recorded_route", "chunks": [{"actions": [action]}]}
            for name in ("outbound", "return")
        }
        config["evidence"]["manual_confirmation"] = []
        runtimes = ControlRuntimes(config)
        reviewed = []

        class Agent:
            def before_grasp(self, packet):
                reviewed.append(packet["observation_id"])
                assert packet["images"]
                time.sleep(0.08)  # observer continues publishing while RPent decides
                return {
                    "observation_id": packet["observation_id"],
                    "decision": "proceed",
                    "instruction": "grasp chips",
                    "max_steps": 3,
                }

        runner = SnackDelivery(config, runtimes, tmp_path / "run", auto_confirm=False)
        runner._agent = Agent()
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        if not confirm_arms:
            with pytest.raises(RecipeError, match="no confirmed stop feedback"):
                runner.run()
            assert len(states["base"].commands) == 1  # no return route
            assert len(states["arms"].commands) == 3  # no handover/release
            assert runner.state["status"] == "failed"
            return
        result = runner.run()
        assert result["status"] == "completed"
        assert result["physical_success"] is None and result["task_success"] == "unverified"
        assert len(states["base"].commands) == 2
        assert len(states["arms"].commands) == 5  # bounded 3-step VLA + extend + release
        assert all(set(cmd) == {"x.vel", "theta.vel"} for cmd in states["base"].commands)
        proposal = json.loads((tmp_path / "run/grasp-proposal-000.json").read_text())
        assert proposal["observation_id"] != reviewed[0]
        assert states["base"].owner is None and states["arms"].owner is None
    finally:
        for service, server, thread in reversed(running):
            _close_control(service, server, thread)
        owner.shutdown()
        owner.server_close()
        owner_thread.join(2)
