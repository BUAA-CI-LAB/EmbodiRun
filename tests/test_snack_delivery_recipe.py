"""Focused contract tests for the guarded XLeRobot snack delivery recipe."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from examples.xlerobot_snack_delivery import run as recipe
from examples.xlerobot_snack_delivery.evidence import ConfigEvidenceNormalizer

from embodirun.robots.lerobot.xlerobot.units import XLEROBOT_ACTION_SPACE


def test_observation_age_is_translated_to_the_recipe_clock(monkeypatch):
    from examples.xlerobot_snack_delivery import run

    runtime = SimpleNamespace(
        simulation=False,
        runtime_id="remote",
        observe=lambda: {
            "observation_id": "remote:1",
            "payload": {
                "fresh": True,
                "age_ns": 25,
                "timestamps": {"source_timestamps_ns": {"state": 999_999_999}, "skew_ns": 5},
            },
        },
    )
    delivery = object.__new__(run.SnackDelivery)
    monkeypatch.setattr(run.time, "monotonic_ns", lambda: 100)
    assert delivery.observe(runtime)["capture_lower_bound_ns"] == 70


BASE_VALUES = {"x.vel": 0.03, "theta.vel": 0.0}
ARM_VALUES = {"right_arm_gripper.pos": 25.0}
BASE_UNITS = {"x.vel": "metres-per-sec", "theta.vel": "angular-degrees-per-sec"}
ARM_UNITS = {"right_arm_gripper.pos": "range_0_100"}

BASE_PAYLOAD = {
    "fresh": True,
    "safety": {"base_control_ready": True, "stopped": True, "stop_confirmed": True},
    "navigation": {"arrived": True, "zero_velocity": True},
}
ARM_PAYLOAD = {
    "fresh": True,
    "safety": {"stopped": True, "stop_confirmed": True},
    "task_evidence": {"grasp_confirmed": True},
}


def _action_payload(values: Mapping[str, Any], units: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "action_space": XLEROBOT_ACTION_SPACE,
        "units": dict(units if units is not None else {}),
    }
    return {"timestamp_s": 0.0, "values": dict(values), "metadata": metadata}


def _write_route(path: Path, values: Mapping[str, Any], *, units: Mapping[str, Any] | None) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": path.stem,
                "fixture": True,
                "chunks": [{"action": _action_payload(values, units)}],
            }
        )
    )
    return path


def _config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    routes = {
        "outbound": str(_write_route(tmp_path / "outbound.json", BASE_VALUES, units=BASE_UNITS)),
        "return": str(_write_route(tmp_path / "return.json", BASE_VALUES, units=BASE_UNITS)),
    }
    config: dict[str, Any] = {
        "control": {
            "caller_id": "recipe-test",
            "session_id": "session",
            "timeout_s": 5,
            "endpoints": {
                "base": {
                    "endpoint": "http://127.0.0.1:1",
                    "runtime_id": "xlerobot-base",
                    "scope": "base",
                },
                "manipulation": {
                    "endpoint": "http://127.0.0.1:2",
                    "runtime_id": "xlerobot-pi05",
                    "scope": "arms",
                },
            },
        },
        "planning": {"mode": "recorded_route"},
        "navigation": {
            "control_hz": 15,
            "require_zero_velocity": True,
            "arrival_evidence": ["route_arrived", "stopped", "stop_confirmed"],
            "routes": routes,
        },
        "grasp": {"instruction": "grasp the chips", "timeout_s": 5, "control_hz": 15},
        "handover": {
            "control_hz": 15,
            "forward_pose": {"right_arm_shoulder_pan.pos": -20.0},
            "gripper_opening": {"right_arm_gripper.pos": 21.6},
        },
        "evidence": {
            "paths": {
                "base_control_ready": ["safety", "base_control_ready"],
                "stopped": ["safety", "stopped"],
                "stop_confirmed": ["safety", "stop_confirmed"],
                "grasp_confirmed": ["task_evidence", "grasp_confirmed"],
                "route_arrived": ["navigation", "arrived"],
                "zero_velocity": ["navigation", "zero_velocity"],
            }
        },
        "_config_dir": str(tmp_path),
    }
    config.update(overrides)
    return config


class FakeRuntime:
    simulation = True

    def __init__(
        self,
        *,
        runtime_id: str,
        scope: str,
        payload: Mapping[str, Any],
        receipt: Mapping[str, Any] | None = None,
        stop_receipt: Mapping[str, Any] | None = None,
        proposal: Mapping[str, Any] | None = None,
        interrupt_on_execute: bool = False,
    ) -> None:
        self.runtime_id = runtime_id
        self.scope = scope
        self.payload = dict(payload)
        self.receipt = dict(
            receipt
            or {
                "status": "completed",
                "dispatch_status": "completed",
                "accepted": True,
                "physical_status": "simulation_fixture_only",
            }
        )
        self.stop_receipt = dict(stop_receipt or {"status": "stopped", "stop_confirmed": False})
        self.proposal = dict(
            proposal
            or {
                "status": "proposed",
                "proposal_id": "p1",
                "actions": [_action_payload(ARM_VALUES, ARM_UNITS)],
            }
        )
        self.interrupt_on_execute = interrupt_on_execute
        self.observed = 0
        self.executed: list[Any] = []
        self.stopped: list[str] = []
        self.proposals = 0

    def observe(self) -> dict[str, Any]:
        self.observed += 1
        return {
            "observation_id": f"{self.runtime_id}:{self.observed}",
            "payload": dict(self.payload),
        }

    def propose(self, **kwargs: Any) -> dict[str, Any]:
        self.proposals += 1
        return dict(self.proposal)

    def execute(self, action: Any, **kwargs: Any) -> dict[str, Any]:
        self.executed.append((action, kwargs))
        if self.interrupt_on_execute:
            raise KeyboardInterrupt()
        return dict(self.receipt)

    def stop(self, request_id: str) -> dict[str, Any]:
        self.stopped.append(request_id)
        return dict(self.stop_receipt)


class FakeRuntimes:
    def __init__(self, base: FakeRuntime, manipulation: FakeRuntime, *, simulation: bool = True) -> None:
        self.base = base
        self.manipulation = manipulation
        self.simulation = simulation


def _runtimes(
    *,
    base_payload: Mapping[str, Any] | None = None,
    arm_payload: Mapping[str, Any] | None = None,
    simulation: bool = True,
    **runtime_kwargs: Any,
) -> FakeRuntimes:
    base = FakeRuntime(
        runtime_id="xlerobot-base",
        scope="base",
        payload=base_payload or BASE_PAYLOAD,
        **runtime_kwargs,
    )
    manipulation = FakeRuntime(
        runtime_id="xlerobot-pi05",
        scope="arms",
        payload=arm_payload or ARM_PAYLOAD,
        **runtime_kwargs,
    )
    return FakeRuntimes(base, manipulation, simulation=simulation)


def _runner(
    tmp_path: Path,
    runtimes: FakeRuntimes,
    *,
    config: Mapping[str, Any] | None = None,
    auto_confirm: bool = True,
) -> recipe.SnackDelivery:
    resolved = dict(config or _config(tmp_path))
    recipe._validate_config(resolved)
    return recipe.SnackDelivery(
        resolved,
        runtimes,
        tmp_path / "out",
        auto_confirm=auto_confirm,
    )


# --- action contract --------------------------------------------------------


def test_recipe_authors_action_space_and_canonical_units() -> None:
    action = recipe._action_from_values(ARM_VALUES, "handover-extension", scope="arms")
    assert action["metadata"]["action_space"] == XLEROBOT_ACTION_SPACE
    assert action["metadata"]["units"] == ARM_UNITS

    base = recipe._action_from_values(BASE_VALUES, "route", scope="base")
    assert base["metadata"]["units"] == BASE_UNITS


def test_external_action_must_declare_action_space_and_units() -> None:
    missing_units = {
        "timestamp_s": 0.0,
        "values": dict(BASE_VALUES),
        "metadata": {"action_space": XLEROBOT_ACTION_SPACE},
    }
    with pytest.raises(recipe.RecipeError, match="units"):
        recipe._action(missing_units, "route[0]", scope="base")

    unknown_unit = {
        "timestamp_s": 0.0,
        "values": dict(BASE_VALUES),
        "metadata": {
            "action_space": XLEROBOT_ACTION_SPACE,
            "units": {"x.vel": "miles-per-hour", "theta.vel": "angular-degrees-per-sec"},
        },
    }
    with pytest.raises(recipe.RecipeError, match="unsupported unit"):
        recipe._action(unknown_unit, "route[0]", scope="base")

    missing_space = {
        "timestamp_s": 0.0,
        "values": dict(BASE_VALUES),
        "metadata": {"units": BASE_UNITS},
    }
    with pytest.raises(recipe.RecipeError, match="action_space"):
        recipe._action(missing_space, "route[0]", scope="base")


def test_base_action_is_never_sent_to_the_arms_scope() -> None:
    config = _config(Path("/tmp"))
    runtime = recipe.ControlEndpointRuntime(config, role="manipulation")
    assert runtime.scope == "arms"
    with pytest.raises(recipe.RecipeError, match="refusing to send a base action"):
        runtime.require_scope({"values": dict(BASE_VALUES)})

    base_runtime = recipe.ControlEndpointRuntime(config, role="base")
    with pytest.raises(recipe.RecipeError, match="refusing to send a arms action"):
        base_runtime.require_scope({"values": dict(ARM_VALUES)})


def test_mixed_scope_action_is_refused() -> None:
    mixed = {"x.vel": 0.1, "right_arm_gripper.pos": 10.0}
    with pytest.raises(recipe.RecipeError, match="unknown XLeRobot action field|mixes"):
        recipe._action(
            {"timestamp_s": 0.0, "values": mixed, "metadata": {"action_space": XLEROBOT_ACTION_SPACE}},
            "mixed",
            scope="arms",
        )


def test_endpoint_role_and_scope_must_match() -> None:
    with pytest.raises(recipe.RecipeError, match="scope must be 'arms'"):
        recipe._endpoint_spec(
            {
                "control": {
                    "endpoints": {
                        "manipulation": {
                            "endpoint": "http://127.0.0.1:2",
                            "runtime_id": "rt",
                            "scope": "base",
                        }
                    }
                }
            },
            "manipulation",
        )


# --- evidence boundary ------------------------------------------------------


def test_missing_grasp_evidence_fails_closed(tmp_path: Path) -> None:
    arm_payload = {"fresh": True, "safety": {"stopped": True, "stop_confirmed": True}}
    runner = _runner(tmp_path, _runtimes(arm_payload=arm_payload))
    with pytest.raises(recipe.RecipeError, match="grasp is not confirmed"):
        runner.require_evidence(arm_payload, "grasp_confirmed", label="grasp")


def test_full_run_fails_closed_without_grasp_evidence(tmp_path: Path) -> None:
    arm_payload = {"fresh": True, "safety": {"stopped": True, "stop_confirmed": True}}
    runner = _runner(tmp_path, _runtimes(arm_payload=arm_payload))
    with pytest.raises(recipe.RecipeError, match="grasp is not confirmed"):
        runner.run()
    assert runner.state["status"] == "failed"
    assert runner.state["task_success"] == "unverified"
    assert runner.state["physical_success"] is None


def test_manual_confirmation_is_labelled_not_fabricated(tmp_path: Path) -> None:
    arm_payload = {"fresh": True, "safety": {"stopped": True, "stop_confirmed": True}}
    config = _config(tmp_path)
    config["evidence"]["manual_confirmation"] = ["grasp_confirmed"]
    runner = _runner(tmp_path, _runtimes(arm_payload=arm_payload), config=config)
    state = runner.run()
    assert state["status"] == "completed"
    events = [event for event in state["history"] if event["event"] == "grasp_confirmed"]
    assert events and events[-1]["evidence"]["source"] == "manual_confirmation"
    manual = [event for event in state["history"] if event["event"] == "evidence_manual_confirmation"]
    assert manual and manual[-1]["evidence"]["confirmed"] is True


def test_manual_confirmation_is_opt_in(tmp_path: Path) -> None:
    normalizer = ConfigEvidenceNormalizer.from_config(_config(tmp_path))
    assert normalizer.manual_confirmation_allowed("grasp_confirmed") is False

    config = _config(tmp_path)
    config["evidence"]["manual_confirmation"] = ["grasp_confirmed"]
    opt_in = ConfigEvidenceNormalizer.from_config(config)
    assert opt_in.manual_confirmation_allowed("grasp_confirmed") is True


def test_evidence_only_confirms_explicit_true(tmp_path: Path) -> None:
    normalizer = ConfigEvidenceNormalizer.from_config(_config(tmp_path))
    truthy_string = {"safety": {"stopped": "true"}}
    assert normalizer.evaluate("stopped", truthy_string).confirmed is False
    assert normalizer.evaluate("stopped", {"safety": {"stopped": True}}).confirmed is True
    assert normalizer.evaluate("stopped", {}).source == "missing"


# --- terminal receipts ------------------------------------------------------


@pytest.mark.parametrize(
    "receipt,message",
    [
        ({"status": "accepted", "dispatch_status": "accepted"}, "not a completed dispatch"),
        ({"status": "running", "dispatch_status": "running"}, "not a completed dispatch"),
        ({"status": "unknown", "dispatch_status": "unknown"}, "not a completed dispatch"),
        ({"status": "cancelled", "dispatch_status": "cancelled"}, "not a completed dispatch"),
        (
            {"status": "completed", "dispatch_status": "running"},
            "not a completed dispatch",
        ),
        (
            {
                "status": "completed",
                "dispatch_status": "completed",
                "physical_status": "unknown",
            },
            "physical_status",
        ),
        (
            {
                "status": "completed",
                "dispatch_status": "completed",
                "physical_status": "stop_unconfirmed",
            },
            "physical_status",
        ),
    ],
)
def test_nonterminal_receipts_never_advance(tmp_path: Path, receipt: Mapping[str, Any], message: str) -> None:
    runtimes = _runtimes(receipt=receipt)
    runner = _runner(tmp_path, runtimes)
    with pytest.raises(recipe.RecipeError, match=message):
        runner.execute(
            runtimes.base,
            recipe._action_from_values(BASE_VALUES, "route", scope="base"),
            "obs-1",
            stage="route",
            hz=15,
        )
    assert runner.state["history"] == []


def test_terminal_receipt_is_recorded_with_scope(tmp_path: Path) -> None:
    runtimes = _runtimes()
    runner = _runner(tmp_path, runtimes)
    result = runner.execute(
        runtimes.base,
        recipe._action_from_values(BASE_VALUES, "route", scope="base"),
        "obs-1",
        stage="route",
        hz=15,
    )
    assert result["status"] == "completed"
    receipt = runner.state["history"][-1]
    assert receipt["event"] == "execution_receipt"
    assert receipt["scope"] == "base"
    assert receipt["runtime_id"] == "xlerobot-base"


# --- interruption -----------------------------------------------------------


def test_keyboard_interrupt_stops_the_same_request_and_keeps_unconfirmed(
    tmp_path: Path,
) -> None:
    runtimes = _runtimes(interrupt_on_execute=True)
    runner = _runner(tmp_path, runtimes)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    assert runner.state["status"] == "interrupted"
    assert runner.state["stop_requested"] is True
    assert runner.state["stop_unconfirmed"] is True
    assert runner.state["physical_success"] is None
    request_id = runner.state["history"][-1]["request_id"]
    assert request_id is not None
    # The base endpoint owned the in-flight request and must receive the stop.
    assert runtimes.base.stopped == [request_id]
    assert runtimes.manipulation.stopped == []
    stop_event = [event for event in runner.state["history"] if event["event"] == "stop_requested_after_failure"]
    assert stop_event and stop_event[-1]["runtime_id"] == "xlerobot-base"


def test_confirmed_stop_is_recorded_for_interrupts(tmp_path: Path) -> None:
    runtimes = _runtimes(
        interrupt_on_execute=True,
        stop_receipt={"status": "stopped", "stop_confirmed": True},
    )
    runner = _runner(tmp_path, runtimes)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    assert runner.state["stop_confirmed"] is True
    assert runner.state["stop_unconfirmed"] is False


# --- route arrival evidence -------------------------------------------------


def test_route_arrival_without_evidence_is_unverified_and_blocks_grasp(
    tmp_path: Path,
) -> None:
    base_payload = {"fresh": True, "safety": {"base_control_ready": True, "stopped": True, "stop_confirmed": True}}
    runtimes = _runtimes(base_payload=base_payload)
    runner = _runner(tmp_path, runtimes)
    with pytest.raises(recipe.RecipeError, match="arrival is unverified"):
        runner.run()
    assert runner.state["route_arrival"]["outbound"]["verified"] is False
    complete = [event for event in runner.state["history"] if event["event"] == "route_complete"]
    assert complete and complete[0]["arrival"] == "unverified"
    assert runtimes.manipulation.proposals == 0


def test_zero_velocity_evidence_is_required_by_default(tmp_path: Path) -> None:
    base_payload = {
        "fresh": True,
        "safety": {"base_control_ready": True, "stopped": True, "stop_confirmed": True},
        "navigation": {"arrived": True},
    }
    runner = _runner(tmp_path, _runtimes(base_payload=base_payload))
    with pytest.raises(recipe.RecipeError, match="arrival is unverified"):
        runner.run()


def test_verified_route_allows_grasp(tmp_path: Path) -> None:
    runtimes = _runtimes()
    runner = _runner(tmp_path, runtimes)
    state = runner.run()
    assert state["status"] == "completed"
    assert state["route_arrival"]["outbound"]["verified"] is True
    assert state["route_arrival"]["return"]["verified"] is True
    assert runtimes.manipulation.proposals == 1


# --- success semantics ------------------------------------------------------


def test_completion_never_claims_physical_success_without_evidence(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _runtimes(simulation=False))
    runner.state["last_handover_payload"] = {"safety": {"stopped": True, "stop_confirmed": True}}
    completion = runner.completion_evidence()
    assert completion["task_success"] == "unverified"
    assert completion["physical_success"] is None


def test_completion_reports_success_only_from_declared_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["evidence"]["completion"] = {
        "task_success": ["task_evidence", "delivery_confirmed"],
        "physical_success": ["task_evidence", "handover_sensor_confirmed"],
    }
    runner = _runner(tmp_path, _runtimes(simulation=False), config=config)
    runner.state["last_handover_payload"] = {
        "task_evidence": {"delivery_confirmed": True, "handover_sensor_confirmed": True}
    }
    completion = runner.completion_evidence()
    assert completion["task_success"] is True
    assert completion["physical_success"] is True
    assert completion["evidence"]["physical_success"]["confirmed"] is True


def test_simulation_never_claims_physical_success(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["evidence"]["completion"] = {
        "task_success": ["task_evidence", "delivery_confirmed"],
        "physical_success": ["task_evidence", "handover_sensor_confirmed"],
    }
    runner = _runner(tmp_path, _runtimes(simulation=True), config=config)
    runner.state["last_handover_payload"] = {
        "task_evidence": {"delivery_confirmed": True, "handover_sensor_confirmed": True}
    }
    completion = runner.completion_evidence()
    assert completion["task_success"] == "unverified"
    assert completion["physical_success"] is None


# --- configuration ----------------------------------------------------------


def test_recipe_config_rejects_credentials(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["control"]["token"] = "nope"
    with pytest.raises(recipe.RecipeError, match="credential"):
        recipe._validate_config(config)


def test_recipe_config_rejects_legacy_owner_endpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["control"]["endpoint"] = "http://127.0.0.1:8766"
    with pytest.raises(recipe.RecipeError, match="control.endpoints"):
        recipe._validate_config(config)


def test_recipe_config_rejects_endpoint_with_credentials(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["control"]["endpoints"]["base"]["endpoint"] = "http://user:pass@127.0.0.1:8100"
    with pytest.raises(recipe.RecipeError, match="without credentials"):
        recipe._validate_config(config)


def test_plan_records_scopes_and_unverified_success(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _runtimes())
    summary = runner.plan()
    assert summary["control_endpoints"]["base"]["scope"] == "base"
    assert summary["control_endpoints"]["manipulation"]["scope"] == "arms"
    assert summary["task_success"] == "unverified"
    assert summary["physical_success"] is None
    assert "zero_velocity" in summary["arrival_evidence"]
