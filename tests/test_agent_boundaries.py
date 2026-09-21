from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.astra_pi05.decision import validate_decision, validate_proposal  # noqa: E402


def _proposal() -> tuple[list[list[float]], dict[str, object]]:
    return (
        [[float(step), 0.0, 0.0, 0.0, 0.0, 0.0] * 2 for step in range(50)],
        {
            "action_semantics": "biso101_so101_v1",
            "action_layout": "left6_right6",
            "action_encoding": "absolute",
            "joint_position_unit": "degrees",
            "action_dim": 12,
            "horizon": 50,
        },
    )


def _base_decision() -> dict[str, object]:
    return {
        "proposal_id": "proposal-1",
        "observation_id": "observation-1",
        "decision": "execute_prefix",
        "execute_steps": 1,
        "corrections": [],
        "reason": "bounded review",
    }


def _correction() -> dict[str, object]:
    waypoint = {
        "reach_m": 0.1,
        "height_m": 0.2,
        "pan_deg": 0.0,
        "wrist_flex_deg": 0.0,
        "wrist_roll_deg": 0.0,
        "gripper": 0.5,
    }
    return {
        "left": waypoint,
        "right": None,
        "frame": "so101_shoulder_plane",
        "units": "m_deg",
        "duration_s": 0.2,
    }


@pytest.mark.parametrize("unit", ["degrees", "range_m100_100"])
def test_real_proposal_shape_is_exactly_50_by_12(unit: str) -> None:
    actions, metadata = _proposal()
    metadata["joint_position_unit"] = unit
    result = validate_proposal(actions, metadata)
    assert len(result) == 50
    assert all(len(row) == 12 for row in result)


def test_proposal_rejects_implicit_or_unknown_units() -> None:
    actions, metadata = _proposal()
    metadata["joint_position_unit"] = "radians"
    with pytest.raises(ValueError, match="joint_position_unit"):
        validate_proposal(actions, metadata)


@pytest.mark.parametrize("steps", [1, 15])
def test_execute_prefix_accepts_only_bounded_prefix_lengths(steps: int) -> None:
    decision = _base_decision()
    decision["execute_steps"] = steps
    result = validate_decision(decision, proposal_id="proposal-1", observation_id="observation-1")
    assert result["decision"] == "execute_prefix"


@pytest.mark.parametrize("steps", [0, 16])
def test_execute_prefix_rejects_out_of_range_prefix_lengths(steps: int) -> None:
    decision = _base_decision()
    decision["execute_steps"] = steps
    with pytest.raises(ValueError, match="execute_prefix"):
        validate_decision(decision, proposal_id="proposal-1", observation_id="observation-1")


@pytest.mark.parametrize("count", [1, 5])
def test_correction_branch_accepts_one_to_five_declared_waypoints(count: int) -> None:
    decision = _base_decision()
    decision.update(
        decision="correct",
        execute_steps=0,
        corrections=[_correction()] * count,
    )
    result = validate_decision(decision, proposal_id="proposal-1", observation_id="observation-1")
    assert result["decision"] == "correct"
    assert len(result["corrections"]) == count


@pytest.mark.parametrize("count", [0, 6])
def test_correction_branch_rejects_out_of_range_waypoint_count(count: int) -> None:
    decision = _base_decision()
    decision.update(
        decision="correct",
        execute_steps=0,
        corrections=[_correction()] * count,
    )
    with pytest.raises(ValueError, match="1-5"):
        validate_decision(decision, proposal_id="proposal-1", observation_id="observation-1")


def test_correction_contract_is_not_generic_six_d() -> None:
    correction = _correction()
    correction["frame"] = "world"
    decision = _base_decision()
    decision.update(decision="correct", execute_steps=0, corrections=[correction])
    with pytest.raises(ValueError, match="so101_shoulder_plane"):
        validate_decision(decision, proposal_id="proposal-1", observation_id="observation-1")


def test_decision_result_does_not_share_nested_corrections() -> None:
    decision = _base_decision()
    decision.update(decision="correct", execute_steps=0, corrections=[_correction()])
    original = copy.deepcopy(decision)
    result = validate_decision(decision, proposal_id="proposal-1", observation_id="observation-1")
    result["corrections"][0]["left"]["reach_m"] = 99.0
    assert decision == original


def test_algorithm_module_has_no_driver_or_runtime_imports() -> None:
    source = Path(__file__).parents[1] / "agents" / "astra_pi05" / "decision.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {"embodirun", "embodied_runtime", "feetech", "franky", "unitree"}
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert not forbidden.intersection(imported)
