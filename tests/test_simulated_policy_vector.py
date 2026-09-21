"""Focused contract tests for the unit-preserving simulated Pi0.5 path."""

from __future__ import annotations

import math

import pytest

from embodirun.bindings import binding_definition
from embodirun.bindings.simulated.policy_vector.pi05 import (
    POLICY_ACTION_SPACE,
    SO101_POLICY_FEATURE_NAMES,
    SimulatedPolicyVectorPi05Mapper,
    SimulatedPolicyVectorPi05MapperError,
)
from embodirun.robots import RobotAction, RobotObservation, robot_definition
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.robots.simulated.policy_vector import (
    POLICY_VECTOR_ACTION_SPACE,
    UNVERIFIED_UNITS,
    PolicyVectorAdapterError,
)
from embodirun.services.inference import PolicyAction, PolicyResult


def _frames() -> tuple[CameraFrame, CameraFrame]:
    return (
        CameraFrame("observation.images.front", "image/jpeg", b"front"),
        CameraFrame("observation.images.wrist", "image/jpeg", b"wrist"),
    )


def _result(
    *,
    rows: object | None = None,
    feature_names: object = SO101_POLICY_FEATURE_NAMES,
) -> PolicyResult:
    values = [[float(index + offset) for offset in range(6)] for index in range(50)] if rows is None else rows
    return PolicyResult(
        request_id="request-1",
        session_id="session-1",
        step_id=0,
        session_revision=1,
        action_space=POLICY_ACTION_SPACE,
        actions=(
            PolicyAction(
                "action_chunk",
                {"data": values, "feature_names": feature_names},
            ),
        ),
    )


def test_robot_and_binding_are_discoverable_without_hardware_imports() -> None:
    robot = robot_definition("simulated.policy_vector")
    binding = binding_definition("simulated.policy_vector.pi05")

    assert robot.kind == "simulated.policy_vector"
    assert binding.robot_kind == robot.kind
    assert binding.model_kind == "pi05"
    assert binding.maximum_chunk_steps == 50
    assert binding.adapter_config["state_fields"] == ("state_native",)


def test_adapter_preserves_native_values_and_reports_simulated_receipts() -> None:
    definition = robot_definition("simulated.policy_vector")
    config = definition.config_factory(
        "native-demo",
        {"initial_state_native": [-104.5, 0.25, 93.0, 70.5, -3.0, 16.0]},
    )
    adapter = definition.adapter_type(config)

    adapter.connect(prepare=False)
    passive = adapter.observe()
    assert passive.values == {"state_native": [-104.5, 0.25, 93.0, 70.5, -3.0, 16.0]}
    assert passive.metadata["units"] == UNVERIFIED_UNITS
    assert passive.metadata["simulated"] is True
    assert passive.metadata["hardware_access"] is False
    with pytest.raises(PolicyVectorAdapterError, match="not prepared"):
        adapter.execute(
            RobotAction(
                timestamp_s=0.0,
                values={"type": "state_native_vector", "state_native": [1] * 6},
                metadata={"action_space": POLICY_VECTOR_ACTION_SPACE},
            )
        )

    adapter.prepare()
    target = [-150.0, 0.5, 101.25, 70.5, -3.0, 16.0]
    receipt = adapter.execute(
        RobotAction(
            timestamp_s=0.0,
            values={"type": "state_native_vector", "state_native": target},
            metadata={
                "action_space": POLICY_VECTOR_ACTION_SPACE,
                "units": UNVERIFIED_UNITS,
            },
        )
    )
    assert receipt["requested"] == {"state_native": target, "units": UNVERIFIED_UNITS}
    assert receipt["applied"] == receipt["requested"]
    assert receipt["measured"] == receipt["requested"]
    assert receipt["simulated"] is True
    assert receipt["hardware_access"] is False
    assert adapter.observe().values["state_native"] == target

    stop = adapter.stop()
    assert stop["requested"] == "stop"
    assert stop["stop_confirmed"] is True
    assert stop["hardware_access"] is False
    adapter.close()


@pytest.mark.parametrize(
    "values, message",
    [
        ([0.0] * 5, "contain 6"),
        ([0.0] * 6 + [1.0], "contain 6"),
        ([0.0, math.nan, 0.0, 0.0, 0.0, 0.0], "must be finite"),
    ],
)
def test_adapter_rejects_nonfinite_or_wrong_length_native_vectors(values: list[float], message: str) -> None:
    definition = robot_definition("simulated.policy_vector")
    adapter = definition.adapter_type(definition.config_factory("demo", {}))
    adapter.connect()
    with pytest.raises(PolicyVectorAdapterError, match=message):
        adapter.execute(
            RobotAction(
                timestamp_s=0.0,
                values={"type": "state_native_vector", "state_native": values},
                metadata={"action_space": POLICY_VECTOR_ACTION_SPACE},
            )
        )


def test_mapper_preserves_state_and_requires_front_and_wrist_frames() -> None:
    mapper = SimulatedPolicyVectorPi05Mapper()
    observation = RobotObservation(
        timestamp_s=1.5,
        values={"state_native": [-104.5, 0.25, 93.0, 70.5, -3.0, 16.0]},
        metadata={
            "units": UNVERIFIED_UNITS,
            "simulated": True,
            "hardware_access": False,
        },
    )

    request = mapper.map_observation(
        observation,
        session_id="session-1",
        request_id="request-1",
        step_id=0,
        instruction="pick up the cube",
        frames=_frames(),
    )

    assert request.state == {"state_native": [-104.5, 0.25, 93.0, 70.5, -3.0, 16.0]}
    assert tuple(image.name for image in request.images) == (
        "observation.images.front",
        "observation.images.wrist",
    )
    assert tuple(image.data for image in request.images) == (b"front", b"wrist")
    assert request.metadata["units"] == UNVERIFIED_UNITS

    with pytest.raises(SimulatedPolicyVectorPi05MapperError, match="missing camera"):
        mapper.map_observation(
            observation,
            session_id="session-1",
            request_id="request-2",
            step_id=0,
            instruction="pick up the cube",
            frames=(_frames()[0],),
        )


def test_mapper_maps_exact_fifty_named_rows_to_native_actions_without_conversion() -> None:
    mapper = SimulatedPolicyVectorPi05Mapper()
    actions = mapper.map_result(_result())

    assert len(actions) == 50
    assert actions[0].values == {
        "type": "state_native_vector",
        "state_native": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
    assert actions[-1].values["state_native"] == [49.0, 50.0, 51.0, 52.0, 53.0, 54.0]
    assert actions[0].metadata["action_space"] == POLICY_VECTOR_ACTION_SPACE
    assert actions[0].metadata["units"] == UNVERIFIED_UNITS
    assert actions[0].metadata["chunk_size"] == 50


@pytest.mark.parametrize(
    "result, message",
    [
        (_result(rows=[[0.0] * 6]), "exactly 50 action rows"),
        (_result(feature_names=("libero.x",) * 6), "exactly match"),
        (_result(feature_names=None), "six SO-101 names"),
        (_result(rows=[[0.0, 1.0, 2.0, 3.0, 4.0, float("nan")]] * 50), "finite"),
    ],
)
def test_mapper_rejects_non_pi05_or_malformed_chunks(result: PolicyResult, message: str) -> None:
    with pytest.raises(SimulatedPolicyVectorPi05MapperError, match=message):
        SimulatedPolicyVectorPi05Mapper().map_result(result)
