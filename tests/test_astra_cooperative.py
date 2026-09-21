from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agents.astra_pi05.cooperative import CooperativeLoop, Proposal

from embodirun.client import Observation, UnsupportedOperation
from test_agent_client import _client_server, _close

NAMES = tuple(f"joint_{index}" for index in range(12))


def _proposal(observation) -> dict[str, Any]:
    return {
        "proposal_id": "proposal-1",
        "actions": [[float(row)] * 12 for row in range(50)],
        "metadata": {
            "action_semantics": "biso101_so101_v1",
            "action_layout": "left6_right6",
            "action_encoding": "absolute",
            "joint_position_unit": "degrees",
            "action_dim": 12,
            "horizon": 50,
            "feature_names": list(NAMES),
        },
    }


def test_real_http_cooperative_round_discards_unexecuted_tail_and_reobserves() -> None:
    client, server, port = _client_server()
    try:

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 3,
                "corrections": [],
                "reason": "software fake reviewer selected a short prefix",
            }

        result = CooperativeLoop(
            client,
            proposal_provider=_proposal,
            reviewer=review,
        ).run_round(prompt="move")
        assert result.status == "completed"
        assert result.executed_steps == 3
        assert result.discarded_steps == 47
        assert result.post_observation is not None
        assert result.post_observation.observation_id == "obs-3"
        assert len(port.actions) == 3
        assert [action.values["joint_0"] for action in port.actions] == [0.0, 1.0, 2.0]
    finally:
        _close(server)


def test_post_observation_waits_past_intermediate_publication() -> None:
    client, server, _ = _client_server()
    try:
        observations = iter(
            [
                Observation("obs-before", {"observation_id": "obs-before", "sequence": 1}),
                Observation("obs-during", {"observation_id": "obs-during", "sequence": 2}),
                Observation("obs-after", {"observation_id": "obs-after", "sequence": 3}),
            ]
        )
        client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
        client.execute = lambda _actions, **kwargs: {  # type: ignore[method-assign]
            "status": "completed",
            "request_id": kwargs["request_id"],
            "result": {"executed_steps": 1},
        }

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 1,
                "corrections": [],
                "reason": "wait for publication after the in-flight snapshot",
            }

        result = CooperativeLoop(
            client,
            proposal_provider=_proposal,
            reviewer=review,
            post_observation_timeout_s=0.05,
        ).run_round()
        assert result.status == "completed"
        assert result.post_observation is not None
        assert result.post_observation.observation_id == "obs-after"
    finally:
        _close(server)


def test_http_202_pending_job_stays_pending_with_request_identity() -> None:
    client, server, _ = _client_server()
    try:
        dispatch = server.api.dispatch

        def pending_dispatch(method, path, *, body=None, headers=None):
            if method == "POST" and path.split("?", 1)[0] == "/v1/execute":
                request_id = body["request_id"]
                return SimpleNamespace(
                    status=202,
                    payload={
                        "status": "pending",
                        "request_id": request_id,
                        "error": "job timed out before completion",
                    },
                )
            return dispatch(method, path, body=body, headers=headers)

        server.api.dispatch = pending_dispatch  # type: ignore[method-assign]

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 1,
                "corrections": [],
                "reason": "pending HTTP job",
            }

        result = CooperativeLoop(
            client,
            proposal_provider=_proposal,
            reviewer=review,
        ).run_round()
        assert result.status == "pending"
        assert result.request_id is not None
        assert result.execution is not None
        assert result.execution["status"] == "pending"
        assert result.execution["request_id"] == result.request_id
    finally:
        _close(server)


def test_unsupported_execute_discards_the_entire_proposal() -> None:
    client, server, _ = _client_server()
    try:

        def reject_execute(_actions, **_kwargs):
            raise UnsupportedOperation(
                409,
                {"code": "unsupported", "error": "binding cannot execute"},
            )

        client.execute = reject_execute  # type: ignore[method-assign]

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 3,
                "corrections": [],
                "reason": "unsupported execution regression",
            }

        result = CooperativeLoop(
            client,
            proposal_provider=_proposal,
            reviewer=review,
        ).run_round()
        assert result.status == "unsupported"
        assert result.discarded_steps == 50
        assert result.request_id is not None
    finally:
        _close(server)


def test_hold_and_unsupported_correction_do_not_send_actions() -> None:
    client, server, port = _client_server()
    try:

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "hold",
                "execute_steps": 0,
                "corrections": [],
                "reason": "software reviewer requests hold",
            }

        held = CooperativeLoop(client, proposal_provider=_proposal, reviewer=review).run_round()
        assert held.status == "held"
        assert port.actions == []

        def unsupported(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "correct",
                "execute_steps": 0,
                "corrections": [
                    {
                        "left": {
                            "reach_m": 0.1,
                            "height_m": 0.2,
                            "pan_deg": 0.0,
                            "wrist_flex_deg": 0.0,
                            "wrist_roll_deg": 0.0,
                            "gripper": 0.0,
                        },
                        "right": None,
                        "frame": "so101_shoulder_plane",
                        "units": "m_deg",
                        "duration_s": 0.2,
                    }
                ],
                "reason": "needs an explicit robot mapper",
            }

        unsupported_result = CooperativeLoop(client, proposal_provider=_proposal, reviewer=unsupported).run_round()
        assert unsupported_result.status == "unsupported"
        assert port.actions == []
    finally:
        _close(server)


def test_proposal_observation_mismatch_and_pending_execution_are_explicit() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Proposal.from_payload(
            {
                "proposal_id": "p",
                "observation_id": "old",
                "actions": [[0.0] * 12 for _ in range(50)],
                "metadata": {
                    "action_semantics": "biso101_so101_v1",
                    "action_layout": "left6_right6",
                    "action_encoding": "absolute",
                    "joint_position_unit": "degrees",
                    "action_dim": 12,
                },
            },
            observation_id="new",
        )

    client, server, _ = _client_server()
    try:
        observations = iter(
            [
                Observation("obs-pending", {"observation_id": "obs-pending", "robot": {}}),
            ]
        )
        client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
        client.execute = lambda actions, **kwargs: {  # type: ignore[method-assign]
            "status": "accepted",
            "request_id": kwargs["request_id"],
        }

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 1,
                "corrections": [],
                "reason": "pending test",
            }

        result = CooperativeLoop(client, proposal_provider=_proposal, reviewer=review).run_round()
        assert result.status == "pending"
        assert result.post_observation is None
        assert result.request_id is not None
    finally:
        _close(server)


@pytest.mark.parametrize(
    ("execution_status", "expected_status"),
    [("unknown", "unknown"), ("cancelled", "cancelled"), ("stopped", "stopped")],
)
def test_terminal_execution_status_is_preserved_without_reobserve(execution_status: str, expected_status: str) -> None:
    client, server, _ = _client_server()
    try:
        client.observe = lambda **_kwargs: Observation(  # type: ignore[method-assign]
            "obs-terminal", {"observation_id": "obs-terminal", "robot": {}}
        )
        client.execute = lambda _actions, **kwargs: {  # type: ignore[method-assign]
            "status": execution_status,
            "request_id": kwargs["request_id"],
        }

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 1,
                "corrections": [],
                "reason": "terminal-status test",
            }

        result = CooperativeLoop(client, proposal_provider=_proposal, reviewer=review).run_round()
        assert result.status == expected_status
        assert result.post_observation is None
        assert result.executed_steps == 0
    finally:
        _close(server)


def test_completed_without_valid_step_evidence_does_not_claim_execution() -> None:
    client, server, _ = _client_server()
    try:
        observations = repeat(Observation("obs-evidence", {"observation_id": "obs-evidence", "robot": {}}))
        client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
        client.execute = lambda _actions, **kwargs: {  # type: ignore[method-assign]
            "status": "completed",
            "request_id": kwargs["request_id"],
            "result": {"executed_steps": -1},
        }

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "execute_prefix",
                "execute_steps": 2,
                "corrections": [],
                "reason": "invalid evidence test",
            }

        result = CooperativeLoop(
            client,
            proposal_provider=_proposal,
            reviewer=review,
            post_observation_timeout_s=0.01,
        ).run_round()
        assert result.status == "post_observation_pending"
        assert result.executed_steps == 0
        assert result.discarded_steps == 50
        assert result.post_observation is None
        assert result.execution_evidence_unknown is True
    finally:
        _close(server)


def test_correction_discards_entire_model_tail_even_when_mapped_segment_completes() -> None:
    client, server, _ = _client_server()
    try:
        observations = repeat(Observation("obs-correction", {"observation_id": "obs-correction", "robot": {}}))
        client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
        client.execute = lambda _actions, **kwargs: {  # type: ignore[method-assign]
            "status": "completed",
            "request_id": kwargs["request_id"],
            "result": {"executed_steps": 1},
        }

        def review(packet):
            return {
                "proposal_id": packet["proposal"]["proposal_id"],
                "observation_id": packet["observation"]["observation_id"],
                "decision": "correct",
                "execute_steps": 0,
                "corrections": [
                    {
                        "left": {
                            "reach_m": 0.1,
                            "height_m": 0.2,
                            "pan_deg": 0.0,
                            "wrist_flex_deg": 0.0,
                            "wrist_roll_deg": 0.0,
                            "gripper": 0.0,
                        },
                        "right": None,
                        "frame": "so101_shoulder_plane",
                        "units": "m_deg",
                        "duration_s": 0.2,
                    }
                ],
                "reason": "mapped correction test",
            }

        result = CooperativeLoop(
            client,
            proposal_provider=_proposal,
            reviewer=review,
            correction_mapper=lambda _corrections, *, state: ((0.0,) * 12,),
            post_observation_timeout_s=0.01,
        ).run_round()
        assert result.status == "post_observation_pending"
        assert result.executed_steps == 1
        assert result.discarded_steps == 50
        assert result.post_observation is None
    finally:
        _close(server)


def test_runnable_demo_uses_public_propose_review_execute_and_new_state() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{root}:{root / 'src'}"
    completed = subprocess.run(
        [sys.executable, "-m", "agents.astra_pi05", "--demo"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["proposal_steps"] == 50
    assert payload["executed_steps"] == 3
    assert payload["discarded_steps"] == 47
    assert payload["model_requests"] == 1
    assert payload["hardware_access"] is False
    assert payload["before_state"]["joint_positions"] == [0.0] * 12
    assert payload["after_state"]["joint_positions"] == [0.25] * 12
