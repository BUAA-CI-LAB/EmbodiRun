from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from agents.astra_pi05.recording import SessionRecorder
from agents.astra_pi05.reviewer import MODEL, AstraCodexReviewer
from agents.rpent.session import PublicCooperativeSession, normalize_public_proposal

from embodirun.client import ControlClient, Observation

NAMES = tuple(f"joint_{index}" for index in range(12))
META = {
    "action_semantics": "biso101_so101_v1",
    "action_layout": "left6_right6",
    "action_encoding": "absolute",
    "joint_position_unit": "degrees",
    "action_dim": 12,
    "horizon": 50,
    "feature_names": list(NAMES),
}


def _client() -> ControlClient:
    return ControlClient("http://127.0.0.1:1", caller_id="test", session_id="session")


def _proposal_response() -> dict[str, Any]:
    return {
        "status": "proposed",
        "proposal_id": "proposal-1",
        "actions": [[float(index)] * 12 for index in range(50)],
        "metadata": dict(META),
    }


def _observations() -> list[Observation]:
    return [
        Observation(
            "obs-1",
            {"observation_id": "obs-1", "sequence": 1, "state": [0.0] * 12},
        ),
        Observation(
            "obs-2",
            {"observation_id": "obs-2", "sequence": 2, "state": [0.0] * 12},
        ),
        Observation(
            "obs-3",
            {"observation_id": "obs-3", "sequence": 3, "state": [1.0] * 12},
        ),
    ]


def test_public_session_runs_propose_review_execute_reobserve_and_records(tmp_path: Path) -> None:
    client = _client()
    observations = iter(_observations())
    client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
    proposals: list[dict[str, Any]] = []
    client.propose = lambda **kwargs: proposals.append(kwargs) or _proposal_response()  # type: ignore[method-assign]
    executed: list[tuple[Any, dict[str, Any]]] = []

    def execute(actions, **kwargs):
        executed.append((actions, kwargs))
        return {
            "status": "completed",
            "request_id": kwargs["request_id"],
            "result": {"executed_steps": len(actions)},
        }

    client.execute = execute  # type: ignore[method-assign]
    reviewer_packets: list[dict[str, Any]] = []

    def reviewer(packet):
        reviewer_packets.append(packet)
        return {
            "proposal_id": packet["proposal"]["proposal_id"],
            "observation_id": packet["observation"]["observation_id"],
            "decision": "execute_prefix",
            "execute_steps": 3,
            "corrections": [],
            "reason": "fake Astra selected a short prefix",
        }

    recorder = SessionRecorder(tmp_path / "recording")
    result = PublicCooperativeSession(
        client,
        reviewer=reviewer,
        instruction="move",
        feature_names=NAMES,
        recorder=recorder,
        max_rounds=1,
    ).run()
    recorder.close()
    assert result["status"] == "completed"
    assert result["task_success"] == "unverified"
    assert proposals[0]["observation_id"] == "obs-1"
    assert len(executed) == 1 and len(executed[0][0]) == 3
    assert reviewer_packets[0]["proposal"]["observation_id"] == "obs-1"
    assert result["rounds"][0]["post_observation_id"] == "obs-3"
    events = [json.loads(line) for line in (tmp_path / "recording" / "events.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == [
        "session_started",
        "execute_submitted",
        "round_result",
    ]
    assert json.loads((tmp_path / "recording" / "status.json").read_text())["status"] == "completed"


def test_public_proposal_normalization_rejects_missing_explicit_contract() -> None:
    payload = {"proposal_id": "p", "actions": [[0.0] * 12 for _ in range(50)]}
    try:
        normalize_public_proposal(payload, observation_id="o")
    except ValueError:
        pass
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing action metadata must be rejected")


def test_public_proposal_metadata_conflict_cannot_relabel_declared_units() -> None:
    payload = _proposal_response()
    payload["metadata"]["joint_position_unit"] = "range_m100_100"
    try:
        normalize_public_proposal(
            payload,
            observation_id="obs-1",
            metadata={"joint_position_unit": "degrees"},
        )
    except ValueError as error:
        assert "conflicting proposal metadata" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("caller metadata must not relabel model units")


def test_correction_without_mapper_is_unsupported_and_mapper_receives_observation() -> None:
    client = _client()
    observations = iter([_observations()[0]])
    client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
    client.propose = lambda **_kwargs: _proposal_response()  # type: ignore[method-assign]
    called: list[Observation] = []

    def reviewer(packet):
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
            "reason": "fake Astra correction",
        }

    def mapper(corrections, *, observation):
        called.append(observation)
        return [[0.0] * 12]

    client.execute = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute"))  # type: ignore[method-assign]
    unsupported = PublicCooperativeSession(
        client,
        reviewer=reviewer,
        instruction="move",
        feature_names=NAMES,
        max_rounds=1,
    ).run()
    assert unsupported["status"] == "unsupported"

    observations = iter(_observations())
    client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
    client.execute = lambda actions, **kwargs: {  # type: ignore[method-assign]
        "status": "completed",
        "request_id": kwargs["request_id"],
        "result": {"executed_steps": len(actions)},
    }
    mapped = PublicCooperativeSession(
        client,
        reviewer=reviewer,
        instruction="move",
        feature_names=NAMES,
        correction_mapper=mapper,
        max_rounds=1,
        post_observation_timeout_s=0.01,
    ).run()
    assert mapped["status"] in {"completed", "post_observation_pending"}
    assert called and called[-1].observation_id == "obs-1"


def test_cancel_targets_request_before_blocking_public_execute() -> None:
    client = _client()
    observations = iter(_observations())
    client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
    client.propose = lambda **_kwargs: _proposal_response()  # type: ignore[method-assign]
    entered = threading.Event()
    release = threading.Event()
    cancelled: list[str] = []

    def execute(actions, **kwargs):
        entered.set()
        release.wait(1.0)
        return {
            "status": "cancelled",
            "request_id": kwargs["request_id"],
        }

    client.execute = execute  # type: ignore[method-assign]
    client.cancel = lambda request_id: (
        cancelled.append(request_id)
        or {  # type: ignore[method-assign]
            "status": "cancel_requested",
            "request_id": request_id,
        }
    )

    def reviewer(packet):
        return {
            "proposal_id": packet["proposal"]["proposal_id"],
            "observation_id": packet["observation"]["observation_id"],
            "decision": "execute_prefix",
            "execute_steps": 1,
            "corrections": [],
            "reason": "blocking fake reviewer",
        }

    session = PublicCooperativeSession(
        client,
        reviewer=reviewer,
        instruction="move",
        feature_names=NAMES,
    )
    result_holder: list[dict[str, Any]] = []
    thread = threading.Thread(target=lambda: result_holder.append(session.run()), daemon=True)
    thread.start()
    assert entered.wait(1.0)
    response = session.cancel()
    release.set()
    thread.join(1.0)
    assert response["request_id"] == cancelled[0]
    assert result_holder[0]["status"] == "cancelled"


def test_session_does_not_continue_after_invalid_execution_evidence() -> None:
    client = _client()
    observations = iter(
        [
            Observation("obs-1", {"observation_id": "obs-1", "sequence": 1}),
            Observation("obs-2", {"observation_id": "obs-2", "sequence": 2}),
            Observation("obs-3", {"observation_id": "obs-3", "sequence": 3}),
        ]
    )
    client.observe = lambda **_kwargs: next(observations)  # type: ignore[method-assign]
    proposals = 0

    def propose(**_kwargs):
        nonlocal proposals
        proposals += 1
        return _proposal_response()

    client.propose = propose  # type: ignore[method-assign]
    client.execute = lambda actions, **kwargs: {  # type: ignore[method-assign]
        "status": "completed",
        "request_id": kwargs["request_id"],
        "result": {"executed_steps": -1},
    }

    def reviewer(packet):
        return {
            "proposal_id": packet["proposal"]["proposal_id"],
            "observation_id": packet["observation"]["observation_id"],
            "decision": "execute_prefix",
            "execute_steps": 1,
            "corrections": [],
            "reason": "invalid evidence fixture",
        }

    result = PublicCooperativeSession(
        client,
        reviewer=reviewer,
        instruction="move",
        feature_names=NAMES,
        max_rounds=3,
        post_observation_timeout_s=0.01,
    ).run()
    assert proposals == 1
    assert result["round_count"] == 1


def test_run_skill_rejects_unconfigured_skill() -> None:
    session = PublicCooperativeSession(_client(), reviewer=lambda _packet: {}, instruction="move")
    try:
        session.run_skill("arbitrary", "move")
    except ValueError as error:
        assert "unsupported skill" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("an unconfigured skill must not be silently executed")


def test_astra_reviewer_uses_exact_model_with_fake_runner(tmp_path: Path) -> None:
    images = {}
    for name in ("front", "left_wrist", "right_wrist"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"fake-image")
        images[name] = str(path.resolve())
    packet = {
        "observation": {
            "observation_id": "obs-1",
            "state": [0.0] * 12,
            "images": images,
        },
        "proposal": {
            "proposal_id": "prop-1",
            "observation_id": "obs-1",
            "actions": [[0.0] * 12 for _ in range(50)],
            "metadata": {key: value for key, value in META.items() if key != "feature_names"},
            "trajectory_preview": {
                "frame": "so101_shoulder_plane",
                "left": [[0.1, 0.2] for _ in range(50)],
                "right": [[-0.1, 0.2] for _ in range(50)],
            },
        },
    }
    seen: dict[str, Any] = {}

    async def runner(argv, cwd):
        seen["argv"] = list(argv)
        output = Path(argv[argv.index("--output-last-message") + 1])
        output.write_text(
            json.dumps(
                {
                    "proposal_id": "prop-1",
                    "observation_id": "obs-1",
                    "decision": "hold",
                    "execute_steps": 0,
                    "corrections": [],
                    "reason": "fake review",
                }
            )
        )
        return 0, "", ""

    result = asyncio.run(AstraCodexReviewer(command_runner=runner).review(packet))
    assert result["decision"] == "hold"
    argv = seen["argv"]
    assert argv[argv.index("--model") + 1] == MODEL
    assert sum(item == "--image" for item in argv) == 4
