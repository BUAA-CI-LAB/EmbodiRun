"""Live local integration coverage for the software-only inference walkthrough.

The control HTTP server, service, simulated adapter, mapper, camera replay, and
recorder are real objects.  Only the EmbodiInfer HTTP client is replaced with a
deterministic transport endpoint, so this test does not need a model process,
GPU, network device, or physical robot.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Mapping

import pytest
from examples import run_shared_device_inference as experiment

from embodirun.services.inference import PolicyAction, PolicyObservation, PolicyResult, Session


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_record(tmp_path):
    front = tmp_path / "front.jpg"
    wrist = tmp_path / "wrist.jpg"
    # CameraFrame validates the encoding marker and preserves these bytes.
    front.write_bytes(b"\xff\xd8\xfffront-replay")
    wrist.write_bytes(b"\xff\xd8\xffwrist-replay")
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence": 13,
                "cameras": {"2": front.name, "0": wrist.name},
                "present": {
                    "shoulder_pan": 1.0,
                    "shoulder_lift": 2.0,
                    "elbow_flex": 3.0,
                    "wrist_flex": 4.0,
                    "wrist_roll": 5.0,
                    "gripper": 6.0,
                },
                # Keep command-side values different from measured present so
                # the test proves the first model input comes from present.
                "action": {
                    "shoulder_pan.pos": 101.0,
                    "shoulder_lift.pos": 102.0,
                    "elbow_flex.pos": 103.0,
                    "wrist_flex.pos": 104.0,
                    "wrist_roll.pos": 105.0,
                    "gripper.pos": 106.0,
                },
                "captured_timestamp_ns": 123456789,
                "clock_domain": "dataset_clock",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, front, wrist


class _DeterministicEmbodiInferClient:
    """Small in-process EmbodiInfer endpoint used through the normal client seam."""

    _lock = threading.Lock()
    states: list[tuple[float, ...]] = []
    sessions: list[str] = []
    closed_sessions: list[str] = []
    outputs = ((10.0, 11.0, 12.0, 13.0, 14.0, 15.0), (20.0, 21.0, 22.0, 23.0, 24.0, 25.0))

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 5.0,
        transport: object | None = None,
    ) -> None:
        del token, timeout_s, transport
        self.base_url = base_url

    @classmethod
    def reset_calls(cls) -> None:
        with cls._lock:
            cls.states.clear()
            cls.sessions.clear()
            cls.closed_sessions.clear()

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def capabilities(self) -> dict[str, object]:
        return {
            "action_space": experiment.POLICY_ACTION_SPACE,
            "state_fields": ["state_native"],
            "action_feature_names": list(experiment.POLICY_FEATURE_NAMES),
            "return_steps": experiment.PROPOSED_STEPS,
        }

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, object] | None = None,
    ) -> Session:
        del robot_id, action_space, metadata
        with self._lock:
            session_id = f"replay-session-{len(self.sessions) + 1}"
            self.sessions.append(session_id)
        return Session(session_id=session_id, revision=0)

    def step(self, observation: PolicyObservation) -> PolicyResult:
        raw_state = observation.state.get("state_native")
        assert isinstance(raw_state, list)
        state = tuple(float(value) for value in raw_state)
        with self._lock:
            call_index = len(self.states)
            self.states.append(state)
        action_row = self.outputs[call_index]
        return PolicyResult(
            request_id=observation.request_id,
            session_id=observation.session_id,
            step_id=observation.step_id,
            session_revision=0,
            action_space=experiment.POLICY_ACTION_SPACE,
            actions=(
                PolicyAction(
                    kind="action_chunk",
                    values={
                        "feature_names": list(experiment.POLICY_FEATURE_NAMES),
                        "data": [list(action_row) for _ in range(experiment.PROPOSED_STEPS)],
                    },
                ),
            ),
            timing={"inference_s": 0.0},
            policy_revision="deterministic-test",
        )

    def reset(self, session_id: str, *, request_id: str) -> Session:
        del request_id
        return Session(session_id=session_id, revision=1)

    def close(self, session_id: str) -> None:
        with self._lock:
            self.closed_sessions.append(session_id)


def test_run_experiment_uses_present_then_shared_readback_and_records_actions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest, front, wrist = _write_record(tmp_path)
    _DeterministicEmbodiInferClient.reset_calls()
    monkeypatch.setattr(experiment, "EmbodiInferHttpClient", _DeterministicEmbodiInferClient)

    output_dir = tmp_path / "output"
    state_dir = tmp_path / "state"
    report = experiment.run_experiment(
        "http://embodiinfer.test.invalid:1",
        manifest,
        output_dir=output_dir,
        state_dir=state_dir,
        model_units=experiment.MODEL_UNITS,
        port=_free_port(),
        control_hz=100.0,
        inference_timeout_s=5.0,
    )

    # The first policy request saw measured ``present``.  The second saw the
    # readback after the first three actions, proving shared owner reuse rather
    # than reopening a fresh adapter for each task/session.
    assert _DeterministicEmbodiInferClient.states == [
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
    ]
    assert len(_DeterministicEmbodiInferClient.sessions) == 2
    assert len(set(_DeterministicEmbodiInferClient.sessions)) == 2

    assert report["status"] == "software_complete"
    assert report["owners"]["robot_owner_reused"] is True
    assert report["owners"]["camera_source_count"] == 1
    assert report["model_trace"]["step_count"] == 2
    assert len(report["tasks"]) == 2
    assert report["tasks"][0]["observation_after"] != report["tasks"][1]["observation_after"]
    assert all(task["stale_after"] is False for task in report["tasks"])
    assert report["shared_consumer_observation_ids"]

    raw_results = json.loads((output_dir / "raw-model-results.json").read_text(encoding="utf-8"))
    assert [tuple(raw["actions"][0]["values"]["data"][2]) for raw in raw_results["results"]] == [
        (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        (20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
    ]

    recording = report["recording"]
    assert recording["state"] == "stopped"
    # Recording starts after the initial read; each policy task then publishes
    # one shared snapshot consumed by both the runtime and recorder.
    assert recording["observation_count"] >= 2
    assert recording["action_count"] >= 6
    assert report["recording_validation"]["paired_observations"] >= 2
    recording_dir = state_dir / "recordings" / "policy-vector-replay"
    actions = [
        json.loads(line)
        for line in (recording_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert actions
    assert all(item["kind"] == "action" for item in actions)
    action_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for item in actions:
        metadata = item["payload"]["metadata"]
        key = (str(metadata["request_id"]), int(metadata["chunk_index"]))
        action_groups.setdefault(key, []).append(item)
    assert len(action_groups) == 6
    for events in action_groups.values():
        assert {item["stage"] for item in events} >= {"queued", "executed"}
        assert {item["action_id"] for item in events} == {events[0]["action_id"]}
        terminal = next(item for item in events if item["stage"] == "executed")
        receipt = terminal["payload"]["driver_receipt"]
        assert receipt["simulated"] is True
        assert receipt["hardware_access"] is False
        assert receipt["measured"]["state_native"] == terminal["payload"]["requested"]["state_native"]
    assert (recording_dir / "media" / "00000001" / "000.jpg").read_bytes() == front.read_bytes()
    assert (recording_dir / "media" / "00000001" / "001.jpg").read_bytes() == wrist.read_bytes()
