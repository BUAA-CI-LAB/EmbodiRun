from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from embodirun.services.inference import (
    EmbodiInferHttpClient,
    ImagePayload,
    PolicyObservation,
)

EMBODIINFER_ROOT = Path(__file__).parents[1] / "third_party" / "embodiinfer"
sys.path.insert(0, str(EMBODIINFER_ROOT))
pytest.importorskip("torch", reason="pinned EmbodiInfer integration requires Torch")

from embodiinfer.engine.serve.contracts import ModelAction, ModelResult  # noqa: E402
from embodiinfer.engine.serve.http_server import (  # noqa: E402
    PolicyHttpService,
    create_http_server,
)


class Pi05StubAdapter:
    action_space = "pi05.action_chunk.v1"

    def __init__(self) -> None:
        self.infer_calls = 0
        self.reset_calls: list[str] = []

    def capabilities(self) -> dict[str, object]:
        return {
            "model": "pi05-stub",
            "action_space": self.action_space,
            "state_fields": ["joint_state"],
            "image_fields": ["camera.jpg"],
        }

    def infer(self, request) -> ModelResult:
        self.infer_calls += 1
        assert request.instruction == "pick up the cube"
        assert request.state == {"joint_state": [0.0] * 8}
        assert [image.name for image in request.images] == ["camera.jpg"]
        return ModelResult(
            action_space=self.action_space,
            actions=(
                ModelAction(
                    kind="action_chunk",
                    values={"data": [[0.0] * 7], "feature_names": []},
                ),
            ),
            timing={"policy_ms": 1.0},
            policy_revision="integration-test",
        )

    def reset(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


def test_deploy_client_round_trips_against_pinned_embodiinfer_http_server() -> None:
    adapter = Pi05StubAdapter()
    service = PolicyHttpService(
        adapter,
        token=None,
        max_body_bytes=64 * 1024 * 1024,
    )
    server = create_http_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = EmbodiInferHttpClient(f"http://{host}:{port}", timeout_s=5.0)

    try:
        assert client.health()["status"] == "ok"
        assert client.capabilities()["adapter"]["action_space"] == adapter.action_space
        session = client.open_session(
            robot_id="fr3-integration",
            action_space=adapter.action_space,
        )
        observation = PolicyObservation(
            session_id=session.session_id,
            request_id="request-0",
            step_id=0,
            instruction="pick up the cube",
            state={"joint_state": [0.0] * 8},
            images=(ImagePayload("camera.jpg", "image/jpeg", b"jpeg-bytes"),),
        )

        first = client.step(observation)
        replay = client.step(observation)

        assert first == replay
        assert first.action_space == adapter.action_space
        assert first.actions[0].kind == "action_chunk"
        assert first.actions[0].values["data"] == [[0.0] * 7]
        assert adapter.infer_calls == 1
        client.close(session.session_id)
        assert adapter.reset_calls == [session.session_id]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
