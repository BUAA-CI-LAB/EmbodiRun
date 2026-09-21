from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodirun.services.inference import (
    ImagePayload,
    PolicyObservation,
    VvlaWirelessClient,
)


class FakeWirelessTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any] | None, float]] = []
        self.shutdown_calls = 0

    def request(self, method, payload, *, timeout_s):
        self.calls.append((method, payload, timeout_s))
        if method == "open_session":
            return {"session_id": "session-1", "session_revision": 0}
        if method == "step":
            return {
                "request_id": payload["request_id"],
                "session_id": payload["session_id"],
                "step_id": payload["step_id"],
                "session_revision": 1,
                "action_space": "pi05.action_chunk.v1",
                "actions": [{"type": "action_chunk", "values": {"data": [[0.0]]}}],
                "timing": {"policy_ms": 1.0},
            }
        raise AssertionError(f"unexpected method: {method}")

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_wireless_client_builds_structured_step_payload() -> None:
    transport = FakeWirelessTransport()
    client = VvlaWirelessClient(transport, timeout_s=2.5)
    session = client.open_session(robot_id="fr3", action_space="pi05.action_chunk.v1")
    result = client.step(
        PolicyObservation(
            session_id=session.session_id,
            request_id="request-1",
            step_id=0,
            instruction="move",
            state={"joints": [0.0]},
            images=(ImagePayload("wrist", "image/jpeg", b"jpeg-bytes"),),
        )
    )

    method, payload, timeout_s = transport.calls[1]
    assert method == "step"
    assert payload["schema"] == "vvla.policy.step.v1"
    assert payload["images"] == [{"name": "wrist", "mime_type": "image/jpeg", "data": b"jpeg-bytes"}]
    assert timeout_s == 2.5
    assert result.request_id == "request-1"


def test_wireless_client_timeout_view_does_not_close_shared_transport() -> None:
    transport = FakeWirelessTransport()
    owner = VvlaWirelessClient(transport, timeout_s=2.0)
    request_client = owner.with_timeout(60.0)

    request_client.open_session(robot_id="fr3", action_space="pi05.action_chunk.v1")
    request_client.shutdown()

    assert transport.calls[0][2] == 60.0
    assert transport.shutdown_calls == 0

    owner.shutdown()

    assert transport.shutdown_calls == 1
