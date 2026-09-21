import json

from embodirun.services.inference import (
    HttpResponse,
    ImagePayload,
    PolicyObservation,
    VvlaHttpClient,
)


class FakeTransport:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, *, headers, body, timeout_s, maximum_bytes):
        self.calls.append((method, url, dict(headers), body, timeout_s, maximum_bytes))
        payload = {
            "request_id": "request-1",
            "session_id": "session-1",
            "step_id": 0,
            "session_revision": 1,
            "action_space": "franka.fr3.control.v1",
            "actions": [
                {
                    "type": "joint_position",
                    "values": {"joint_positions_rad": [0.0] * 7},
                }
            ],
            "timing": {"policy_ms": 10.0},
        }
        return HttpResponse(200, {"content-type": "application/json"}, json.dumps(payload).encode())


def test_step_uses_multipart_binary_images_and_idempotency_key() -> None:
    transport = FakeTransport()
    client = VvlaHttpClient("http://vvla:8000", transport=transport)
    result = client.step(
        PolicyObservation(
            session_id="session-1",
            request_id="request-1",
            step_id=0,
            instruction="pick up the cube",
            state={"joint_positions_rad": [0.0] * 7},
            images=(ImagePayload("wrist.jpg", "image/jpeg", b"jpeg-bytes"),),
        )
    )
    method, url, headers, body, _, _ = transport.calls[0]
    assert method == "POST"
    assert url.endswith("/v1/sessions/session-1/steps")
    assert headers["Idempotency-Key"] == "request-1"
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b"jpeg-bytes" in body
    assert b"base64" not in body
    assert result.actions[0].kind == "joint_position"
