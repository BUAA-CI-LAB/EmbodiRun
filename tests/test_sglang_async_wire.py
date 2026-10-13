"""SGLang action payloads preserve the shared async-inference contract."""

from __future__ import annotations

import json

from embodirun.application.async_inference.contracts import (
    RTCAsyncRequest,
    build_async_metadata,
)
from embodirun.model_services import ImagePayload, PolicyObservation
from embodirun.model_services.backends.sglang.http import SglangHttpClient
from embodirun.model_services.protocols.http import HttpResponse


class _Transport:
    def __init__(self) -> None:
        self.posts: list[dict] = []

    def request(self, method, url, *, headers, body, timeout_s, maximum_bytes):
        del headers, timeout_s, maximum_bytes
        payload = json.loads(body) if body is not None else None
        if method == "POST":
            self.posts.append(payload)
            response = {
                "id": "request-1",
                "data": [{"action": {"values": [[0.1, 0.2]]}}],
            }
        else:
            response = {"async_inference": {"rtc": True}}
        return HttpResponse(200, {"content-type": "application/json"}, json.dumps(response).encode())


def test_sglang_client_forwards_rtc_metadata() -> None:
    transport = _Transport()
    client = SglangHttpClient("http://sglang", transport=transport, output_action_dim=2)
    session = client.open_session(robot_id="robot-1", action_space="pi05")
    observation = PolicyObservation(
        session_id=session.session_id,
        request_id="request-1",
        step_id=0,
        instruction="pick up the block",
        state={"observation.state": [0.0, 1.0]},
        images=(ImagePayload("front", "image/png", b"png"),),
        metadata=build_async_metadata(
            rtc=RTCAsyncRequest(None, 3, 8, cache_start_index=10),
        ),
    )

    client.step(observation)

    sent = transport.posts[0]["input"]["observation"]
    assert sent["metadata"]["async"]["schema"] == "embodiinfer.async.v1"
    assert sent["metadata"]["async"]["rtc"]["inference_delay"] == 3
    assert sent["metadata"]["async"]["rtc"]["cache_start_index"] == 10


def test_sglang_client_maps_both_libero_cameras_to_checkpoint_keys() -> None:
    transport = _Transport()
    client = SglangHttpClient(
        "http://sglang",
        transport=transport,
        output_action_dim=2,
        image_keys={
            "observation.images.image": "image",
            "observation.images.wrist_image": "image2",
        },
    )
    session = client.open_session(robot_id="robot-1", action_space="pi05")
    observation = PolicyObservation(
        session_id=session.session_id,
        request_id="request-1",
        step_id=0,
        instruction="pick up the block",
        state={"observation.state": [0.0, 1.0]},
        images=(
            ImagePayload("observation.images.image", "image/png", b"front"),
            ImagePayload("observation.images.wrist_image", "image/png", b"wrist"),
        ),
    )

    client.step(observation)

    images = transport.posts[0]["input"]["observation"]["images"]
    assert set(images) == {"image", "image2"}
    assert images["image"]["base64"] != images["image2"]["base64"]
