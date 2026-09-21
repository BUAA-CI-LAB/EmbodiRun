"""The deployment binding sends RGB only through the existing policy client."""

import asyncio
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from embodirun.bindings.xlerobot.lightnav0.client import (
    ACTION_SPACE,
    LightNav0Client,
)
from embodirun.services.inference import PolicyAction, PolicyResult, Session


class FakeClient:
    def __init__(self):
        self.requests = []
        self.closed = []
        self.reset_calls = []
        self.values = {
            "data": [[0.1, 0.0, 0.2]] * 10,
            "stop": False,
            "raw_text": "action",
        }
        self.response_change = lambda result: result

    def health(self):
        return {"status": "ok"}

    def capabilities(self):
        return {"adapter": {"action_space": ACTION_SPACE}}

    def open_session(self, *, robot_id, action_space):
        assert robot_id == "xlerobot" and action_space == ACTION_SPACE
        return Session("remote-session", 0)

    def reset(self, session_id, *, request_id):
        self.reset_calls.append((session_id, request_id))
        return Session(session_id, 1)

    def close(self, session_id):
        self.closed.append(session_id)

    def step(self, observation):
        self.requests.append(observation)
        return self.response_change(
            PolicyResult(
                request_id=observation.request_id,
                session_id=observation.session_id,
                step_id=observation.step_id,
                session_revision=0,
                action_space=ACTION_SPACE,
                actions=(PolicyAction("waypoints", self.values),),
            )
        )


def test_lightnav0_client_preserves_rgb_and_remote_session_order():
    transport = FakeClient()
    client = LightNav0Client(transport)
    rgb = np.array([[[1, 7, 255], [0, 9, 100]]], dtype=np.uint8)

    async def run():
        await client.start()
        await client.reset_session("local")
        prediction = await client.predict(rgb, instruction="go", session_id="local", timestamp_s=2.5)
        assert prediction.output.metadata["stop"] is False
        await client.predict(rgb, instruction="go", session_id="local", timestamp_s=3.0)
        await client.reset_session("local")
        await client.predict(rgb, instruction="go", session_id="local", timestamp_s=0.0)
        await client.aclose()

    asyncio.run(run())
    assert [r.step_id for r in transport.requests] == [0, 1, 0]
    assert len({r.request_id for r in transport.requests}) == 3
    request = transport.requests[0]
    assert request.state == {}
    assert request.metadata == {"timestamp_s": 2.5}
    with Image.open(BytesIO(request.images[0].data)) as image:
        np.testing.assert_array_equal(image, rgb)
    assert transport.closed == ["remote-session"]
    assert len(transport.reset_calls) == 1


@pytest.mark.parametrize(
    "values",
    [
        {"data": [[0, 0, 0]] * 9, "stop": False},
        {"data": [[0, 0, 0]] * 10, "stop": "false"},
        {"data": [[True, 0, 0]] * 10, "stop": False},
        {"data": [[float("nan"), 0, 0]] * 10, "stop": False},
    ],
)
def test_lightnav0_client_rejects_malformed_actions(values):
    transport = FakeClient()
    transport.values = values
    client = LightNav0Client(transport)

    async def run():
        await client.reset_session("local")
        with pytest.raises(ValueError):
            await client.predict(
                np.zeros((1, 1, 3), dtype=np.uint8),
                instruction="go",
                session_id="local",
                timestamp_s=0,
            )
        await client.aclose()

    asyncio.run(run())


def test_lightnav0_client_rejects_mismatched_response():
    transport = FakeClient()
    transport.response_change = lambda result: replace(result, request_id="wrong")
    client = LightNav0Client(transport)

    async def run():
        await client.reset_session("local")
        with pytest.raises(ValueError, match="current request"):
            await client.predict(
                np.zeros((1, 1, 3), dtype=np.uint8),
                instruction="go",
                session_id="local",
                timestamp_s=0,
            )

    asyncio.run(run())


def test_lightnav0_client_rejects_wrong_service_before_opening_session():
    transport = FakeClient()
    transport.capabilities = lambda: {"adapter": {"action_space": "pi05"}}
    with pytest.raises(ValueError, match="advertise"):
        asyncio.run(LightNav0Client(transport).start())
    assert transport.requests == []
