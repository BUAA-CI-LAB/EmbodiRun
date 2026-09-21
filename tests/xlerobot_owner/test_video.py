import asyncio

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")
from aiohttp.test_utils import TestClient, TestServer
from aiortc import RTCBundlePolicy, RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from embodirun_xlerobot_owner.robot import DemoRobot
from embodirun_xlerobot_owner.server import Platform, create_app
from embodirun_xlerobot_owner.video import CameraVideoTrack

TOKEN = "video-test-token-not-real"


@pytest.mark.asyncio
async def test_actual_webrtc_encodes_and_receives_three_tracks(tmp_path):
    platform = Platform(DemoRobot(), TOKEN, output=tmp_path)
    client = TestClient(TestServer(create_app(platform)))
    receiver = RTCPeerConnection(RTCConfiguration(iceServers=[], bundlePolicy=RTCBundlePolicy.MAX_BUNDLE))
    received = []
    tasks = []

    @receiver.on("track")
    def track_added(track):
        async def consume():
            first = await track.recv()
            second = await track.recv()
            assert first.width == second.width == 640
            assert first.height == second.height == 480
            assert second.pts > first.pts
            received.append(track.id)

        tasks.append(asyncio.create_task(consume()))

    await client.start_server()
    try:
        assert (await client.post("/api/webrtc/offer", json={})).status == 401
        for _ in range(3):
            receiver.addTransceiver("video", direction="recvonly")
        await receiver.setLocalDescription(await receiver.createOffer())
        response = await client.post(
            "/api/webrtc/offer",
            headers={"Authorization": "Bearer " + TOKEN},
            json={"sdp": receiver.localDescription.sdp, "type": "offer"},
        )
        assert response.status == 200, await response.text()
        result = await response.json()
        assert result["cameras"] == ["front", "left_wrist", "right_wrist"]
        assert "H264" in result["sdp"]
        await receiver.setRemoteDescription(RTCSessionDescription(sdp=result["sdp"], type=result["type"]))
        await asyncio.wait_for(asyncio.gather(*tasks), 12)
        assert len(set(received)) == 3
        assert not platform.armed
        peer = next(iter(platform.video_service.peers))
        stats = await peer.getStats()
        outgoing = [s for s in stats.values() if s.type == "outbound-rtp"]
        assert len(outgoing) == 3 and all(s.bytesSent > 0 for s in outgoing)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await receiver.close()
        await client.close()
    assert not platform.video_service.peers


@pytest.mark.asyncio
async def test_no_stale_or_duplicate_camera_is_encoded_as_fresh(tmp_path):
    platform = Platform(DemoRobot(), TOKEN, output=tmp_path)
    platform.robot.connect()
    platform.observation, platform.images = platform.robot.read()
    platform.received_at = asyncio.get_running_loop().time()
    platform.connected = True
    track = CameraVideoTrack(platform, "front")
    first = await track.recv()
    assert first.width == 640
    next_frame = asyncio.create_task(track.recv())
    await asyncio.sleep(0.08)
    assert not next_frame.done()
    platform.observation, platform.images = platform.robot.read()
    platform.received_at = 0
    await asyncio.sleep(0.02)
    assert not next_frame.done()
    platform.received_at = asyncio.get_running_loop().time()
    await asyncio.wait_for(next_frame, 1)
    assert track.frames_encoded == 2
    track.stop()
