"""Low-latency WebRTC viewing; original JPEGs remain the recording source.

Encoding belongs on the Mac, not in the motor/AGX control loop. Each receiver
consumes the latest camera frame, never an unbounded video queue. LAN-only ICE
does not contact public STUN/TURN servers.
"""

from __future__ import annotations

import asyncio
import time
from fractions import Fraction
from typing import Any

import av
from aiortc import (
    RTCBundlePolicy,
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.mediastreams import MediaStreamError

from .control import CAMERAS


class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, platform: Any, camera: str, fps: float = 20):
        super().__init__()
        self.platform, self.camera, self.fps = platform, camera, fps
        self.decoder = av.CodecContext.create("mjpeg", "r")
        self.last_stamp = None
        self.started = time.monotonic()
        self.next_frame_at = self.started
        self.frames_encoded = 0

    def _decode(self, data: bytes) -> av.VideoFrame:
        frames = self.decoder.decode(av.Packet(data))
        if not frames:
            raise ValueError("camera JPEG did not decode")
        return frames[0]

    async def recv(self) -> av.VideoFrame:
        while self.readyState == "live":
            now = time.monotonic()
            obs = self.platform.observation
            stamp = obs.get("camera_timestamps_ns", {}).get(self.camera)
            source = obs.get("source_timestamp_ns")
            data = self.platform.images.get(self.camera)
            if (
                self.platform.connected
                and now - self.platform.received_at < 0.5
                and isinstance(stamp, int)
                and isinstance(source, int)
                and abs(source - stamp) <= 500_000_000
                and stamp != self.last_stamp
                and data
                and now >= self.next_frame_at
            ):
                frame = await asyncio.to_thread(self._decode, data)
                frame.pts = round((time.monotonic() - self.started) * 90_000)
                frame.time_base = Fraction(1, 90_000)
                self.last_stamp = stamp
                # An absolute schedule avoids accumulating timer drift
                # against the independently sampled AGX producer.
                self.next_frame_at = max(self.next_frame_at + 1 / self.fps, now)
                self.frames_encoded += 1
                return frame
            # A lost camera stops producing frames instead of repeatedly
            # painting an old frame as "live". The browser stops control.
            await asyncio.sleep(0.005)
        raise MediaStreamError


class VideoService:
    def __init__(self, platform: Any, max_viewers: int = 3):
        self.platform, self.max_viewers = platform, max_viewers
        self.peers: set[RTCPeerConnection] = set()
        self.tracks: dict[RTCPeerConnection, list[CameraVideoTrack]] = {}
        self.expiries: set[asyncio.Task] = set()

    async def remove(self, peer: RTCPeerConnection) -> None:
        self.peers.discard(peer)
        for track in self.tracks.pop(peer, []):
            track.stop()
        await peer.close()

    async def offer(self, data: dict) -> dict:
        if not isinstance(data, dict) or data.get("type") != "offer":
            raise ValueError("WebRTC offer required")
        sdp = data.get("sdp")
        if not isinstance(sdp, str) or len(sdp) > 60_000:
            raise ValueError("invalid video SDP")
        media = [line for line in sdp.splitlines() if line.startswith("m=")]
        if len(media) != 3 or any(not line.startswith("m=video ") for line in media):
            raise ValueError("offer must contain exactly three receive-only video tracks")
        if len(self.peers) >= self.max_viewers:
            raise RuntimeError("at most three video viewers; close another viewer and retry")
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[], bundlePolicy=RTCBundlePolicy.MAX_BUNDLE))
        self.peers.add(peer)
        self.tracks[peer] = []

        @peer.on("connectionstatechange")
        async def state_changed():
            if peer.connectionState in ("failed", "closed") and peer in self.peers:
                await self.remove(peer)

        async def negotiate():
            await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            transceivers = peer.getTransceivers()
            if len(transceivers) != 3 or any(t.kind != "video" for t in transceivers):
                raise ValueError("three video receivers required")
            capabilities = RTCRtpSender.getCapabilities("video").codecs
            # Prefer H.264, with VP8 available if the headset doesn't offer it.
            codecs = sorted(capabilities, key=lambda c: c.mimeType.lower() != "video/h264")
            for camera, transceiver in zip(CAMERAS, transceivers):
                track = CameraVideoTrack(self.platform, camera, fps=min(self.platform.fps, 20))
                self.tracks[peer].append(track)
                transceiver.sender.replaceTrack(track)
                transceiver.direction = "sendonly"
                transceiver.setCodecPreferences(codecs)
            await peer.setLocalDescription(await peer.createAnswer())

        try:
            await asyncio.wait_for(negotiate(), timeout=10)
        except BaseException:
            await self.remove(peer)
            raise

        async def expire_unconnected():
            await asyncio.sleep(20)
            if peer in self.peers and peer.connectionState != "connected":
                await self.remove(peer)

        expiry = asyncio.create_task(expire_unconnected())
        self.expiries.add(expiry)
        expiry.add_done_callback(self.expiries.discard)
        return {
            "type": peer.localDescription.type,
            "sdp": peer.localDescription.sdp,
            "cameras": list(CAMERAS),
        }

    async def close(self):
        for task in self.expiries:
            task.cancel()
        await asyncio.gather(*self.expiries, return_exceptions=True)
        await asyncio.gather(*(self.remove(peer) for peer in list(self.peers)))
