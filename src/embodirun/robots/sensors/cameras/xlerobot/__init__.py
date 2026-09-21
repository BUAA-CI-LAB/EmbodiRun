"""Read shared camera JPEGs from the existing XLeRobot owner; no camera SDK."""

from __future__ import annotations

import base64
import time
from collections.abc import Sequence

from embodirun.robots.lerobot.xlerobot.adapter import XLeRobotAdapter
from embodirun.robots.lerobot.xlerobot.config import XLeRobotConfig
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras.camera import CameraFrame


class XLeRobotCameraSource:
    def __init__(self, inputs: Sequence[SensorInput]):
        self.inputs = tuple(inputs)
        if not self.inputs:
            raise ValueError("XLeRobot camera inputs are required")
        self.clients = {}
        self.closed = False
        for item in self.inputs:
            options = dict(item.options)
            camera = options.pop("camera", None)
            if not isinstance(camera, str) or not camera:
                raise ValueError("XLeRobot camera requires its owner camera name")
            config = XLeRobotConfig.from_mapping("camera-reader", options)
            key = (config.url, config.token, config.scope, config.timeout_s)
            if key not in self.clients:
                self.clients[key] = XLeRobotAdapter(config)

    def capture(self) -> tuple[CameraFrame, ...]:
        if self.closed:
            raise RuntimeError("XLeRobot camera source is closed")
        replies = {}
        for key, client in self.clients.items():
            started = time.monotonic_ns()
            replies[key] = (started, client._request("observe"), time.monotonic_ns())
        frames = []
        for item in self.inputs:
            options = dict(item.options)
            camera = options.pop("camera")
            config = XLeRobotConfig.from_mapping("camera-reader", options)
            started, reply, received = replies[(config.url, config.token, config.scope, config.timeout_s)]
            observation = reply["observation"]
            age = observation.get("camera_ages_ns", {}).get(camera)
            if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= started:
                raise ValueError(f"owner did not report elapsed capture age for {camera}")
            data = base64.b64decode(reply["images"][camera], validate=True)
            frames.append(
                CameraFrame(
                    name=item.name,
                    mime_type="image/jpeg",
                    data=data,
                    captured_timestamp_ns=started - age,
                    received_timestamp_ns=received,
                    clock_domain="host_monotonic_ns",
                    profile={
                        "source": "xlerobot.external_owner",
                        "camera": camera,
                        "remote_capture_timestamp_ns": observation.get("camera_timestamps_ns", {}).get(camera),
                    },
                )
            )
        return tuple(frames)

    def close(self) -> None:
        # The source never armed anything and must never release another owner.
        self.closed = True


def create_source(inputs: Sequence[SensorInput]) -> XLeRobotCameraSource:
    return XLeRobotCameraSource(inputs)
