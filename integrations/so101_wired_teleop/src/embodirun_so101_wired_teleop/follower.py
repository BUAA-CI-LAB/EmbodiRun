"""Drive one follower arm from the leader that owns its UDP port.

Everything that can move an arm lives here, so the safety limits are enforced on
this side and never trusted to the sender:

* only datagrams from the configured leader address are accepted
* the receive queue is drained so a burst can never replay stale targets
* the commanded target advances by at most ``max_step`` per cycle and may not
  lead the measured position by more than ``max_lead``
* a stream that stops for ``watchdog_s`` holds the last position instead of
  releasing the arm
"""

from __future__ import annotations

import logging
import signal
import socket
import time
from pathlib import Path
from typing import Any

from .config import FollowerConfig, LeaderConfig, TeleopConfig
from .protocol import PACKET, ProtocolError, decode, is_newer
from .recorder import EpisodeRecorder

_LOG = logging.getLogger(__name__)


def _follower_sdk() -> tuple[Any, Any]:
    """Import the LeRobot follower stack only when hardware is actually used."""

    try:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "the follower node needs the 'hardware' extra: pip install 'embodirun-so101-wired-teleop[hardware]'"
        ) from error
    return SOFollowerRobotConfig, SOFollower


def clamp_target(
    desired: dict[str, float],
    present: dict[str, float],
    commanded: dict[str, float],
    *,
    max_step: float,
    max_lead: float,
) -> dict[str, float]:
    """Advance ``commanded`` toward ``desired`` by one bounded step.

    ``max_step`` bounds how far one cycle may move; ``max_lead`` bounds how far
    the target may run ahead of the measured position. Together they turn a
    jumped or stale target into a slow, observable approach rather than a lurch.
    """

    result: dict[str, float] = {}
    for key, target in desired.items():
        motor = key.removesuffix(".pos")
        previous = commanded.get(key, present[motor])
        delta = max(-max_step, min(max_step, target - previous))
        stepped = previous + delta
        result[key] = max(
            present[motor] - max_lead,
            min(present[motor] + max_lead, stepped),
        )
    return result


class FollowerNode:
    """Receive targets from one leader and drive one follower arm."""

    def __init__(self, config: TeleopConfig, follower: FollowerConfig) -> None:
        self.config = config
        self.follower = follower
        self.leader: LeaderConfig = config.leader(follower.leader)

    def calibration_path(self) -> Path:
        return Path(self.follower.calibration_dir) / f"{self.follower.arm_id}.json"

    def bind_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, PACKET.size * 8)
        sock.bind((self.follower.bind, self.leader.port))
        sock.settimeout(0.10)
        return sock

    def network_test(self, *, timeout_s: float = 10.0) -> int:
        """Report whether the leader's packets arrive, without touching the arm."""

        sock = self.bind_socket()
        try:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    data, peer = sock.recvfrom(256)
                except TimeoutError:
                    continue
                if peer[0] != self.leader.advertise:
                    continue
                try:
                    sequence, action = decode(data, probe=True)
                except ProtocolError:
                    continue
                print(f"NETWORK_OK follower={self.follower.id} seq={sequence} joints={len(action)} from={peer[0]}")
                return 0
            print(f"NETWORK_TIMEOUT follower={self.follower.id}")
            return 2
        finally:
            sock.close()

    def run(self, *, direct: bool = False) -> int:
        """Run the control loop until interrupted."""

        sdk_config, sdk_follower = _follower_sdk()
        calibration = self.calibration_path()
        if not calibration.is_file():
            raise SystemExit(f"missing follower calibration: {calibration}")

        recorder = (
            EpisodeRecorder(
                self.follower.record_dir,
                list(self.follower.cameras),
                self.follower.camera_fps,
                self.follower.camera_roles,
            )
            if self.follower.record_dir
            else None
        )

        sock = self.bind_socket()
        arm = sdk_follower(
            sdk_config(
                port=self.follower.serial,
                id=self.follower.arm_id,
                calibration_dir=Path(self.follower.calibration_dir),
                # Smoothing happens in clamp_target: LeRobot's present-relative
                # limiter can stall under load when the first small step does not
                # actually move the servo.
                max_relative_target=None,
            )
        )

        stopping = False
        episode_request: str | None = None

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stopping
            stopping = True

        def request_episode(request: str) -> Any:
            def handler(_signum: int, _frame: Any) -> None:
                nonlocal episode_request
                episode_request = request

            return handler

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        if recorder is not None:
            signal.signal(signal.SIGUSR1, request_episode("start"))
            signal.signal(signal.SIGUSR2, request_episode("stop"))

        connected = False
        last_rx = 0.0
        last_sequence = -1
        commanded: dict[str, float] = {}
        watchdog_reported = False
        _LOG.info(
            "follower %s listening on %s:%d for leader %s",
            self.follower.id,
            self.follower.bind,
            self.leader.port,
            self.leader.id,
        )
        try:
            while not stopping:
                if recorder is not None and episode_request:
                    request, episode_request = episode_request, None
                    outcome = recorder.start() if request == "start" else recorder.stop()
                    print(f"RECORD_{request.upper()} {outcome}")

                try:
                    data, peer = sock.recvfrom(256)
                except TimeoutError:
                    if (
                        connected
                        and last_rx
                        and time.monotonic() - last_rx > self.follower.watchdog_s
                        and not watchdog_reported
                    ):
                        print(
                            "WATCHDOG: command stream stopped; holding last position",
                            flush=True,
                        )
                        watchdog_reported = True
                    continue
                if peer[0] != self.leader.advertise:
                    continue

                try:
                    sequence, action = decode(data)
                except ProtocolError as error:
                    _LOG.warning("discarding malformed datagram: %s", error)
                    continue

                # Drain whatever else arrived and keep the newest target, so a
                # burst is never replayed as a sequence of stale positions.
                sock.setblocking(False)
                while True:
                    try:
                        newer, newer_peer = sock.recvfrom(256)
                    except BlockingIOError:
                        break
                    if newer_peer[0] == self.leader.advertise:
                        try:
                            newer_sequence, newer_action = decode(newer)
                        except ProtocolError:
                            continue
                        if is_newer(newer_sequence, sequence):
                            sequence, action = newer_sequence, newer_action
                sock.settimeout(0.10)

                if not is_newer(sequence, last_sequence):
                    continue
                last_sequence = sequence
                last_rx = time.monotonic()
                watchdog_reported = False

                if not connected:
                    arm.connect(calibrate=False)
                    connected = True
                    initial = arm.bus.sync_read("Present_Position")
                    commanded = {f"{name}.pos": value for name, value in initial.items()}

                present = arm.bus.sync_read("Present_Position")
                if direct:
                    sent = arm.send_action(action)
                else:
                    commanded = clamp_target(
                        action,
                        present,
                        commanded,
                        max_step=self.follower.max_step,
                        max_lead=self.follower.max_lead,
                    )
                    sent = arm.send_action(commanded)
                if recorder is not None:
                    recorder.record(sequence, action, present, sent)
        finally:
            # Finish the episode before releasing torque: a motor can refuse the
            # release, and that must never cost the dataset its stop marker.
            if recorder is not None:
                try:
                    outcome = recorder.stop()
                    if outcome:
                        print(f"RECORD_STOP_ON_EXIT {outcome}")
                except Exception as error:  # noqa: BLE001 - reported, not raised
                    print(f"RECORD_STOP_ERROR {error!r}")
            if connected:
                try:
                    arm.disconnect()
                except Exception as error:  # noqa: BLE001 - reported, not raised
                    print(f"ARM_DISCONNECT_ERROR {error!r}")
            sock.close()
        return 0
