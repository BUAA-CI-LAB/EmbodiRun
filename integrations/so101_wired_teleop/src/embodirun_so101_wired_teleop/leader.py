"""Read one leader arm and broadcast its targets to the wired followers.

The leader never writes to its own motors: it is a read-only sensor. Everything
that can move lives on the follower side, so a fault here cannot drive an arm.
"""

from __future__ import annotations

import logging
import signal
import socket
import time
from pathlib import Path
from typing import Any

from .config import LeaderConfig, TeleopConfig
from .protocol import JOINTS, encode

_LOG = logging.getLogger(__name__)


def _leader_sdk() -> tuple[Any, Any]:
    """Import the LeRobot leader stack only when hardware is actually used."""

    try:
        from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "the leader node needs the 'hardware' extra: pip install 'embodirun-so101-wired-teleop[hardware]'"
        ) from error
    return SO101LeaderConfig, SOLeader


class LeaderBroadcaster:
    """Broadcast one leader arm's targets to every wired follower that obeys it."""

    def __init__(self, config: TeleopConfig, leader: LeaderConfig) -> None:
        self.config = config
        self.leader = leader
        self.destinations = config.destinations(leader.id)

    def calibration_path(self) -> Path:
        return Path(self.leader.calibration_dir) / f"{self.leader.arm_id}.json"

    def run(self, *, stop_after: float | None = None) -> int:
        """Publish until interrupted, or for ``stop_after`` seconds."""

        sdk_config, sdk_leader = _leader_sdk()
        calibration = self.calibration_path()
        if not calibration.is_file():
            raise SystemExit(f"missing leader calibration: {calibration}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Bind the source address explicitly: followers filter datagrams by
            # source IP, and a multi-homed host would otherwise pick whatever the
            # routing table prefers and be silently ignored.
            sock.bind((self.leader.advertise, 0))
            arm = sdk_leader(
                sdk_config(
                    port=self.leader.serial,
                    id=self.leader.arm_id,
                    calibration_dir=Path(self.leader.calibration_dir),
                )
            )
            stopping = False

            def request_stop(_signum: int, _frame: Any) -> None:
                nonlocal stopping
                stopping = True

            signal.signal(signal.SIGINT, request_stop)
            signal.signal(signal.SIGTERM, request_stop)

            arm.connect(calibrate=False)
            _LOG.info(
                "leader %s publishing to %s on port %d at %g fps",
                self.leader.id,
                list(self.destinations) or "(no followers)",
                self.leader.port,
                self.leader.fps,
            )
            try:
                self._publish(sock, arm, lambda: stopping, stop_after)
            finally:
                arm.disconnect()
        finally:
            sock.close()
        return 0

    def _publish(
        self,
        sock: socket.socket,
        arm: Any,
        stopping: Any,
        stop_after: float | None,
    ) -> None:
        period = 1.0 / self.leader.fps
        deadline = None if stop_after is None else time.monotonic() + stop_after
        sequence = 0
        while not stopping():
            if deadline is not None and time.monotonic() >= deadline:
                return
            started = time.monotonic()
            packet = encode(sequence, arm.get_action())
            for destination in self.destinations:
                sock.sendto(packet, (destination, self.leader.port))
            sequence = (sequence + 1) & 0xFFFFFFFF
            delay = period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    def network_test(self, *, packets: int = 20) -> int:
        """Send neutral packets so followers can prove the path without an arm."""

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.leader.advertise, 0))
            neutral = dict.fromkeys(JOINTS, 0.0)
            for sequence in range(packets):
                packet = encode(sequence, neutral, probe=True)
                for destination in self.destinations:
                    sock.sendto(packet, (destination, self.leader.port))
                time.sleep(0.03)
        finally:
            sock.close()
        print(f"NETWORK_TEST_SENT leader={self.leader.id} targets={list(self.destinations)}")
        return 0
