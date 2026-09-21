"""Heartbeat-managed velocity lease independent of a robot SDK."""

from __future__ import annotations

from ..interfaces import MobileBase, MobileBaseError
from ..motion import PlanarVelocityCommand


class VelocityLease:
    def __init__(self, base: MobileBase, *, duration_s: float = 10.0) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.base = base
        self.duration_s = float(duration_s)
        self.lease_id: str | None = None
        self.last_command = PlanarVelocityCommand.stopped()

    def _start(self, command: PlanarVelocityCommand) -> None:
        lease_id = self.base.start_velocity_lease(command, duration_s=self.duration_s)
        if not isinstance(lease_id, str) or not lease_id:
            raise MobileBaseError("mobile base did not return a velocity lease id")
        self.lease_id = lease_id

    def send(self, command: PlanarVelocityCommand) -> None:
        if not isinstance(command, PlanarVelocityCommand):
            raise TypeError("command must be a PlanarVelocityCommand")
        if self.lease_id is None:
            if command.moving:
                self._start(command)
            self.last_command = command
            return
        try:
            self.base.update_velocity_lease(self.lease_id, command)
        except MobileBaseError:
            state = self.base.state()
            if state.active_velocity_lease_id == self.lease_id:
                raise
            self.lease_id = None
            if command.moving:
                self._start(command)
        self.last_command = command

    def stop(self) -> None:
        self.base.stop()
        self.lease_id = None
        self.last_command = PlanarVelocityCommand.stopped(self.last_command.limits)


__all__ = ["VelocityLease"]
