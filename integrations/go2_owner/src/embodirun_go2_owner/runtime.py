"""One lifecycle owner for motor control and independently scheduled capture."""

from __future__ import annotations

import threading

from .camera.capture import build_capture
from .config import OwnerConfig
from .control.executor import ActionExecutor
from .control.sdk2 import UnitreeTransport
from .control.transport import DryRunTransport


class Go2Owner:
    """Keep StopMove independent of camera capture, encoding and HTTP readers."""

    def __init__(self, config: OwnerConfig, *, transport=None, camera=None) -> None:
        self.config = config
        self._close_lock = threading.Lock()
        self._closed = False
        control = config.control
        if transport is None:
            transport = (
                UnitreeTransport(
                    control.interface,
                    control.state_topic,
                    rpc_timeout_s=control.rpc_timeout_s,
                    required_dds_lib_dir=control.cyclonedds_lib_dir,
                )
                if control.mode == "live"
                else DryRunTransport()
            )
        self.executor = ActionExecutor(transport, operator_motion_ready=control.operator_ready)
        try:
            self.camera = camera if camera is not None else (build_capture(config.camera) if config.camera else None)
        except BaseException:
            self.executor.close()
            raise

    def start(self) -> None:
        """Start camera workers; motor authority remains unchanged."""
        try:
            if self.camera is not None:
                self.camera.start()
        except BaseException:
            self.close()
            raise

    def health(self) -> dict:
        """Expose component health without conflating RGB readiness with depth."""
        control = self.executor.snapshot()
        camera = self.camera.frame_store.health(self.camera.max_frame_age) if self.camera else None
        return {
            "ok": not self._closed and control["robot_state_fresh"] and (camera is None or camera["status"] == "ok"),
            "control": control,
            "camera": camera,
            "capabilities": {
                "control": True,
                "rgb": self.camera is not None,
                "rgbd": self.camera is not None and self.camera.depth_store is not None,
            },
        }

    def close(self) -> None:
        """Revoke motion and stop the SDK before waiting for camera shutdown."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.executor.close()
            finally:
                if self.camera is not None:
                    self.camera.close()
