"""Map legacy Go2 service options onto the shared checked process supervisor."""

from __future__ import annotations

import posixpath
from typing import Literal

from embodirun.deployment.executor import Command
from embodirun.deployment.plan import ServiceSpec
from embodirun.deployment.supervisor import ProcessStatus, ServiceSupervisor

from .options import (
    CameraStartOptions,
    ControlStartOptions,
    ServiceName,
    ServiceSelection,
    StartOptions,
    selected_services,
)
from .transport import RemoteTransport

SERVICE_MODULES: dict[ServiceName, str] = {
    "control": "embodirun.robots.unitree.go2.agent.control",
    "camera": "embodirun.robots.unitree.go2.agent.camera",
}


class Go2ServiceSupervisor:
    """Start, inspect, and stop Go2 services through the shared supervisor."""

    def __init__(
        self,
        transport: RemoteTransport,
        *,
        source_root: str,
        run_root: str,
        log_root: str,
    ) -> None:
        self.transport = transport
        self.source_root = source_root
        self.run_root = run_root
        self.log_root = log_root

    def start(self, options: StartOptions) -> dict[str, object]:
        results: dict[str, object] = {}
        supervisor = self._supervisor(options.python)
        for service in selected_services(options.services):
            if service == "control":
                assert options.control is not None
                command, environment = self._control_command(
                    options.python,
                    options.control,
                )
                endpoint = f"http://{options.control.bind}:{options.control.port}"
                kind: Literal["control", "sensor"] = "control"
            else:
                assert options.camera is not None
                command, environment = self._camera_command(
                    options.python,
                    options.camera,
                )
                endpoint = f"http://{options.camera.bind}:{options.camera.port}"
                kind = "sensor"
            status = supervisor.start(
                ServiceSpec(
                    service_id=service,
                    kind=kind,
                    node="go2",
                    environment_id="go2-agent",
                    endpoint=endpoint,
                    health_endpoint=f"{endpoint}/healthz",
                    command=Command(tuple(command), environment=environment),
                )
            )
            results[service] = {
                "state": status.state,
                "pid": status.pid,
                "log": status.log,
            }
        return {"services": results}

    def status(self, services: ServiceSelection = "all") -> dict[str, object]:
        supervisor = self._supervisor("python3")
        return {
            "services": {
                service: _status_payload(supervisor.status(service)) for service in selected_services(services)
            }
        }

    def stop(self, services: ServiceSelection = "all") -> dict[str, object]:
        selected = selected_services(services)
        selected.sort(key=lambda name: 0 if name == "control" else 1)
        supervisor = self._supervisor("python3")
        return {"services": {service: _status_payload(supervisor.stop(service)) for service in selected}}

    def _supervisor(self, python: str) -> ServiceSupervisor:
        return ServiceSupervisor(
            self.transport,
            python=python,
            agent_path=posixpath.join(
                self.source_root,
                "embodirun/services/host/supervisor.py",
            ),
            run_root=self.run_root,
            log_root=self.log_root,
        )

    def _control_command(
        self,
        python: str,
        options: ControlStartOptions,
    ) -> tuple[list[str], dict[str, str]]:
        command = [
            python,
            "-m",
            SERVICE_MODULES["control"],
            "--mode",
            options.mode,
            "--interface",
            options.interface,
            "--state-topic",
            options.state_topic,
            "--host",
            options.bind,
            "--port",
            str(options.port),
        ]
        if options.cyclonedds_lib_dir:
            command.extend(["--cyclonedds-lib-dir", options.cyclonedds_lib_dir])
        if options.operator_ready:
            command.append("--operator-ready")
        environment = {"PYTHONPATH": self.source_root}
        if options.api_token is not None:
            environment["GO2_API_TOKEN"] = options.api_token
        return command, environment

    def _camera_command(
        self,
        python: str,
        options: CameraStartOptions,
    ) -> tuple[list[str], dict[str, str]]:
        command = [
            python,
            "-m",
            SERVICE_MODULES["camera"],
            "--backend",
            options.backend,
            "--width",
            str(options.width),
            "--height",
            str(options.height),
            "--jpeg-fps",
            str(options.jpeg_fps),
            "--bind",
            options.bind,
            "--port",
            str(options.port),
        ]
        if options.backend == "realsense":
            assert options.depth_scale is not None
            command.extend(
                [
                    "--realsense-fps",
                    str(options.camera_fps),
                    "--depth-calibrated",
                    "--rgb-depth-aligned",
                    "--depth-scale",
                    str(options.depth_scale),
                ]
            )
            if options.realsense_serial:
                command.extend(["--realsense-serial", options.realsense_serial])
        else:
            command.extend(["--device", options.device])
        environment = {"PYTHONPATH": self.source_root}
        if options.camera_token is not None:
            environment["GO2_CAMERA_TOKEN"] = options.camera_token
        return command, environment


def _status_payload(status: ProcessStatus) -> dict[str, object]:
    payload = {"state": status.state}
    if status.pid is not None:
        payload["pid"] = status.pid
    if status.log is not None:
        payload["log"] = status.log
    return payload


__all__ = ["SERVICE_MODULES", "Go2ServiceSupervisor"]
