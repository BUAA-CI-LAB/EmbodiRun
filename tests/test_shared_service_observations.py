from __future__ import annotations

from dataclasses import dataclass

import pytest

from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.contracts import (
    ControlRuntimeProfile,
    ControlServiceConfig,
)
from embodirun.services.control.devices import DeviceManager, DeviceOpenError
from embodirun.services.control.server import ControlService


@dataclass
class _Source:
    name: str
    data: bytes
    failure: Exception | None = None
    opens: list[str] | None = None

    def capture(self) -> tuple[CameraFrame, ...]:
        if self.failure is not None:
            raise self.failure
        return (
            CameraFrame(
                "front",
                "image/jpeg",
                self.data,
                captured_timestamp_ns=100,
                received_timestamp_ns=101,
                clock_domain="host_monotonic_ns",
            ),
        )

    def close(self) -> None:
        return None


def _config(
    primary: SensorInput,
    profile: ControlRuntimeProfile,
) -> ControlServiceConfig:
    resources = tuple(
        {
            "identity": f"node:sensor:{item.options['device']}",
            "node": "node",
            "kind": "sensor",
            "value": item.options["device"],
        }
        for item in (primary, profile.inputs[0])
    )
    return ControlServiceConfig(
        runtime_id="runtime-a",
        binding_kind="test.binding",
        bind="127.0.0.1",
        port=18120,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
        inputs=(primary,),
        runtime_options={},
        node_id="node",
        device_resources=resources,
        runtime_profiles={profile.runtime_id: profile},
    )


def _profile(sensor: SensorInput, runtime_id: str = "runtime-b") -> ControlRuntimeProfile:
    return ControlRuntimeProfile(
        runtime_id=runtime_id,
        binding_kind="test.binding",
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        inputs=(sensor,),
        runtime_options={},
    )


def test_runtime_views_share_one_source_bytes_and_snapshot_lookup(tmp_path) -> None:
    primary = SensorInput("camera-a", "front", "v4l2", {"device": "/dev/a"})
    alias = SensorInput("camera-b", "front", "v4l2", {"device": "/dev/b"})
    config = _config(primary, _profile(alias))
    opened: list[str] = []

    def factory(items):
        device = items[0].options["device"]
        opened.append(device)
        return _Source(device, device.encode())

    service = ControlService(
        config,
        camera_factory=factory,
        device_manager=DeviceManager("node", lock_dir=tmp_path / "locks"),
    )
    try:
        first = service.observe(runtime_id="runtime-a")
        second = service.observe(runtime_id="runtime-b")
        assert opened == ["/dev/a", "/dev/b"]
        assert first["frames"][0]["name"] == "front"
        assert second["frames"][0]["name"] == "front"
        assert first["frames"][0]["data"] != second["frames"][0]["data"]
        restored = service.get_snapshot(second["snapshot_id"], runtime_id="runtime-b")
        assert restored["frames"] == second["frames"]
    finally:
        service.close()


def test_bad_source_does_not_make_healthy_runtime_view_unavailable(tmp_path) -> None:
    bad = SensorInput("camera-bad", "front", "v4l2", {"device": "/dev/bad"})
    good = SensorInput("camera-good", "front", "v4l2", {"device": "/dev/good"})
    config = _config(bad, _profile(good))

    def factory(items):
        device = items[0].options["device"]
        return _Source(
            device,
            b"good",
            failure=RuntimeError("disconnected") if device == "/dev/bad" else None,
        )

    service = ControlService(
        config,
        camera_factory=factory,
        device_manager=DeviceManager("node", lock_dir=tmp_path / "locks"),
    )
    try:
        failed = service.observe(runtime_id="runtime-a")
        restored = service.get_snapshot(
            failed["snapshot_id"],
            runtime_id="runtime-a",
        )
        assert restored["frames"] == []
        assert restored["observation_status"]["available"] is False
        healthy = service.observe(runtime_id="runtime-b")
        assert healthy["observation_status"]["available"] is True
        assert healthy["observation_status"]["global"]["errors"]
        assert healthy["frames"][0]["data"]
    finally:
        service.close()


def test_service_instance_ids_do_not_alias_across_restarts(tmp_path) -> None:
    primary = SensorInput("camera", "front", "v4l2", {"device": "/dev/a"})
    profile = _profile(SensorInput("camera-b", "front-b", "v4l2", {"device": "/dev/b"}))
    config = _config(primary, profile)

    def factory(_items):
        return _Source("camera", b"frame")

    first_service = ControlService(
        config,
        camera_factory=factory,
        device_manager=DeviceManager("node", lock_dir=tmp_path / "first"),
    )
    second_service = ControlService(
        config,
        camera_factory=factory,
        device_manager=DeviceManager("node", lock_dir=tmp_path / "second"),
    )
    try:
        first_id = first_service.observe()["snapshot_id"]
        second_id = second_service.observe()["snapshot_id"]
        assert first_id != second_id
    finally:
        first_service.close()
        second_service.close()


def test_failed_second_camera_keeps_first_owner_for_healthy_profile(tmp_path) -> None:
    first = SensorInput("camera-a", "front", "v4l2", {"device": "/dev/a"})
    second = SensorInput("camera-b", "wrist", "v4l2", {"device": "/dev/bad"})
    healthy_profile = _profile(SensorInput("camera-a", "front", "v4l2", {"device": "/dev/a"}))
    config = ControlServiceConfig(
        runtime_id="runtime-a",
        binding_kind="test.binding",
        bind="127.0.0.1",
        port=18121,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
        inputs=(first, second),
        runtime_options={},
        node_id="node",
        device_resources=tuple(
            {
                "identity": f"node:sensor:{item.options['device']}",
                "node": "node",
                "kind": "sensor",
                "value": item.options["device"],
            }
            for item in (first, second)
        ),
        runtime_profiles={healthy_profile.runtime_id: healthy_profile},
    )
    closed: list[str] = []

    class ClosingSource(_Source):
        def close(self) -> None:
            closed.append(self.name)

    def factory(items):
        device = items[0].options["device"]
        if device == "/dev/bad":
            raise OSError("camera open failed")
        return ClosingSource(device, b"healthy")

    service = ControlService(
        config,
        camera_factory=factory,
        device_manager=DeviceManager("node", lock_dir=tmp_path / "locks"),
    )
    try:
        with pytest.raises(DeviceOpenError):
            service.observe(runtime_id="runtime-a")
        healthy = service.observe(runtime_id="runtime-b")
        assert healthy["frames"][0]["name"] == "front"
        assert closed == []
    finally:
        service.close()
    assert closed == ["/dev/a"]
