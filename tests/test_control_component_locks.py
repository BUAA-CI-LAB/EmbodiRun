"""Public ControlService coverage for multi-bus robot ownership."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from embodirun.application.contracts import ControlServiceConfig
from embodirun.application.control_service import ControlService
from embodirun.devices import (
    DeviceBusyError,
    DeviceManager,
    DeviceResource,
    ResourceIdentity,
)
from embodirun.robots import RobotAdapter, RobotObservation


class _FakeBiAdapter(RobotAdapter):
    def __init__(self, config):
        self.robot_id = config.robot_id
        self.events: list[str] = []
        self.prepared = False

    def connect(self, *, prepare=True):
        self.events.append(f"connect:{prepare}")
        self.prepared = prepare

    def prepare(self):
        self.events.append("prepare")
        self.prepared = True

    def observe(self):
        return RobotObservation(1.0, {"state": 1})

    def execute(self, action):
        del action

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


def _config(resources):
    return ControlServiceConfig(
        runtime_id="pair-runtime",
        binding_kind="lerobot.bi_so101.pi05",
        bind="127.0.0.1",
        port=8100,
        inference_transport="disabled",
        inference_endpoint="disabled://inference",
        inference_options={},
        robot_id="pair",
        robot_kind="lerobot.bi_so101",
        robot_options={
            "left_port": "/dev/left",
            "right_port": "/dev/right",
        },
        inputs=(),
        runtime_options={},
        inference_enabled=False,
        device_resources=tuple(resources),
    )


def test_public_control_service_reserves_pair_before_single_alias(monkeypatch, tmp_path: Path):
    primary = "node:robot:pair|ports=/dev/left|/dev/right"
    resources = (
        {
            "identity": primary,
            "node": "node",
            "kind": "robot",
            "value": "pair|ports=/dev/left|/dev/right",
        },
        {
            "identity": "node:robot:/dev/left",
            "node": "node",
            "kind": "robot",
            "value": "/dev/left",
        },
        {
            "identity": "node:robot:/dev/right",
            "node": "node",
            "kind": "robot",
            "value": "/dev/right",
        },
    )
    binding = SimpleNamespace(kind="lerobot.bi_so101.pi05", mapper_factory=lambda: None)
    definition = SimpleNamespace(
        config_factory=lambda robot_id, options: SimpleNamespace(robot_id=robot_id),
        adapter_type=_FakeBiAdapter,
    )
    monkeypatch.setattr(
        "embodirun.application.control_service._definitions",
        lambda _config: (binding, definition),
    )
    manager = DeviceManager("node", owner_id="control", lock_dir=tmp_path / "locks")
    service = ControlService(_config(resources), device_manager=manager)

    try:
        payload = service.observe(include_robot=True)
        assert payload["robot"]["values"] == {"state": 1}
        alias_manager = DeviceManager("node", owner_id="single", lock_dir=tmp_path / "locks")
        with pytest.raises(DeviceBusyError):
            alias_manager.acquire(
                DeviceResource(
                    ResourceIdentity("node", "robot", "/dev/left"),
                    lambda: object(),
                )
            )
    finally:
        service.close()
