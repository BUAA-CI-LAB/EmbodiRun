"""Deployment resource checks for the dual SO-101 definition."""

from types import SimpleNamespace

import pytest

from embodirun.deployment.config.loader import _validate_unique_robot_ports
from embodirun.deployment.config.robot import parse_robot
from embodirun.deployment.config.validation import ConfigError
from embodirun.deployment.executor import CommandResult
from embodirun.deployment.operations.init import _probe_resources
from embodirun.deployment.plan import _control_resources
from embodirun.deployment.state import NodeState


def _robot(robot_id: str, kind: str, **options):
    return parse_robot(robot_id, {"type": kind, "node": "local", **options})


def test_dual_ports_cannot_hide_single_arm_overlap_behind_resource_alias():
    pair = _robot(
        "pair",
        "lerobot.bi_so101",
        resource="shared",
        left_port="/dev/left",
        right_port="/dev/right",
    )
    single = _robot(
        "single",
        "lerobot.so101",
        resource="shared",
        port="/dev/left",
    )

    with pytest.raises(ConfigError, match="share port '/dev/left'"):
        _validate_unique_robot_ports({"pair": pair, "single": single})


def test_same_resource_alias_requires_same_complete_dual_adapter():
    first = _robot(
        "first",
        "lerobot.bi_so101",
        resource="shared",
        left_port="/dev/left",
        right_port="/dev/right",
    )
    alias = _robot(
        "alias",
        "lerobot.bi_so101",
        resource="shared",
        left_port="/dev/left",
        right_port="/dev/right",
    )
    _validate_unique_robot_ports({"first": first, "alias": alias})


def test_probe_resources_checks_both_buses_and_calibration_files():
    robot = _robot(
        "pair",
        "lerobot.bi_so101",
        left_port="/dev/left",
        right_port="/dev/right",
        calibration_dir="calibration",
        left_calibration_id="left",
        right_calibration_id="right",
    )
    context = SimpleNamespace(
        config=SimpleNamespace(robots={"pair": robot}, models={}),
    )
    node = NodeState(
        node_id="local",
        home="/home/test",
        root="/home/test/.embodirun",
        deploy_project="/srv/deploy",
        inference_project="/srv/inference",
        platform="linux",
        machine="test",
        python="python3",
        python_version="3.12",
    )

    class Executor:
        def __init__(self):
            self.commands = []

        def run(self, command, *, check=True):
            self.commands.append(command.argv)
            return CommandResult(0)

    executor = Executor()
    _probe_resources(context, executor, node)

    assert executor.commands == [
        ("test", "-e", "/dev/left"),
        ("test", "-e", "/dev/right"),
        ("test", "-d", "/srv/deploy/calibration"),
        ("test", "-f", "/srv/deploy/calibration/left.json"),
        ("test", "-f", "/srv/deploy/calibration/right.json"),
    ]


def test_plan_lists_dual_adapter_and_each_physical_bus_resource():
    robot = _robot(
        "pair",
        "lerobot.bi_so101",
        resource="pair",
        left_port="/dev/left",
        right_port="/dev/right",
    )
    runtime = SimpleNamespace(robot="pair", inputs={})
    config = SimpleNamespace(robots={"pair": robot}, sensors={})

    resources = _control_resources(config, runtime)
    identities = {item["identity"] for item in resources}
    assert "local:robot:pair|ports=/dev/left|/dev/right" in identities
    assert "local:robot:/dev/left" in identities
    assert "local:robot:/dev/right" in identities


def test_plan_keeps_single_arm_port_lock_when_resource_is_an_alias():
    robot = _robot(
        "single",
        "lerobot.so101",
        resource="logical-single",
        port="/dev/left",
    )
    resources = _control_resources(
        SimpleNamespace(robots={"single": robot}, sensors={}),
        SimpleNamespace(robot="single", inputs={}),
    )

    assert {item["identity"] for item in resources} == {
        "local:robot:logical-single",
        "local:robot:/dev/left",
    }
