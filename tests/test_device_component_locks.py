"""Multi-identity DeviceManager ownership regressions."""

import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

from embodirun.devices import (
    DeviceBusyError,
    DeviceCloseError,
    DeviceManager,
    DeviceResource,
    DeviceUncertainError,
    ResourceIdentity,
)


def _pair_resource(opener, closer=None):
    primary = ResourceIdentity("node", "robot", "pair")
    components = (
        ResourceIdentity("node", "robot", "/dev/left"),
        ResourceIdentity("node", "robot", "/dev/right"),
    )
    return (
        DeviceResource(
            primary,
            opener,
            closer=closer,
            component_identities=components,
        ),
        primary,
        components,
    )


def test_component_conflict_is_rejected_before_adapter_opener(tmp_path: Path):
    lock_dir = tmp_path / "locks"
    first = DeviceManager("node", owner_id="first", lock_dir=lock_dir)
    opened: list[str] = []
    resource, _, components = _pair_resource(lambda: opened.append("pair") or object())
    lease = first.acquire(resource)
    second = DeviceManager("node", owner_id="second", lock_dir=lock_dir)
    with pytest.raises(DeviceBusyError):
        second.acquire(
            DeviceResource(
                components[0],
                lambda: opened.append("single") or object(),
            )
        )
    assert opened == ["pair"]
    lease.close()
    first.close()


def test_component_lock_conflict_is_cross_process(tmp_path: Path):
    lock_dir = tmp_path / "locks"
    state_path = tmp_path / "state.json"
    holder_script = """
from embodirun.devices import DeviceManager, DeviceResource, ResourceIdentity
primary = ResourceIdentity('node', 'robot', 'pair')
components = (ResourceIdentity('node', 'robot', '/dev/left'), ResourceIdentity('node', 'robot', '/dev/right'))
manager = DeviceManager('node', owner_id='holder', lock_dir=__import__('sys').argv[1], state_path=__import__('sys').argv[2])
manager.acquire(DeviceResource(primary, lambda: object(), component_identities=components))
print('ready', flush=True)
__import__('sys').stdin.read()
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script, str(lock_dir), str(state_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        ready, _, _ = select.select([holder.stdout], [], [], 5)
        assert ready and holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        probe_script = """
from embodirun.devices import DeviceBusyError, DeviceManager, DeviceResource, ResourceIdentity
identity = ResourceIdentity('node', 'robot', '/dev/left')
manager = DeviceManager('node', owner_id='probe', lock_dir=__import__('sys').argv[1], state_path=__import__('sys').argv[2])
try:
    manager.acquire(DeviceResource(identity, lambda: object()))
except DeviceBusyError:
    print('busy')
else:
    print('opened')
"""
        result = subprocess.run(
            [sys.executable, "-c", probe_script, str(lock_dir), str(state_path)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "busy"
    finally:
        assert holder.stdin is not None
        holder.stdin.write("release\n")
        holder.stdin.close()
        holder.wait(timeout=5)


def test_component_lock_failure_rolls_back_earlier_component_lock(tmp_path: Path):
    lock_dir = tmp_path / "locks"
    holder = DeviceManager("node", owner_id="holder", lock_dir=lock_dir)
    right = ResourceIdentity("node", "robot", "/dev/right")
    right_lease = holder.acquire(DeviceResource(right, lambda: object()))
    attempted = False

    def opener():
        nonlocal attempted
        attempted = True
        return object()

    manager = DeviceManager("node", owner_id="pair", lock_dir=lock_dir)
    resource, _, _ = _pair_resource(opener)
    with pytest.raises(DeviceBusyError):
        manager.acquire(resource)
    assert attempted is False

    left = ResourceIdentity("node", "robot", "/dev/left")
    probe = holder.acquire(DeviceResource(left, lambda: object()))
    probe.close()
    right_lease.close()
    holder.close()


def test_close_failure_retains_all_component_locks_and_uncertainty(tmp_path: Path):
    lock_dir = tmp_path / "locks"
    state_path = tmp_path / "state.json"
    manager = DeviceManager("node", owner_id="pair", lock_dir=lock_dir, state_path=state_path)

    def fail_close(_value):
        raise OSError("stop unknown")

    resource, primary, components = _pair_resource(lambda: object(), fail_close)
    manager.acquire(resource)
    with pytest.raises(DeviceCloseError):
        manager.close()

    other = DeviceManager("node", owner_id="other", lock_dir=lock_dir, state_path=state_path)
    for identity in (primary, *components):
        with pytest.raises(DeviceBusyError):
            other.acquire(DeviceResource(identity, lambda: object()))
        assert other.status(identity).state == "uncertain"
        with pytest.raises(DeviceUncertainError):
            other.acquire(DeviceResource(identity, lambda: object()), prepare=True)


def test_close_failure_persists_component_uncertainty_after_process_exit(
    tmp_path: Path,
):
    lock_dir = tmp_path / "locks"
    state_path = tmp_path / "state.json"
    script = """
from embodirun.devices import DeviceCloseError, DeviceManager, DeviceResource, ResourceIdentity
primary = ResourceIdentity('node', 'robot', 'pair')
components = (ResourceIdentity('node', 'robot', '/dev/left'), ResourceIdentity('node', 'robot', '/dev/right'))
def fail_close(_value):
    raise OSError('stop unknown')
manager = DeviceManager('node', owner_id='child', lock_dir=__import__('sys').argv[1], state_path=__import__('sys').argv[2])
manager.acquire(DeviceResource(primary, lambda: object(), closer=fail_close, component_identities=components))
print('ready', flush=True)
__import__('sys').stdin.read()
try:
    manager.close()
except DeviceCloseError:
    print('persisted', flush=True)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_dir), str(state_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        ready, _, _ = select.select([child.stdout], [], [], 5)
        assert ready and child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        assert child.stdin is not None
        child.stdin.write("close\n")
        child.stdin.close()
        persisted, _, _ = select.select([child.stdout], [], [], 5)
        assert persisted
        assert child.stdout.readline().strip() == "persisted"
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

    manager = DeviceManager("node", owner_id="restarted", lock_dir=lock_dir, state_path=state_path)
    identities = (
        ResourceIdentity("node", "robot", "pair"),
        ResourceIdentity("node", "robot", "/dev/left"),
        ResourceIdentity("node", "robot", "/dev/right"),
    )
    for identity in identities:
        with pytest.raises(DeviceUncertainError):
            manager.acquire(DeviceResource(identity, lambda: object()), prepare=True)
        lease = manager.acquire(DeviceResource(identity, lambda: object()))
        lease.close()
        assert manager.status(identity).state == "uncertain"
