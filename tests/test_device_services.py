import os
import subprocess
import sys
import time
from threading import Barrier, Event, Lock, Thread, current_thread

import pytest

from embodirun.application import control_service as control_service_impl
from embodirun.robots import RobotAdapter, RobotDefinition, RobotObservation
from embodirun.robots.lerobot.so101 import (
    SO101Adapter,
    SO101AdapterError,
    SO101Config,
)
from embodirun.robots.sensors import SensorInput
from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.contracts import (
    ControlRuntimeProfile,
    ControlServiceConfig,
    TaskRequest,
)
from embodirun.services.control.devices import (
    DeviceBusyError,
    DeviceCloseError,
    DeviceManager,
    DeviceOpenError,
    DeviceResource,
    DeviceStateError,
    DeviceUncertainError,
    ResourceIdentity,
    canonical_resource_identity,
)
from embodirun.services.control.server import (
    ControlService,
    ControlServiceError,
    ControlTaskRejected,
)


def _resource(tmp_path, name="serial0"):
    return ResourceIdentity("node-a", "robot", str(tmp_path / name))


def test_manager_reuses_one_open_connection_until_last_lease(tmp_path):
    calls = []
    resource = _resource(tmp_path)
    manager = DeviceManager(
        "node-a",
        owner_id="test",
        lock_dir=tmp_path / "locks",
        state_path=tmp_path / "state.json",
    )
    managed = DeviceResource(
        resource,
        lambda: calls.append("open") or object(),
        closer=lambda _value: calls.append("close"),
    )

    first = manager.acquire(managed)
    second = manager.acquire(managed)
    assert first.value is second.value
    assert calls == ["open"]
    first.close()
    assert calls == ["open"]
    second.close()
    assert calls == ["open", "close"]


def test_acquire_reports_busy_while_close_callback_is_still_running(tmp_path):
    identity = _resource(tmp_path, "blocked-close")
    started = Event()
    release = Event()
    close_errors = []

    def closer(_value):
        started.set()
        release.wait(5.0)

    manager = DeviceManager(
        "node-a",
        lock_dir=tmp_path / "locks",
        state_path=tmp_path / "state.json",
    )
    lease = manager.acquire(DeviceResource(identity, lambda: object(), closer=closer))

    def release_lease():
        try:
            lease.close()
        except BaseException as error:  # pragma: no cover - diagnostic guard
            close_errors.append(error)

    thread = Thread(target=release_lease)
    thread.start()
    assert started.wait(1.0)
    try:
        with pytest.raises(DeviceBusyError, match="closing"):
            manager.acquire(DeviceResource(identity, lambda: object()))
    finally:
        release.set()
        thread.join(timeout=2.0)
        manager.close()
    assert not close_errors


def test_aliases_share_canonical_identity_and_process_lock(tmp_path):
    target = tmp_path / "device"
    target.write_text("", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    resolved = canonical_resource_identity("node-a", "sensor", alias)
    direct = canonical_resource_identity("node-a", "sensor", target)
    assert resolved == direct

    manager = DeviceManager("node-a", lock_dir=tmp_path / "locks")
    lease = manager.acquire(DeviceResource(resolved, lambda: object(), closer=lambda _value: None))
    with pytest.raises(DeviceBusyError):
        DeviceManager("node-a", lock_dir=tmp_path / "locks").acquire(
            DeviceResource(direct, lambda: object(), closer=lambda _value: None)
        )
    lease.close()
    manager.close()


def test_default_lifecycle_state_uses_durable_xdg_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("EMBODIRUN_DEVICE_LOCK_DIR", str(tmp_path / "locks"))
    manager = DeviceManager("node-a")
    assert manager.state_store.path == (tmp_path / "xdg-state" / "embodirun" / "device-state.json")


def test_process_lock_releases_after_owner_crash(tmp_path):
    lock_dir = tmp_path / "locks"
    state_path = tmp_path / "state.json"
    identity = _resource(tmp_path)
    script = """
import sys
import time
from embodirun.services.control.devices import DeviceManager, DeviceResource, ResourceIdentity
identity = ResourceIdentity(sys.argv[1], 'robot', sys.argv[2])
manager = DeviceManager(identity.node, owner_id='child', lock_dir=sys.argv[3], state_path=sys.argv[4])
manager.acquire(DeviceResource(identity, lambda: object(), closer=lambda value: None))
print('ready', flush=True)
time.sleep(30)
"""
    environment = dict(os.environ, PYTHONPATH="src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            identity.node,
            identity.value,
            str(lock_dir),
            str(state_path),
        ],
        cwd=os.getcwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        manager = DeviceManager("node-a", lock_dir=lock_dir, state_path=state_path)
        with pytest.raises(DeviceBusyError):
            manager.acquire(DeviceResource(identity, lambda: object()))
        process.kill()
        process.wait(timeout=5)
        lease = manager.acquire(DeviceResource(identity, lambda: object()))
        lease.close()
        manager.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_prepare_fails_closed_for_corrupt_state(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("[]", encoding="utf-8")
    manager = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)
    identity = _resource(tmp_path)
    with pytest.raises(DeviceUncertainError, match="malformed"):
        manager.acquire(DeviceResource(identity, lambda: object()), prepare=True)
    assert manager.status(identity).state == "closed"
    assert manager.state_store.error is not None


def test_close_failure_survives_read_only_reopen(tmp_path):
    state = tmp_path / "state.json"
    identity = _resource(tmp_path)
    first = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)

    def fail_close(_value):
        raise RuntimeError("stop unknown")

    failed = first.acquire(DeviceResource(identity, lambda: object(), closer=fail_close))
    with pytest.raises(RuntimeError, match="stop unknown"):
        failed.close()
    second = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)
    with pytest.raises(DeviceBusyError):
        second.acquire(DeviceResource(identity, lambda: object()))

    # A separate resource demonstrates the persisted-fault behavior once the
    # original owner really did release its OS lock.
    recovered_identity = _resource(tmp_path, "serial1")
    first_recovered = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)
    recovered = first_recovered.acquire(
        DeviceResource(recovered_identity, lambda: object(), closer=lambda _value: None)
    )
    first_recovered.mark_uncertain(recovered_identity, "stop unknown")
    recovered.close()
    second = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)
    passive = second.acquire(DeviceResource(recovered_identity, lambda: object(), closer=lambda _value: None))
    assert second.status(recovered_identity).state == "uncertain"
    passive.close()
    assert second.status(recovered_identity).state == "uncertain"
    with pytest.raises(DeviceUncertainError):
        second.acquire(DeviceResource(recovered_identity, lambda: object()), prepare=True)
    second.clear_uncertainty(recovered_identity)
    assert second.status(recovered_identity).state == "closed"


def test_manager_close_retries_quarantined_handle_after_sdk_release(tmp_path):
    identity = _resource(tmp_path, "retry-close")
    released = Event()

    def closer(_value):
        if not released.wait(0.05):
            raise RuntimeError("SDK still in flight")

    manager = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=tmp_path / "state.json")
    manager.acquire(DeviceResource(identity, lambda: object(), closer=closer))
    with pytest.raises(DeviceCloseError, match="SDK still in flight"):
        manager.close()
    other = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=tmp_path / "state.json")
    with pytest.raises(DeviceBusyError):
        other.acquire(DeviceResource(identity, lambda: object()))
    released.set()
    manager.close()
    lease = other.acquire(DeviceResource(identity, lambda: object()))
    lease.close()
    other.close()


def test_prepare_failure_with_failed_rollback_retains_live_lock(tmp_path):
    identity = _resource(tmp_path, "prepare-rollback")
    close_calls = []

    def closer(_value):
        close_calls.append(len(close_calls))
        if len(close_calls) == 1:
            raise RuntimeError("cleanup still owns SDK")

    manager = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=tmp_path / "state.json")
    with pytest.raises(DeviceOpenError, match="prepare failed"):
        manager.acquire(
            DeviceResource(
                identity,
                lambda: object(),
                closer=closer,
                preparer=lambda _value: (_ for _ in ()).throw(RuntimeError("prepare failed")),
            ),
            prepare=True,
        )
    assert manager.status(identity).state == "uncertain"
    other = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=tmp_path / "state.json")
    with pytest.raises(DeviceBusyError):
        other.acquire(DeviceResource(identity, lambda: object()))
    manager.close()
    lease = other.acquire(DeviceResource(identity, lambda: object()))
    lease.close()
    other.close()


def test_corrupt_state_is_not_overwritten_by_new_fault(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("[]", encoding="utf-8")
    identity = _resource(tmp_path, "corrupt-state")
    manager = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)
    with pytest.raises(DeviceStateError, match="malformed"):
        manager.mark_uncertain(identity, "new fault")
    assert state.read_text(encoding="utf-8") == "[]"


def test_unresolved_state_follows_physical_identity_across_node_alias(tmp_path):
    state = tmp_path / "state.json"
    first_identity = ResourceIdentity("node-a", "robot", str(tmp_path / "arm"))
    second_identity = ResourceIdentity("node-b", "robot", str(tmp_path / "arm"))
    first = DeviceManager("node-a", lock_dir=tmp_path / "locks", state_path=state)
    first.mark_uncertain(first_identity, "stop unknown")
    second = DeviceManager("node-b", lock_dir=tmp_path / "locks", state_path=state)
    assert second.status(second_identity).state == "uncertain"
    with pytest.raises(DeviceUncertainError):
        second.acquire(DeviceResource(second_identity, lambda: object()), prepare=True)


class _PassiveSO101:
    is_connected = False
    is_calibrated = True

    def __init__(self):
        self.calls = []

    def connect(self, *, calibrate, prepare):
        self.calls.append(("connect", prepare))
        self.is_connected = True

    def prepare(self):
        self.calls.append("prepare")

    def get_observation(self):
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,
        }

    def send_action(self, action):
        self.calls.append(("action", action))

    def disconnect(self):
        self.calls.append("disconnect")
        self.is_connected = False


def test_so101_passive_connect_and_close_do_not_prepare_or_write():
    controller = _PassiveSO101()
    adapter = SO101Adapter(SO101Config(port="/dev/fake"), controller=controller)
    adapter.connect(prepare=False)
    adapter.observe()
    adapter.close()
    assert controller.calls == [("connect", False), "disconnect"]
    with pytest.raises(SO101AdapterError, match="not prepared"):
        adapter.execute(None)  # type: ignore[arg-type]


def test_so101_rejects_legacy_controller_for_passive_connect():
    class Legacy:
        is_connected = False
        is_calibrated = True

        def connect(self, *, calibrate):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

    adapter = SO101Adapter(SO101Config(port="/dev/fake"), controller=Legacy())
    with pytest.raises(SO101AdapterError, match="passive"):
        adapter.connect(prepare=False)


def test_device_only_config_describes_and_observes_without_inference(monkeypatch):
    class FakeAdapter(RobotAdapter):
        def __init__(self, config, *, read_only=False):
            self.calls = []
            self.read_only = read_only
            self.robot_id = "arm"

        def connect(self, *, prepare=True):
            self.calls.append(("connect", prepare))

        def prepare(self):
            self.calls.append("prepare")

        def observe(self):
            return RobotObservation(1.0, {"state": 1})

        def execute(self, action):
            raise AssertionError("device-only observe must not execute")

        def stop(self):
            raise AssertionError("device-only observe must not stop")

        def close(self):
            self.calls.append("close")

    definition = RobotDefinition(
        kind="test.robot",
        config_factory=lambda robot_id, options: object(),
        adapter_type=FakeAdapter,
        environment_group="test",
    )
    monkeypatch.setattr(control_service_impl, "robot_definition", lambda _kind: definition)
    config = ControlServiceConfig.device_only(
        runtime_id="device-runtime",
        bind="127.0.0.1",
        port=18101,
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
    )
    encoded = config.to_json()
    assert '"enabled":false' in encoded
    assert '"endpoint"' not in encoded
    parsed = ControlServiceConfig.from_json(encoded)
    service = ControlService(
        parsed,
        client_factory=lambda *_args: (_ for _ in ()).throw(AssertionError("inference called")),
    )
    assert service.health()["inference"] == "disabled"
    result = service.observe(include_robot=True)
    assert result["robot"] == {
        "timestamp_s": 1.0,
        "values": {"state": 1},
        "metadata": {},
    }
    with pytest.raises(ControlTaskRejected, match="no inference"):
        service.execute(TaskRequest("request", "device-runtime", "move", 1, 1, 1.0, 1.0))
    service.close()


def test_shared_camera_runtime_alias_reuses_bytes_and_open(tmp_path):
    opened = []

    class Source:
        def __init__(self, name):
            self.name = name

        def capture(self):
            return (CameraFrame(self.name, "image/jpeg", b"same-frame"),)

        def close(self):
            opened.append("close")

    camera_input = SensorInput("camera", "front-a", "v4l2", {"device": "/dev/camera0"})
    alias_input = SensorInput("camera", "front-b", "v4l2", {"device": "/dev/camera0"})
    profile = ControlRuntimeProfile(
        runtime_id="runtime-b",
        binding_kind="test.binding",
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        inputs=(alias_input,),
        runtime_options={},
    )
    config = ControlServiceConfig(
        runtime_id="runtime-a",
        binding_kind="test.binding",
        bind="127.0.0.1",
        port=18102,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
        inputs=(camera_input,),
        runtime_options={},
        node_id="node-a",
        device_resources=(
            {
                "identity": "node-a:sensor:/dev/camera0",
                "node": "node-a",
                "kind": "sensor",
                "value": "/dev/camera0",
            },
        ),
        runtime_profiles={"runtime-b": profile},
    )
    manager = DeviceManager("node-a", lock_dir=tmp_path / "locks")

    def factory(items):
        opened.append(items[0].name)
        return Source(items[0].name)

    service = ControlService(config, camera_factory=factory, device_manager=manager)
    first = service.observe(runtime_id="runtime-a")
    second = service.observe(runtime_id="runtime-b")
    assert first["frames"][0]["name"] == "front-a"
    assert second["frames"][0]["name"] == "front-b"
    assert first["frames"][0]["data"] == second["frames"][0]["data"]
    assert opened == ["front-a"]
    service.release_observer("runtime-a")
    service.release_observer("runtime-b")
    service.close()
    assert opened == ["front-a", "close"]


def test_shared_camera_profile_conflict_is_rejected_before_reopen(tmp_path):
    first_input = SensorInput("camera", "front-a", "v4l2", {"device": "/dev/camera-profile", "width": 640})
    second_input = SensorInput("camera", "front-b", "v4l2", {"device": "/dev/camera-profile", "width": 320})
    profile = ControlRuntimeProfile(
        runtime_id="runtime-b",
        binding_kind="test.binding",
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        inputs=(second_input,),
        runtime_options={},
    )
    config = ControlServiceConfig(
        runtime_id="runtime-a",
        binding_kind="test.binding",
        bind="127.0.0.1",
        port=18104,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
        inputs=(first_input,),
        runtime_options={},
        node_id="node-a",
        device_resources=(
            {
                "identity": "node-a:sensor:/dev/camera-profile",
                "node": "node-a",
                "kind": "sensor",
                "value": "/dev/camera-profile",
            },
        ),
        runtime_profiles={"runtime-b": profile},
    )
    opened = []

    class Source:
        def capture(self):
            return (CameraFrame("front-a", "image/jpeg", b"frame"),)

        def close(self):
            opened.append("close")

    service = ControlService(
        config,
        camera_factory=lambda _items: opened.append("open") or Source(),
        device_manager=DeviceManager("node-a", lock_dir=tmp_path / "locks"),
    )
    service.observe(runtime_id="runtime-a")
    with pytest.raises(ControlServiceError, match="incompatible capture profiles"):
        service.observe(runtime_id="runtime-b")
    service.close()
    assert opened == ["open", "close"]


def test_shared_camera_alias_rejects_unmapped_multi_frame_source(tmp_path):
    first_input = SensorInput("camera", "rgb", "realsense", {"serial": "serial-1"})
    alias_input = SensorInput("camera", "runtime-role", "realsense", {"serial": "serial-1"})
    profile = ControlRuntimeProfile(
        runtime_id="runtime-b",
        binding_kind="test.binding",
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        inputs=(alias_input,),
        runtime_options={},
    )
    config = ControlServiceConfig(
        runtime_id="runtime-a",
        binding_kind="test.binding",
        bind="127.0.0.1",
        port=18105,
        inference_transport="http",
        inference_endpoint="http://127.0.0.1:1",
        inference_options={},
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
        inputs=(first_input,),
        runtime_options={},
        node_id="node-a",
        device_resources=(
            {
                "identity": "node-a:sensor:serial-1",
                "node": "node-a",
                "kind": "sensor",
                "value": "serial-1",
            },
        ),
        runtime_profiles={"runtime-b": profile},
    )

    class Source:
        def capture(self):
            return (
                CameraFrame("rgb", "image/jpeg", b"rgb"),
                CameraFrame("depth", "image/png", b"depth"),
            )

        def close(self):
            return None

    service = ControlService(
        config,
        camera_factory=lambda _items: Source(),
        device_manager=DeviceManager("node-a", lock_dir=tmp_path / "locks"),
    )
    first = service.observe(runtime_id="runtime-a")
    assert [frame["name"] for frame in first["frames"]] == ["rgb", "depth"]
    with pytest.raises(ControlServiceError, match="multi-frame camera sources"):
        service.observe(runtime_id="runtime-b")
    service.close()


def test_configured_control_service_uses_default_manager_and_reuses_robot_owner(monkeypatch, tmp_path):
    events = []

    class FakeAdapter(RobotAdapter):
        def __init__(self, config, *, read_only=False):
            self.read_only = read_only
            self.robot_id = "arm"
            events.append(("construct", read_only))

        def connect(self, *, prepare=True):
            events.append(("connect", prepare))

        def observe(self):
            events.append("observe")
            return RobotObservation(1.0, {"state": 1})

        def execute(self, action):
            events.append("execute")

        def stop(self):
            events.append("stop")

        def close(self):
            events.append("close")

        def prepare(self):
            events.append("prepare")

    definition = RobotDefinition(
        kind="test.robot",
        config_factory=lambda robot_id, options: object(),
        adapter_type=FakeAdapter,
        environment_group="test",
    )
    binding = type(
        "Binding",
        (),
        {
            "robot_kind": "test.robot",
            "maximum_chunk_steps": 1,
            "mapper_factory": lambda self: object(),
        },
    )()
    monkeypatch.setattr(control_service_impl, "_definitions", lambda _config: (binding, definition))
    resource = {
        "identity": f"node-a:robot:{tmp_path / 'arm'}",
        "node": "node-a",
        "kind": "robot",
        "value": str(tmp_path / "arm"),
    }
    config = ControlServiceConfig.device_only(
        runtime_id="device-runtime",
        bind="127.0.0.1",
        port=18103,
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
        node_id="node-a",
        device_resources=(resource,),
    )
    service = ControlService(config)
    assert service.device_manager is not None
    service._ensure_robot_arbiter(definition)
    result = service.observe(include_robot=True)
    assert result["robot"]["values"] == {"state": 1}
    assert events.count(("construct", False)) == 1
    assert events.count(("construct", False)) + events.count(("construct", True)) == 1
    service.close()
    assert events.count("close") == 1


def test_first_robot_observe_is_atomic_and_reuses_one_arbiter(monkeypatch, tmp_path):
    events = []
    observe_lock = Lock()
    observe_active = 0
    max_observe_active = 0

    class FakeAdapter(RobotAdapter):
        robot_id = "arm"

        def __init__(self, config):
            events.append("construct")

        def connect(self, *, prepare=True):
            events.append(("connect", prepare))
            time.sleep(0.03)

        def prepare(self):
            events.append("prepare")

        def observe(self):
            nonlocal observe_active, max_observe_active
            with observe_lock:
                observe_active += 1
                max_observe_active = max(max_observe_active, observe_active)
                time.sleep(0.01)
                value = RobotObservation(
                    1.0,
                    {"state": 1},
                    metadata={
                        "captured_timestamp_ns": time.monotonic_ns(),
                        "clock_domain": "host_monotonic_ns",
                    },
                )
                observe_active -= 1
            events.append("observe")
            return value

        def execute(self, action):
            return None

        def stop(self):
            return None

        def close(self):
            events.append("close")

    definition = RobotDefinition(
        kind="test.robot",
        config_factory=lambda robot_id, options: object(),
        adapter_type=FakeAdapter,
        environment_group="test",
    )
    monkeypatch.setattr(control_service_impl, "robot_definition", lambda _kind: definition)
    config = ControlServiceConfig.device_only(
        runtime_id="device-runtime",
        bind="127.0.0.1",
        port=18106,
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
    )
    service = ControlService(
        config,
        device_manager=DeviceManager("node-a", lock_dir=tmp_path / "locks"),
    )
    start = Barrier(3)
    results = []
    errors = []

    def read():
        try:
            start.wait()
            results.append(service.observe(include_robot=True))
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=read), Thread(target=read)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(2)
    try:
        assert not errors
        assert len(results) == 2
        assert events.count("construct") == 1
        assert events.count(("connect", False)) == 1
        assert events.count("observe") >= 1
        assert max_observe_active == 1
        assert len({result["snapshot_id"] for result in results}) == 1
        assert all(result["observation_status"]["available"] for result in results)
        assert service._current_arbiter is not None
    finally:
        service.close()


def test_passive_robot_upgrades_through_its_observation_scheduler(monkeypatch, tmp_path):
    events = []

    class FakeAdapter(RobotAdapter):
        robot_id = "arm"

        def __init__(self, config):
            pass

        def connect(self, *, prepare=True):
            events.append(("connect", prepare, current_thread().name))

        def prepare(self):
            events.append(("prepare", current_thread().name))

        def observe(self):
            events.append(("observe", current_thread().name))
            return RobotObservation(1.0, {"state": 1})

        def execute(self, action):
            return None

        def stop(self):
            return None

        def close(self):
            events.append(("close", current_thread().name))

    definition = RobotDefinition(
        kind="test.robot",
        config_factory=lambda robot_id, options: object(),
        adapter_type=FakeAdapter,
        environment_group="test",
    )
    monkeypatch.setattr(control_service_impl, "robot_definition", lambda _kind: definition)
    config = ControlServiceConfig.device_only(
        runtime_id="device-runtime",
        bind="127.0.0.1",
        port=18107,
        robot_id="arm",
        robot_kind="test.robot",
        robot_options={},
    )
    service = ControlService(
        config,
        device_manager=DeviceManager("node-a", lock_dir=tmp_path / "locks"),
    )
    service.observe(include_robot=True)
    service._ensure_robot_arbiter(definition)
    prepare_events = [item for item in events if item[0] == "prepare"]
    assert len(prepare_events) == 1
    assert prepare_events[0][1].startswith("embodirun-device-io-")
    service.close()
