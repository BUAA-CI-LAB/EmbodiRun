from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from embodirun_xlerobot_owner.leader_collection import (
    ACTION_NAMES,
    DIRECT_FOLLOW_ACCELERATION_RAW,
    DIRECT_FOLLOW_VELOCITY_RAW,
    LEADER_NAMES,
    CollectionSettings,
    CollectionSupervisor,
    LeaderReader,
    load_leader_calibration,
    raw_to_action,
)
from embodirun_xlerobot_owner.recording import CAMERA_ROLES

JPEG = b"\xff\xd8so101-test\xff\xd9"


def _metadata() -> dict:
    limits = {name: ([0.0, 100.0] if name.endswith("gripper.pos") else [-180.0, 180.0]) for name in ACTION_NAMES}
    return {
        "source": "physical",
        "allow_motion": True,
        "joint_names": list(ACTION_NAMES),
        "joint_unit": "degrees",
        "gripper_unit": "range_0_100",
        "camera_roles_confirmed": True,
        "cameras": list(CAMERA_ROLES),
        "enabled_arms": ["left", "right"],
        "control_scopes": ["arms"],
        "joint_limits": limits,
        "arm_velocity_raw": {"value": DIRECT_FOLLOW_VELOCITY_RAW},
        "arm_acceleration_raw": {"value": DIRECT_FOLLOW_ACCELERATION_RAW},
    }


class FakeRemote:
    def __init__(self) -> None:
        self.metadata = _metadata()
        self.armed = False
        self.state = dict.fromkeys(ACTION_NAMES, 0.0)
        self.timestamp_ns = 1_000_000_000
        self.stop_calls = 0

    def connect(self) -> None:
        return None

    def read(self):
        self.timestamp_ns += 100_000_000
        observation = {
            "state": dict(self.state),
            "source_timestamp_ns": self.timestamp_ns,
            "state_timestamp_ns": self.timestamp_ns,
            "camera_timestamps_ns": dict.fromkeys(CAMERA_ROLES, self.timestamp_ns),
            "metadata": self.metadata,
            "errors": [],
        }
        return observation, dict.fromkeys(CAMERA_ROLES, JPEG)

    def arm(self) -> dict:
        self.armed = True
        return {
            "armed": True,
            "goal_velocity_raw": DIRECT_FOLLOW_VELOCITY_RAW,
            "errors": [],
        }

    def command(self, action: dict[str, float]) -> dict:
        if not self.armed:
            raise RuntimeError("not armed")
        self.state.update(action)
        return {
            "accepted": True,
            "command_accepted": True,
            "applied_action": dict(action),
            "sent_timestamp_ns": self.timestamp_ns + 1,
            "physical_outcome": "unknown",
            "errors": [],
        }

    def stop(self) -> dict:
        self.stop_calls += 1
        self.armed = False
        return {"stop_confirmed": True}


class FakeLeader:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.read_count = 0
        self.closed = False

    def connect(self) -> None:
        return None

    def read(self, *, check_torque: bool = False) -> dict[str, float]:
        del check_torque
        self.read_count += 1
        if self.fail_after is not None and self.read_count > self.fail_after:
            raise ConnectionError("leader unplugged")
        return dict.fromkeys(ACTION_NAMES, 0.0)

    def close(self) -> None:
        self.closed = True


def _settings(tmp_path: Path) -> CollectionSettings:
    return CollectionSettings(
        robot_url="http://127.0.0.1:8766",
        robot_token_file=tmp_path / "token",
        sdk_src=str(tmp_path / "sdk"),
        leader_calibration=tmp_path / "calibration.json",
        leader_ports={"left": "/dev/left", "right": "/dev/right"},
        output=tmp_path / "episodes",
        fps=10.0,
        reconnect_delay_s=0.02,
        alignment_speed_deg_s=90.0,
        alignment_gripper_speed_pct_s=200.0,
        alignment_timeout_s=1.0,
    )


def _wait_for(supervisor: CollectionSupervisor, predicate, timeout_s: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = supervisor.status()
        if predicate(status):
            return status
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for status: {supervisor.status()}")


def test_calibration_conversion_uses_exact_joint_set(tmp_path: Path) -> None:
    payload = {}
    for index, name in enumerate(LEADER_NAMES):
        payload[name] = {
            "id": index % 6 + 1,
            "drive_mode": 1 if name.endswith("gripper") else 0,
            "homing_offset": index,
            "range_min": 100,
            "range_max": 3000,
        }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    calibration = load_leader_calibration(path)
    action = raw_to_action(dict.fromkeys(LEADER_NAMES, 100), calibration)

    assert set(action) == set(ACTION_NAMES)
    assert action["left_arm_gripper.pos"] == pytest.approx(100.0)
    assert action["left_arm_shoulder_pan.pos"] < 0.0


def test_leader_preflight_uses_recovered_sdk_read_signature(tmp_path: Path) -> None:
    payload = {
        name: {
            "id": index % 6 + 1,
            "drive_mode": 0,
            "homing_offset": index,
            "range_min": 100,
            "range_max": 3000,
        }
        for index, name in enumerate(LEADER_NAMES)
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    class SignatureStrictBus:
        def ping(self, motor: int, *, num_retry: int, raise_on_error: bool) -> int:
            del motor, num_retry, raise_on_error
            return 777

        def read(
            self,
            data_name: str,
            motor: str,
            *,
            normalize: bool,
            num_retry: int,
        ) -> int:
            del normalize, num_retry
            if data_name in {"Torque_Enable", "Operating_Mode"}:
                return 0
            if data_name == "Homing_Offset":
                return payload[motor]["homing_offset"]
            raise AssertionError(data_name)

    reader = object.__new__(LeaderReader)
    reader.calibration = load_leader_calibration(path)
    reader.buses = {"left": SignatureStrictBus(), "right": SignatureStrictBus()}
    reader._preflight()


def test_supervisor_records_multiple_operator_labeled_episodes(tmp_path: Path) -> None:
    remote = FakeRemote()
    supervisor = CollectionSupervisor(
        _settings(tmp_path),
        remote_factory=lambda: remote,
        leader_factory=FakeLeader,
    )
    supervisor.start()
    _wait_for(supervisor, lambda status: status["state"] == "ready")

    first = supervisor.command("begin", task="pick red block")
    assert first["state"] == "recording"
    _wait_for(
        supervisor,
        lambda status: status["recording"] and Path(status["episode_path"], "frames.jsonl").stat().st_size > 0,
    )
    result = supervisor.command("finish", success=True)
    assert result["status"] == "succeeded"
    assert result["frame_count"] >= 1
    assert remote.armed is True

    second = supervisor.command("begin", task="retry placement")
    assert second["recording"] is True
    result = supervisor.command("abort")
    assert result["status"] == "aborted"
    assert result["reason"] == "operator_abort"

    final = supervisor.shutdown()
    assert final["state"] == "stopped"
    assert final["stop_confirmed"] is True
    assert remote.stop_calls == 1


def test_hardware_disconnect_interrupts_episode_and_requires_new_start(tmp_path: Path) -> None:
    remote = FakeRemote()
    leader = FakeLeader(fail_after=3)
    remote_factory_calls = 0

    def remote_factory():
        nonlocal remote_factory_calls
        remote_factory_calls += 1
        if remote_factory_calls > 1:
            raise ConnectionError("hardware remains offline")
        return remote

    supervisor = CollectionSupervisor(
        _settings(tmp_path),
        remote_factory=remote_factory,
        leader_factory=lambda: leader,
    )
    supervisor.start()
    _wait_for(supervisor, lambda status: status["state"] == "ready")
    supervisor.command("begin", task="disconnect test")

    status = _wait_for(
        supervisor,
        lambda item: item["state"] == "waiting_hardware" and item.get("last_episode") is not None,
    )
    assert status["recording"] is False
    assert status["armed"] is False
    assert status["last_episode"]["status"] == "interrupted"
    assert status["last_episode"]["reason"] == "hardware_disconnect"
    assert remote.stop_calls == 1
    assert supervisor._thread is not None and supervisor._thread.is_alive()
    with pytest.raises(RuntimeError, match="hardware is not ready"):
        supervisor.command("begin", task="must not auto resume")
    supervisor.shutdown()
