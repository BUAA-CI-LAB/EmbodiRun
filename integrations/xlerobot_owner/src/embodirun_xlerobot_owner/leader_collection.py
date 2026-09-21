"""Persistent dual-SO101 leader/follower data collection on the AGX host.

The supervisor deliberately separates process resilience from motion safety:

* an SSH disconnect does not terminate the process when it runs inside tmux;
* a leader, follower, camera, disk, or robot-service fault interrupts the active
  episode, requests an arms-only Stop, and returns to a reconnecting state;
* reconnecting never re-arms the followers.  A human must explicitly start the
  next episode.
"""

from __future__ import annotations

import copy
import json
import math
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .recording import CAMERA_ROLES, EpisodeRecorder
from .robot import RemoteRobot

JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
LEADER_NAMES = tuple(f"{side}_arm_{suffix}" for side in ("left", "right") for suffix in JOINT_SUFFIXES)
ACTION_NAMES = tuple(f"{name}.pos" for name in LEADER_NAMES)
MODEL_NUMBER = 777
DIRECT_FOLLOW_VELOCITY_RAW = 3400
# The left follower bank reports a persistent Maximum_Acceleration of 50.
# Keep the volatile command at that confirmed common value while retaining
# maximum Goal_Velocity; the old 15 deg/s software limiter is not reintroduced.
DIRECT_FOLLOW_ACCELERATION_RAW = 50
LEADER_READ_RETRIES = 2
ALIGNMENT_JOINT_MOTION_TOLERANCE = 3.0
ALIGNMENT_GRIPPER_MOTION_TOLERANCE = 5.0


def load_leader_calibration(path: Path) -> dict[str, dict[str, int]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(LEADER_NAMES):
        raise ValueError("leader calibration must contain the exact 12 left/right joints")
    result: dict[str, dict[str, int]] = {}
    for offset in (0, 6):
        for expected_id, name in enumerate(LEADER_NAMES[offset : offset + 6], start=1):
            entry = value[name]
            if not isinstance(entry, dict):
                raise TypeError(f"calibration entry {name} must be an object")
            fields: dict[str, int] = {}
            for field_name in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
                item = entry.get(field_name)
                if isinstance(item, bool) or not isinstance(item, int):
                    raise TypeError(f"calibration.{name}.{field_name} must be an integer")
                fields[field_name] = item
            if fields["id"] != expected_id or fields["drive_mode"] not in (0, 1):
                raise ValueError(f"calibration.{name} has an invalid motor ID or drive mode")
            if not 0 <= fields["range_min"] < fields["range_max"] <= 4095:
                raise ValueError(f"calibration.{name} has an invalid observed range")
            result[name] = fields
    return result


def raw_to_action(raw_by_name: dict[str, int], calibration: dict[str, dict[str, int]]) -> dict[str, float]:
    action: dict[str, float] = {}
    for name in LEADER_NAMES:
        raw = int(raw_by_name[name])
        entry = calibration[name]
        bounded = max(entry["range_min"], min(entry["range_max"], raw))
        if name.endswith("_gripper"):
            value = (bounded - entry["range_min"]) * 100.0 / (entry["range_max"] - entry["range_min"])
            if entry["drive_mode"]:
                value = 100.0 - value
        else:
            middle = (entry["range_min"] + entry["range_max"]) / 2.0
            value = (bounded - middle) * 360.0 / 4095.0
        action[f"{name}.pos"] = value
    return action


def clamp_to_remote_limits(action: dict[str, float], metadata: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    limits = metadata.get("joint_limits")
    if not isinstance(limits, dict):
        raise TypeError("AGX metadata does not contain joint_limits")
    result: dict[str, float] = {}
    clamped: list[str] = []
    for key, value in action.items():
        bounds = limits.get(key)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bounds)
        ):
            raise ValueError(f"AGX metadata has no valid limits for {key}")
        low, high = float(bounds[0]), float(bounds[1])
        bounded = max(low, min(high, float(value)))
        if not math.isclose(bounded, float(value), rel_tol=0.0, abs_tol=1e-9):
            clamped.append(key)
        result[key] = bounded
    return result, clamped


def step_toward_target(
    current: dict[str, float],
    target: dict[str, float],
    *,
    elapsed_s: float,
    joint_speed_deg_s: float,
    gripper_speed_pct_s: float,
) -> tuple[dict[str, float], bool]:
    if set(current) != set(target):
        raise ValueError("alignment actions must have identical keys")
    result: dict[str, float] = {}
    aligned = True
    for key, target_value in target.items():
        speed = gripper_speed_pct_s if key.endswith("_gripper.pos") else joint_speed_deg_s
        maximum_delta = speed * elapsed_s
        delta = float(target_value) - float(current[key])
        if abs(delta) > maximum_delta:
            aligned = False
            result[key] = float(current[key]) + math.copysign(maximum_delta, delta)
        else:
            result[key] = float(target_value)
    return result, aligned


def moved_during_alignment(reference: dict[str, float], current: dict[str, float]) -> list[str]:
    if set(reference) != set(current):
        raise ValueError("alignment actions must have identical keys")
    moved = []
    for key, reference_value in reference.items():
        tolerance = (
            ALIGNMENT_GRIPPER_MOTION_TOLERANCE if key.endswith("_gripper.pos") else ALIGNMENT_JOINT_MOTION_TOLERANCE
        )
        if abs(float(current[key]) - float(reference_value)) > tolerance:
            moved.append(key)
    return moved


def validate_remote_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("source") != "physical" or metadata.get("allow_motion") is not True:
        raise RuntimeError("AGX is not reporting a motion-enabled physical backend")
    if metadata.get("joint_unit") != "degrees":
        raise RuntimeError("AGX joint units do not match the leader conversion")
    if metadata.get("gripper_unit") != "range_0_100":
        raise RuntimeError("AGX gripper units do not match the leader conversion")
    if set(metadata.get("enabled_arms", [])) != {"left", "right"}:
        raise RuntimeError("AGX does not report both follower arms enabled")
    if "arms" not in set(metadata.get("control_scopes", [])):
        raise RuntimeError("AGX does not expose the arms control scope")
    velocity = metadata.get("arm_velocity_raw")
    acceleration = metadata.get("arm_acceleration_raw")
    if not isinstance(velocity, dict) or velocity.get("value") != DIRECT_FOLLOW_VELOCITY_RAW:
        raise RuntimeError("AGX is not using the direct-follow velocity preset")
    if not isinstance(acceleration, dict) or acceleration.get("value") != DIRECT_FOLLOW_ACCELERATION_RAW:
        raise RuntimeError("AGX is not using the direct-follow acceleration preset")


def _import_sdk(sdk_src: str):
    source_path = Path(sdk_src).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"SDK source directory does not exist: {source_path}")
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    loaded = sys.modules.get("lerobot")
    if (
        loaded is not None
        and getattr(loaded, "__file__", None)
        and source_path not in Path(loaded.__file__).resolve().parents
    ):
        for name in list(sys.modules):
            if name == "lerobot" or name.startswith("lerobot."):
                del sys.modules[name]
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus, Motor, MotorNormMode


class LeaderReader:
    """Read two torque-free SO101 leaders without writing their motor registers."""

    def __init__(
        self,
        sdk_src: str,
        ports: dict[str, str],
        calibration: dict[str, dict[str, int]],
    ) -> None:
        bus_type, motor_type, norm_mode = _import_sdk(sdk_src)
        if set(ports) != {"left", "right"}:
            raise ValueError("leader ports must contain left and right")
        self.calibration = calibration
        self.ports = dict(ports)
        self.buses: dict[str, Any] = {}
        self._bus_type = bus_type
        self._motor_type = motor_type
        self._norm_mode = norm_mode

    def connect(self) -> None:
        if self.buses:
            raise RuntimeError("leader buses are already connected")
        try:
            for side in ("left", "right"):
                motors = {
                    f"{side}_arm_{suffix}": self._motor_type(motor_id, "sts3215", self._norm_mode.RANGE_M100_100)
                    for motor_id, suffix in enumerate(JOINT_SUFFIXES, start=1)
                }
                bus = self._bus_type(port=self.ports[side], motors=motors)
                bus.connect(handshake=False)
                bus.set_baudrate(bus.default_baudrate)
                self.buses[side] = bus
            self._preflight()
        except BaseException:
            self.close()
            raise

    def _preflight(self) -> None:
        errors = []
        for side, bus in self.buses.items():
            for motor_id, suffix in enumerate(JOINT_SUFFIXES, start=1):
                name = f"{side}_arm_{suffix}"
                model = bus.ping(
                    motor_id,
                    num_retry=LEADER_READ_RETRIES,
                    raise_on_error=False,
                )
                if model != MODEL_NUMBER:
                    errors.append(f"{name}: model {model!r}, expected {MODEL_NUMBER}")
                    continue
                torque = bus.read(
                    "Torque_Enable",
                    name,
                    normalize=False,
                    num_retry=LEADER_READ_RETRIES,
                )
                mode = bus.read(
                    "Operating_Mode",
                    name,
                    normalize=False,
                    num_retry=LEADER_READ_RETRIES,
                )
                offset = bus.read(
                    "Homing_Offset",
                    name,
                    normalize=False,
                    num_retry=LEADER_READ_RETRIES,
                )
                if torque != 0:
                    errors.append(f"{name}: leader torque must be disabled")
                if mode != 0:
                    errors.append(f"{name}: leader must be in position mode")
                if offset != self.calibration[name]["homing_offset"]:
                    errors.append(
                        f"{name}: homing offset {offset} does not match calibration "
                        f"{self.calibration[name]['homing_offset']}"
                    )
        if errors:
            raise RuntimeError("leader preflight failed: " + "; ".join(errors))

    def read(self, *, check_torque: bool = False) -> dict[str, float]:
        raw: dict[str, int] = {}
        for side, bus in self.buses.items():
            positions = self._sync_read(side, bus, "Present_Position")
            raw.update({name: int(value) for name, value in positions.items()})
            if check_torque:
                torque = self._sync_read(side, bus, "Torque_Enable")
                enabled = [name for name, value in torque.items() if int(value) != 0]
                if enabled:
                    raise RuntimeError(f"leader torque changed unexpectedly: {enabled}")
        return raw_to_action(raw, self.calibration)

    @staticmethod
    def _sync_read(side: str, bus: Any, field: str) -> dict[str, Any]:
        try:
            return bus.sync_read(
                field,
                normalize=False,
                num_retry=LEADER_READ_RETRIES,
            )
        except Exception as exc:
            raise ConnectionError(f"{side} leader failed to sync read {field} after retries: {exc}") from exc

    def close(self) -> None:
        for bus in tuple(self.buses.values()):
            try:
                bus.disconnect(disable_torque=False)
            except Exception:  # noqa: BLE001 -- SDK cleanup varies between recovered versions
                handler = getattr(bus, "port_handler", None)
                close_port = getattr(handler, "closePort", None)
                if callable(close_port):
                    close_port()
        self.buses.clear()


@dataclass(frozen=True)
class CollectionSettings:
    robot_url: str
    robot_token_file: Path
    sdk_src: str
    leader_calibration: Path
    leader_ports: dict[str, str]
    output: Path
    fps: float = 15.0
    reconnect_delay_s: float = 2.0
    alignment_speed_deg_s: float = 90.0
    alignment_gripper_speed_pct_s: float = 200.0
    alignment_timeout_s: float = 30.0

    @classmethod
    def from_file(cls, path: Path) -> CollectionSettings:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("collection config must be a JSON object")
        ports = value.get("leader_ports")
        if not isinstance(ports, dict) or set(ports) != {"left", "right"}:
            raise ValueError("leader_ports must contain left and right")
        return cls(
            robot_url=str(value["robot_url"]),
            robot_token_file=Path(value["robot_token_file"]),
            sdk_src=str(value["sdk_src"]),
            leader_calibration=Path(value["leader_calibration"]),
            leader_ports={side: str(ports[side]) for side in ("left", "right")},
            output=Path(value["output"]),
            fps=float(value.get("fps", 15.0)),
            reconnect_delay_s=float(value.get("reconnect_delay_s", 2.0)),
            alignment_speed_deg_s=float(value.get("alignment_speed_deg_s", 90.0)),
            alignment_gripper_speed_pct_s=float(value.get("alignment_gripper_speed_pct_s", 200.0)),
            alignment_timeout_s=float(value.get("alignment_timeout_s", 30.0)),
        )

    def validate(self) -> None:
        for name in (
            "fps",
            "reconnect_delay_s",
            "alignment_speed_deg_s",
            "alignment_gripper_speed_pct_s",
            "alignment_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not 5 <= self.fps <= 25:
            raise ValueError("fps must be between 5 and 25")
        if not self.robot_url.startswith(("http://", "https://")):
            raise ValueError("robot_url must be HTTP or HTTPS")


@dataclass
class _Request:
    kind: str
    task: str | None = None
    success: bool | None = None
    generation: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class _CommandError(RuntimeError):
    """An operator command was rejected without implying a hardware fault."""


class CollectionSupervisor:
    """Control-loop supervisor with explicit episode boundaries."""

    def __init__(
        self,
        settings: CollectionSettings,
        *,
        remote_factory: Callable[[], Any] | None = None,
        leader_factory: Callable[[], Any] | None = None,
        recorder_factory: Callable[[], EpisodeRecorder] | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._remote_factory = remote_factory or self._default_remote
        self._leader_factory = leader_factory or self._default_leader
        self._recorder_factory = recorder_factory or (lambda: EpisodeRecorder(settings.output, fps=settings.fps))
        self._requests: queue.Queue[_Request] = queue.Queue()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._remote: Any = None
        self._leader: Any = None
        self._recorder: EpisodeRecorder | None = None
        self._iteration = 0
        self._last_source_timestamp_ns: int | None = None
        self._arm_attempted = False
        self._connection_generation = 0
        self._status: dict[str, Any] = {
            "state": "created",
            "connected": False,
            "armed": False,
            "recording": False,
            "task": None,
            "episode_path": None,
            "last_episode": None,
            "last_error": None,
            "stop_confirmed": None,
            "clamped_joints": [],
        }

    def _default_remote(self) -> RemoteRobot:
        token = self.settings.robot_token_file.read_text(encoding="utf-8").strip()
        return RemoteRobot(self.settings.robot_url, token, timeout=2.0, scope="arms")

    def _default_leader(self) -> LeaderReader:
        calibration = load_leader_calibration(self.settings.leader_calibration)
        return LeaderReader(self.settings.sdk_src, self.settings.leader_ports, calibration)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("collection supervisor is already running")
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run, name="so101-collection", daemon=False)
        self._thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def command(
        self,
        kind: str,
        *,
        task: str | None = None,
        success: bool | None = None,
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("collection supervisor is not running")
        status = self.status()
        if kind == "begin" and status["state"] not in {"ready", "following"}:
            raise RuntimeError("hardware is not ready; wait for state=ready")
        request = _Request(
            kind=kind,
            task=task,
            success=success,
            generation=self._connection_generation,
        )
        self._requests.put(request)
        if not request.done.wait(timeout_s):
            raise TimeoutError(f"collection command {kind!r} timed out")
        if request.error is not None:
            raise RuntimeError(str(request.error)) from request.error
        return request.result or self.status()

    def shutdown(self, timeout_s: float = 30.0) -> dict[str, Any]:
        if self._thread is None:
            return self.status()
        if self._thread.is_alive():
            self._shutdown.set()
            self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("collection supervisor did not stop")
        return self.status()

    def _publish(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)

    def _run(self) -> None:
        self._publish(state="waiting_hardware")
        try:
            while not self._shutdown.is_set():
                if self._remote is None or self._leader is None:
                    try:
                        self._connect_once()
                    except Exception as exc:  # noqa: BLE001 -- every hardware failure is reconnectable
                        self._publish(
                            state="waiting_hardware",
                            connected=False,
                            armed=False,
                            last_error=f"{type(exc).__name__}: {exc}",
                        )
                        self._shutdown.wait(self.settings.reconnect_delay_s)
                        continue
                request: _Request | None = None
                try:
                    request = self._next_request(block=not self._remote.armed)
                    if request is not None:
                        self._handle_request(request)
                    elif self._remote.armed:
                        self._control_cycle()
                    else:
                        self._idle_health_check()
                except Exception as exc:  # noqa: BLE001 -- all loop failures must stop and reconnect
                    if request is not None and not request.done.is_set():
                        request.error = exc
                        request.done.set()
                    self._handle_fault(exc)
        finally:
            self._interrupt_recording("supervisor_shutdown")
            self._safe_stop()
            self._close_connections()
            self._publish(state="stopped", connected=False, armed=False, recording=False)
            while True:
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    break
                request.error = RuntimeError("collection supervisor stopped")
                request.done.set()

    def _next_request(self, *, block: bool) -> _Request | None:
        try:
            return self._requests.get(timeout=0.5 if block else 0.0)
        except queue.Empty:
            return None

    def _connect_once(self) -> None:
        remote = self._remote_factory()
        remote.connect()
        validate_remote_metadata(remote.metadata)
        leader = self._leader_factory()
        try:
            leader.connect()
            leader.read(check_torque=True)
            observation, images = remote.read()
            self._validate_observation(observation, images)
        except BaseException:
            leader.close()
            raise
        self._remote = remote
        self._leader = leader
        self._connection_generation += 1
        self._arm_attempted = False
        self._iteration = 0
        self._publish(
            state="ready",
            connected=True,
            armed=False,
            recording=False,
            task=None,
            episode_path=None,
            last_error=None,
            stop_confirmed=True,
        )

    @staticmethod
    def _validate_observation(observation: dict, images: dict[str, bytes]) -> None:
        errors = observation.get("errors")
        if errors:
            raise RuntimeError("AGX observation error: " + "; ".join(errors))
        if not set(ACTION_NAMES) <= set(observation.get("state", {})):
            raise RuntimeError("AGX observation is missing follower joint state")
        missing = set(CAMERA_ROLES) - set(images)
        if missing:
            raise RuntimeError("AGX observation is missing cameras: " + ", ".join(sorted(missing)))
        if not isinstance(observation.get("source_timestamp_ns"), int):
            raise TypeError("AGX observation has no source timestamp")

    def _handle_request(self, request: _Request) -> None:
        try:
            if request.generation != self._connection_generation:
                raise _CommandError("hardware reconnected after this command; explicitly start again")
            if request.kind == "begin":
                if not (request.task or "").strip():
                    raise _CommandError("task description must not be blank")
                if self._recorder is not None and self._recorder.active:
                    raise _CommandError("an episode is already recording")
                request.result = self._begin_episode(request.task or "")
            elif request.kind == "finish":
                if self._recorder is None or not self._recorder.active:
                    raise _CommandError("no episode is recording")
                request.result = self._finish_episode(request.success)
            elif request.kind == "abort":
                if self._recorder is None or not self._recorder.active:
                    raise _CommandError("no episode is recording")
                request.result = self._finish_episode(None, reason="operator_abort")
            elif request.kind == "stop":
                self._interrupt_recording("operator_stop")
                self._safe_stop()
                self._publish(state="ready", armed=False, recording=False, task=None)
                request.result = self.status()
            else:
                raise _CommandError(f"unknown collection command: {request.kind}")
        except _CommandError as exc:
            request.error = exc
        except BaseException as exc:
            request.error = exc
            raise
        finally:
            request.done.set()

    def _begin_episode(self, task: str) -> dict[str, Any]:
        if not task.strip():
            raise ValueError("task description must not be blank")
        if self._recorder is not None and self._recorder.active:
            raise RuntimeError("an episode is already recording")
        if self._remote is None or self._leader is None:
            raise RuntimeError("hardware is not ready")
        if not self._remote.armed:
            self._align_and_arm()
        recorder = self._recorder_factory()
        metadata = {
            **self._remote.metadata,
            "operator_mode": "dual_so101_leader_on_agx",
            "collection_mode": "teleoperation",
            "trainable": True,
            "fps": self.settings.fps,
            "max_skew_ms": 100,
            "max_gap_ms": 200,
            "leader_ports": dict(self.settings.leader_ports),
            "leader_calibration": str(self.settings.leader_calibration),
        }
        path = recorder.start(task, metadata)
        self._recorder = recorder
        self._last_source_timestamp_ns = None
        self._publish(
            state="recording",
            armed=True,
            recording=True,
            task=task.strip(),
            episode_path=str(path),
        )
        return self.status()

    def _align_and_arm(self) -> None:
        assert self._remote is not None and self._leader is not None
        leader_action = self._leader.read(check_torque=True)
        follower, images = self._remote.read()
        self._validate_observation(follower, images)
        target, clamped = clamp_to_remote_limits(leader_action, self._remote.metadata)
        self._publish(state="aligning", clamped_joints=clamped)
        self._arm_attempted = True
        result = self._remote.arm()
        if result.get("armed") is not True:
            raise RuntimeError("AGX refused to arm: " + "; ".join(result.get("errors", [])))
        if result.get("goal_velocity_raw") != DIRECT_FOLLOW_VELOCITY_RAW:
            raise RuntimeError("AGX did not confirm the direct-follow profile")
        commanded = {key: float(follower["state"][key]) for key in target}
        commanded, _ = clamp_to_remote_limits(commanded, self._remote.metadata)
        deadline = time.monotonic() + self.settings.alignment_timeout_s
        period = 1.0 / self.settings.fps
        while not self._shutdown.is_set():
            current = self._leader.read(check_torque=True)
            current, _ = clamp_to_remote_limits(current, self._remote.metadata)
            moved = moved_during_alignment(target, current)
            if moved:
                raise RuntimeError("leaders moved during initial alignment: " + ", ".join(moved))
            commanded, aligned = step_toward_target(
                commanded,
                target,
                elapsed_s=period,
                joint_speed_deg_s=self.settings.alignment_speed_deg_s,
                gripper_speed_pct_s=self.settings.alignment_gripper_speed_pct_s,
            )
            feedback = self._remote.command(commanded)
            self._validate_feedback(feedback)
            applied = feedback.get("applied_action")
            if isinstance(applied, dict) and set(applied) == set(commanded):
                commanded = {key: float(value) for key, value in applied.items()}
            if aligned:
                self._publish(state="following", armed=True)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("initial leader/follower alignment timed out")
            self._shutdown.wait(period)
        raise RuntimeError("collection shutdown during alignment")

    @staticmethod
    def _validate_feedback(feedback: dict) -> None:
        if feedback.get("accepted") is not True or feedback.get("command_accepted") is not True:
            raise RuntimeError("AGX rejected command: " + "; ".join(feedback.get("errors", [])))
        if not isinstance(feedback.get("applied_action"), dict):
            raise TypeError("AGX did not return the applied action")

    def _control_cycle(self) -> None:
        assert self._remote is not None and self._leader is not None
        began = time.monotonic()
        observation, images = self._remote.read()
        self._validate_observation(observation, images)
        leader_action = self._leader.read(check_torque=self._iteration % max(1, round(self.settings.fps)) == 0)
        bounded, clamped = clamp_to_remote_limits(leader_action, self._remote.metadata)
        feedback = self._remote.command(bounded)
        self._validate_feedback(feedback)
        applied = {key: float(value) for key, value in feedback["applied_action"].items()}
        self._publish(clamped_joints=clamped, armed=True)
        if self._recorder is not None and self._recorder.active:
            source = observation["source_timestamp_ns"]
            if source != self._last_source_timestamp_ns:
                self._recorder.append(
                    observation=observation,
                    action=applied,
                    images=images,
                    input_sample={
                        "source": "dual_so101_leader",
                        "leader_action": leader_action,
                        "bounded_action": bounded,
                        "clamped_joints": clamped,
                    },
                    feedback=feedback,
                )
                self._last_source_timestamp_ns = source
        self._iteration += 1
        remaining = 1.0 / self.settings.fps - (time.monotonic() - began)
        if remaining > 0:
            self._shutdown.wait(remaining)

    def _idle_health_check(self) -> None:
        assert self._remote is not None and self._leader is not None
        self._leader.read(check_torque=True)
        observation, images = self._remote.read()
        self._validate_observation(observation, images)

    def _finish_episode(self, success: bool | None, *, reason: str = "operator") -> dict[str, Any]:
        if self._recorder is None or not self._recorder.active:
            raise RuntimeError("no episode is recording")
        result = self._recorder.finish(success=success, reason=reason)
        self._recorder = None
        self._publish(
            state="following" if self._remote is not None and self._remote.armed else "ready",
            recording=False,
            task=None,
            episode_path=None,
            last_episode=result,
        )
        return result

    def _interrupt_recording(self, reason: str) -> None:
        if self._recorder is None or not self._recorder.active:
            return
        try:
            result = self._recorder.force_interrupted_close(reason=reason)
            self._publish(last_episode=result)
        finally:
            self._recorder = None
            self._publish(recording=False, task=None, episode_path=None)

    def _safe_stop(self) -> None:
        remote = self._remote
        if remote is None or not (remote.armed or self._arm_attempted):
            return
        try:
            result = remote.stop()
            confirmed = result.get("stop_confirmed") is True
            self._publish(stop_confirmed=confirmed)
        except Exception as exc:  # noqa: BLE001 -- failure evidence must survive any client error
            self._publish(stop_confirmed=False, last_error=f"stop failed: {exc}")
        finally:
            self._arm_attempted = False
            self._publish(armed=False)

    def _close_connections(self) -> None:
        if self._leader is not None:
            self._leader.close()
        self._leader = None
        self._remote = None

    def _handle_fault(self, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        try:
            self._interrupt_recording("hardware_disconnect")
        except Exception as recorder_exc:  # noqa: BLE001 -- still attempt Stop after disk failure
            detail += f"; recorder close failed: {recorder_exc}"
        self._safe_stop()
        self._close_connections()
        self._publish(
            state="waiting_hardware",
            connected=False,
            armed=False,
            recording=False,
            last_error=detail,
        )
