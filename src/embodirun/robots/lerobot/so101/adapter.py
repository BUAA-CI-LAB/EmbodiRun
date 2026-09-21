"""Fail-closed SO-101 follower execution through the Feetech SDK."""

from __future__ import annotations

import inspect
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...adapter import RobotAction, RobotAdapter, RobotObservation
from .config import SO101Config

SO101_ACTION_SPACE = "lerobot.so101.position.v1"
SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_MOTORS = (*SO101_JOINTS, "gripper")
SO101_POSITION_FEATURES = tuple(f"{name}.pos" for name in SO101_MOTORS)

_MOTOR_IDS = dict(zip(SO101_MOTORS, range(1, 7)))
_STS3215_MODEL_NUMBER = 777
_ENCODER_RESOLUTION = 4096
_READ_RETRIES = 2

_RETURN_DELAY = (7, 1)
_MIN_POSITION = (9, 2)
_MAX_POSITION = (11, 2)
_MAX_TORQUE = (16, 2)
_PHASE = (18, 1)
_P_COEFFICIENT = (21, 1)
_D_COEFFICIENT = (22, 1)
_I_COEFFICIENT = (23, 1)
_PROTECTION_CURRENT = (28, 2)
_HOMING_OFFSET = (31, 2)
_OPERATING_MODE = (33, 1)
_OVERLOAD_TORQUE = (36, 1)
_TORQUE_ENABLE = (40, 1)
_ACCELERATION = (41, 1)
_GOAL_POSITION = (42, 2)
_LOCK = (55, 1)
_PRESENT_POSITION = (56, 2)
_MAXIMUM_ACCELERATION = (85, 1)


class SO101AdapterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _MotorCalibration:
    motor_id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


class _SO101Controller(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def is_calibrated(self) -> bool: ...

    def connect(self, *, calibrate: bool, prepare: bool = True) -> None: ...

    def get_observation(self) -> Mapping[str, object]: ...

    def send_action(self, action: Mapping[str, float]) -> None: ...

    def disconnect(self) -> None: ...


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SO101AdapterError(f"{name} must be an integer")
    return value


def _default_calibration_dir() -> Path:
    configured = os.environ.get("HF_LEROBOT_CALIBRATION")
    if configured:
        root = Path(configured).expanduser()
    else:
        lerobot_home = os.environ.get("HF_LEROBOT_HOME")
        if lerobot_home:
            root = Path(lerobot_home).expanduser() / "calibration"
        else:
            hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
            root = hf_home / "lerobot" / "calibration"
    return root / "robots" / "so_follower"


def _load_calibration(config: SO101Config) -> dict[str, _MotorCalibration]:
    calibration_dir = config.calibration_dir or _default_calibration_dir()
    calibration_id = config.calibration_id or config.robot_id
    path = calibration_dir / f"{calibration_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SO101AdapterError(f"SO-101 calibration file does not exist: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise SO101AdapterError(f"failed to read SO-101 calibration file {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise SO101AdapterError(f"SO-101 calibration file must contain an object: {path}")

    expected = set(SO101_MOTORS)
    actual = set(value)
    if actual != expected:
        raise SO101AdapterError(
            "SO-101 calibration motors do not match the hardware; "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}"
        )

    calibration: dict[str, _MotorCalibration] = {}
    required_fields = {
        "id",
        "drive_mode",
        "homing_offset",
        "range_min",
        "range_max",
    }
    for motor in SO101_MOTORS:
        item = value[motor]
        if not isinstance(item, Mapping) or set(item) != required_fields:
            raise SO101AdapterError(f"SO-101 calibration for {motor!r} must contain {sorted(required_fields)!r}")
        motor_id = _integer(item["id"], f"calibration.{motor}.id")
        drive_mode = _integer(item["drive_mode"], f"calibration.{motor}.drive_mode")
        homing_offset = _integer(item["homing_offset"], f"calibration.{motor}.homing_offset")
        range_min = _integer(item["range_min"], f"calibration.{motor}.range_min")
        range_max = _integer(item["range_max"], f"calibration.{motor}.range_max")
        if motor_id != _MOTOR_IDS[motor]:
            raise SO101AdapterError(f"calibration.{motor}.id must be {_MOTOR_IDS[motor]}, got {motor_id}")
        if drive_mode not in (0, 1):
            raise SO101AdapterError(f"calibration.{motor}.drive_mode must be 0 or 1")
        if range_min == range_max:
            raise SO101AdapterError(f"calibration.{motor} range_min and range_max must differ")
        calibration[motor] = _MotorCalibration(
            motor_id=motor_id,
            drive_mode=drive_mode,
            homing_offset=homing_offset,
            range_min=range_min,
            range_max=range_max,
        )
    return calibration


def _decode_sign_magnitude(value: int, sign_bit: int) -> int:
    mask = 1 << sign_bit
    return -(value & ~mask) if value & mask else value


def _encode_sign_magnitude(value: int, sign_bit: int) -> int:
    return (-value | (1 << sign_bit)) if value < 0 else value


class _FeetechSO101Controller:
    """Minimal STS3215 bus used by the Deploy SO-101 adapter."""

    def __init__(self, config: SO101Config, *, read_only: bool = False) -> None:
        try:
            import scservo_sdk
        except ImportError as error:
            raise SO101AdapterError(
                "SO-101 support requires the Feetech servo SDK; run `uv sync --frozen --no-dev --group robot-so101`"
            ) from error

        self.config = config
        self.read_only = read_only
        self.calibration = _load_calibration(config)
        self.sdk = scservo_sdk
        self.port_handler = scservo_sdk.PortHandler(config.port)
        # The PyPI SDK underestimates packet timeouts for a multi-motor bus.
        self.port_handler.setPacketTimeout = self._set_packet_timeout
        self.packet_handler = scservo_sdk.PacketHandler(0)
        self._connected = False
        self._calibrated = False
        self._prepared = False
        # Startup may fail after partially enabling the bus.  Keep this
        # separate from ``_prepared`` so close can still attempt torque off,
        # while a passive connection never writes torque registers.
        self._torque_owned = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def _set_packet_timeout(self, packet_length: int) -> None:
        self.port_handler.packet_start_time = self.port_handler.getCurrentTime()
        byte_time = self.port_handler.tx_time_per_byte
        self.port_handler.packet_timeout = byte_time * packet_length + byte_time * 3 + 50

    def _communication_error(self, operation: str, result: int, error: int) -> None:
        if result != self.sdk.COMM_SUCCESS:
            detail = self.packet_handler.getTxRxResult(result)
            raise SO101AdapterError(f"{operation} failed: {detail}")
        if error:
            detail = self.packet_handler.getRxPacketError(error)
            raise SO101AdapterError(f"{operation} failed: {detail}")

    def _read_register(
        self,
        motor_id: int,
        register: tuple[int, int],
        *,
        sign_bit: int | None = None,
        retries: int = _READ_RETRIES,
    ) -> int:
        address, size = register
        read = {
            1: self.packet_handler.read1ByteTxRx,
            2: self.packet_handler.read2ByteTxRx,
        }[size]
        for attempt in range(retries + 1):
            value, result, error = read(self.port_handler, motor_id, address)
            if result == self.sdk.COMM_SUCCESS and not error:
                return _decode_sign_magnitude(value, sign_bit) if sign_bit is not None else value
            if attempt == retries:
                self._communication_error(f"read register {address} from motor {motor_id}", result, error)
        raise AssertionError("unreachable")

    def _write_register(
        self,
        motor_id: int,
        register: tuple[int, int],
        value: int,
        *,
        sign_bit: int | None = None,
        retries: int = _READ_RETRIES,
    ) -> None:
        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot write registers")
        address, size = register
        if sign_bit is not None:
            value = _encode_sign_magnitude(value, sign_bit)
        write = {
            1: self.packet_handler.write1ByteTxRx,
            2: self.packet_handler.write2ByteTxRx,
        }[size]
        for attempt in range(retries + 1):
            result, error = write(self.port_handler, motor_id, address, value)
            if result == self.sdk.COMM_SUCCESS and not error:
                return
            if attempt == retries:
                self._communication_error(f"write register {address} on motor {motor_id}", result, error)

    def _handshake(self) -> None:
        firmware_versions: set[tuple[int, int]] = set()
        for motor, motor_id in _MOTOR_IDS.items():
            model, result, error = self.packet_handler.ping(self.port_handler, motor_id)
            self._communication_error(f"ping motor {motor!r}", result, error)
            if model != _STS3215_MODEL_NUMBER:
                raise SO101AdapterError(
                    f"motor {motor!r} has model number {model}; expected STS3215 ({_STS3215_MODEL_NUMBER})"
                )
            firmware_versions.add(
                (
                    self._read_register(motor_id, (0, 1)),
                    self._read_register(motor_id, (1, 1)),
                )
            )
        if len(firmware_versions) != 1:
            raise SO101AdapterError(
                f"SO-101 motors must use the same firmware version; found {sorted(firmware_versions)!r}"
            )

    def _matches_calibration(self) -> bool:
        for item in self.calibration.values():
            motor_id = item.motor_id
            if self._read_register(motor_id, _MIN_POSITION) != item.range_min:
                return False
            if self._read_register(motor_id, _MAX_POSITION) != item.range_max:
                return False
            if self._read_register(motor_id, _HOMING_OFFSET, sign_bit=11) != item.homing_offset:
                return False
        return True

    def _set_torque(self, enabled: bool) -> None:
        errors: list[str] = []
        for motor_id in _MOTOR_IDS.values():
            for register in (_TORQUE_ENABLE, _LOCK):
                try:
                    self._write_register(motor_id, register, int(enabled))
                except Exception as error:
                    if enabled:
                        raise
                    errors.append(str(error))
        if errors:
            raise SO101AdapterError("torque shutdown incomplete: " + "; ".join(errors))

    def _initial_goal(self, motor: str, present: int) -> int:
        item = self.calibration[motor]
        target = min(item.range_max, max(item.range_min, present))
        if motor == "gripper":
            limit = self.config.max_gripper_step * (item.range_max - item.range_min) / 100
        else:
            limit = self.config.max_joint_step_deg * (_ENCODER_RESOLUTION - 1) / 360
        if abs(target - present) > limit:
            raise SO101AdapterError(
                f"motor {motor!r} is too far outside its calibrated range to initialize "
                "within the configured step limit"
            )
        return target

    def _configure(self) -> None:
        try:
            self._set_torque(False)
            for motor, motor_id in _MOTOR_IDS.items():
                self._write_register(motor_id, _RETURN_DELAY, 0)
                self._write_register(motor_id, _MAXIMUM_ACCELERATION, 254)
                self._write_register(motor_id, _ACCELERATION, 254)
                phase = self._read_register(motor_id, _PHASE)
                if phase & 0x10:
                    self._write_register(motor_id, _PHASE, phase & ~0x10)
                self._write_register(motor_id, _OPERATING_MODE, 0)
                self._write_register(motor_id, _P_COEFFICIENT, 16)
                self._write_register(motor_id, _I_COEFFICIENT, 0)
                self._write_register(motor_id, _D_COEFFICIENT, 32)
                if motor == "gripper":
                    self._write_register(motor_id, _MAX_TORQUE, 500)
                    self._write_register(motor_id, _PROTECTION_CURRENT, 250)
                    self._write_register(motor_id, _OVERLOAD_TORQUE, 25)
            initial_goals = {}
            for motor, motor_id in _MOTOR_IDS.items():
                present = self._read_register(motor_id, _PRESENT_POSITION, sign_bit=15)
                initial_goals[motor_id] = self._initial_goal(motor, present)
            for motor_id, target in initial_goals.items():
                self._write_register(motor_id, _GOAL_POSITION, target, sign_bit=15)
            self._torque_owned = True
            self._set_torque(True)
        except BaseException as startup_error:
            # Goal writes and partial enabling can also fail; stop every motor.
            try:
                self._set_torque(False)
                self._torque_owned = False
            except Exception as cleanup_error:
                raise startup_error from cleanup_error
            raise

    def connect(self, *, calibrate: bool, prepare: bool = True) -> None:
        if calibrate:
            raise SO101AdapterError("interactive calibration is not supported by the Deploy runtime")
        if self.is_connected:
            return
        try:
            if not self.port_handler.openPort():
                raise SO101AdapterError(f"failed to open SO-101 port {self.config.port}")
            self._connected = True
            self.port_handler.setPacketTimeoutMillis(1000)
            self._handshake()
            self._calibrated = self._matches_calibration()
            if self._calibrated and prepare and not self.read_only:
                self._configure()
                self._prepared = True
            else:
                self._prepared = False
        except BaseException:
            self._close_port()
            raise

    def prepare(self) -> None:
        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot prepare motors")
        if not self.is_connected:
            raise SO101AdapterError("SO-101 is not connected")
        if not self.is_calibrated:
            raise SO101AdapterError("SO-101 calibration has not been verified")
        if not self._prepared:
            self._configure()
            self._prepared = True

    def _normalized_position(self, motor: str, raw: int) -> float:
        item = self.calibration[motor]
        if motor == "gripper":
            bounded = min(item.range_max, max(item.range_min, raw))
            normalized = (bounded - item.range_min) * 100 / (item.range_max - item.range_min)
            return 100 - normalized if item.drive_mode else normalized
        midpoint = (item.range_min + item.range_max) / 2
        return (raw - midpoint) * 360 / (_ENCODER_RESOLUTION - 1)

    def _raw_position(self, motor: str, normalized: float) -> int:
        item = self.calibration[motor]
        if motor == "gripper":
            bounded = min(100.0, max(0.0, normalized))
            if item.drive_mode:
                bounded = 100 - bounded
            return int(bounded / 100 * (item.range_max - item.range_min) + item.range_min)
        midpoint = (item.range_min + item.range_max) / 2
        return int(normalized * (_ENCODER_RESOLUTION - 1) / 360 + midpoint)

    def _read_positions(self) -> dict[str, float]:
        address, size = _PRESENT_POSITION
        reader = self.sdk.GroupSyncRead(self.port_handler, self.packet_handler, address, size)
        for motor_id in _MOTOR_IDS.values():
            if not reader.addParam(motor_id):
                raise SO101AdapterError(f"failed to prepare position read for motor {motor_id}")
        for attempt in range(_READ_RETRIES + 1):
            result = reader.txRxPacket()
            if result == self.sdk.COMM_SUCCESS:
                break
            if attempt == _READ_RETRIES:
                self._communication_error("read SO-101 positions", result, 0)

        positions: dict[str, float] = {}
        for motor, motor_id in _MOTOR_IDS.items():
            if not reader.isAvailable(motor_id, address, size):
                raise SO101AdapterError(f"position response for motor {motor!r} is incomplete")
            raw = _decode_sign_magnitude(reader.getData(motor_id, address, size), 15)
            positions[f"{motor}.pos"] = self._normalized_position(motor, raw)
        return positions

    def get_observation(self) -> Mapping[str, object]:
        if not self.is_connected:
            raise SO101AdapterError("SO-101 is not connected")
        return self._read_positions()

    def send_action(self, action: Mapping[str, float]) -> None:
        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot send actions")
        if not self.is_connected:
            raise SO101AdapterError("SO-101 is not connected")
        address, size = _GOAL_POSITION
        writer = self.sdk.GroupSyncWrite(self.port_handler, self.packet_handler, address, size)
        for motor, motor_id in _MOTOR_IDS.items():
            raw = self._raw_position(motor, action[f"{motor}.pos"])
            encoded = _encode_sign_magnitude(raw, 15)
            data = [self.sdk.SCS_LOBYTE(encoded), self.sdk.SCS_HIBYTE(encoded)]
            if not writer.addParam(motor_id, data):
                raise SO101AdapterError(f"failed to prepare position command for motor {motor!r}")
        result = writer.txPacket()
        self._communication_error("write SO-101 positions", result, 0)

    def _close_port(self) -> None:
        if getattr(self.port_handler, "is_open", False):
            self.port_handler.closePort()
        self._connected = False
        self._calibrated = False
        self._prepared = False
        self._torque_owned = False

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        try:
            if (
                self.config.disable_torque_on_disconnect
                and not self.read_only
                and (getattr(self, "_torque_owned", False) or self._prepared)
            ):
                self._set_torque(False)
                self._torque_owned = False
        finally:
            self._close_port()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise SO101AdapterError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SO101AdapterError(f"{name} must be numeric") from None
    if not math.isfinite(result):
        raise SO101AdapterError(f"{name} must be finite")
    return result


def _numbers(value: object, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SO101AdapterError(f"{name} must be a sequence")
    if len(value) != length:
        raise SO101AdapterError(f"{name} must contain {length} values")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _clip_step(target: float, present: float, limit: float) -> float:
    delta = target - present
    return present + max(-limit, min(limit, delta))


class SO101Adapter(RobotAdapter):
    """Synchronous adapter for one calibrated SO-101 follower arm."""

    def __init__(
        self,
        config: SO101Config,
        *,
        controller: _SO101Controller | None = None,
        read_only: bool = False,
    ) -> None:
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a boolean")
        if read_only and controller is not None:
            raise ValueError("read-only capture requires the built-in Feetech controller")
        self.config = config
        self.read_only = read_only
        self.robot_id = config.robot_id
        self.controller = controller or _FeetechSO101Controller(config, read_only=read_only)
        self._prepared = False

    def connect(self, *, prepare: bool = True) -> None:
        if not isinstance(prepare, bool):
            raise TypeError("prepare must be a boolean")
        if self.controller.is_connected:
            if prepare and not self.read_only:
                self.prepare()
            return
        try:
            connect = self.controller.connect
            try:
                parameters = inspect.signature(connect).parameters
            except (TypeError, ValueError):
                parameters = {}
            if not prepare and "prepare" not in parameters:
                raise SO101AdapterError("SO-101 controller does not expose a passive connect boundary")
            if "prepare" in parameters:
                connect(calibrate=False, prepare=prepare)
            else:
                connect(calibrate=False)
        except BaseException:
            if self.controller.is_connected:
                self.controller.disconnect()
            raise
        if not self.controller.is_calibrated:
            self.controller.disconnect()
            raise SO101AdapterError("SO-101 motor calibration does not match the configured calibration file")
        self._prepared = prepare and not self.read_only

    def prepare(self) -> None:
        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot prepare motors")
        if self._prepared:
            return
        if not self.controller.is_connected:
            self.connect(prepare=False)
        prepare = getattr(self.controller, "prepare", None)
        if not callable(prepare):
            raise SO101AdapterError("SO-101 controller does not expose an explicit prepare operation")
        prepare()
        self._prepared = True

    def _read_positions(self) -> tuple[float, ...]:
        if not self.controller.is_connected:
            raise SO101AdapterError("SO-101 is not connected")
        raw = self.controller.get_observation()
        if not isinstance(raw, Mapping):
            raise SO101AdapterError("SO-101 observation must be an object")
        missing = set(SO101_POSITION_FEATURES) - raw.keys()
        if missing:
            raise SO101AdapterError(f"SO-101 observation is missing features: {sorted(missing)!r}")
        return tuple(_number(raw[name], name) for name in SO101_POSITION_FEATURES)

    def observe(self) -> RobotObservation:
        positions = self._read_positions()
        captured_timestamp_ns = time.monotonic_ns()
        return RobotObservation(
            timestamp_s=time.time(),
            values={
                "joint_positions_deg": list(positions[:-1]),
                "gripper_position": positions[-1],
            },
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "so101_follower",
                "action_space": SO101_ACTION_SPACE,
                "position_units": "degrees_and_normalized_gripper",
                "captured_timestamp_ns": captured_timestamp_ns,
                "clock_domain": "host_monotonic_ns",
            },
        )

    def execute(self, action: RobotAction) -> None:
        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot execute actions")
        if not self._prepared:
            raise SO101AdapterError("SO-101 motors are not prepared; call prepare explicitly before execute")
        if isinstance(action.values, Mapping) and action.values.get("type") == "stop":
            if action.values != {"type": "stop"}:
                raise SO101AdapterError("stop action must not contain parameters")
            self.stop()
            return
        target_joints, target_gripper = self.validate_action(action)
        current = self._read_positions()
        maximum_joint_step = max(abs(target - present) for target, present in zip(target_joints, current[:-1]))
        gripper_step = abs(target_gripper - current[-1])
        if self.config.step_limit_mode == "reject":
            if maximum_joint_step > self.config.max_joint_step_deg:
                raise SO101AdapterError(
                    f"joint step {maximum_joint_step:.6f} exceeds {self.config.max_joint_step_deg:.6f} degrees"
                )
            if gripper_step > self.config.max_gripper_step:
                raise SO101AdapterError(f"gripper step {gripper_step:.6f} exceeds {self.config.max_gripper_step:.6f}")
        else:
            target_joints = tuple(
                _clip_step(target, present, self.config.max_joint_step_deg)
                for target, present in zip(target_joints, current[:-1])
            )
            target_gripper = _clip_step(
                target_gripper,
                current[-1],
                self.config.max_gripper_step,
            )
        command = dict(
            zip(
                SO101_POSITION_FEATURES,
                (*target_joints, target_gripper),
            )
        )
        self.controller.send_action(command)

    def validate_action(self, action: RobotAction) -> tuple[tuple[float, ...], float]:
        """Validate a position command and read the current pose without writing.

        Dual-arm adapters use this boundary to validate every child before the
        first serial write, so a malformed or over-sized right-arm target cannot
        leave the left arm partially commanded.
        """

        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot execute actions")
        if not self._prepared:
            raise SO101AdapterError("SO-101 motors are not prepared; call prepare explicitly before execute")
        declared_space = action.metadata.get("action_space")
        if declared_space is not None and declared_space != SO101_ACTION_SPACE:
            raise SO101AdapterError(f"unsupported action space {declared_space!r}; expected {SO101_ACTION_SPACE!r}")
        if not isinstance(action.values, Mapping):
            raise SO101AdapterError("SO-101 action values must be an object")
        values = dict(action.values)
        kind = values.pop("type", None)
        if kind != "joint_position":
            raise SO101AdapterError(f"unsupported SO-101 action type: {kind!r}")
        expected_fields = {"joint_positions_deg", "gripper_position"}
        if set(values) != expected_fields:
            missing = expected_fields - values.keys()
            unexpected = values.keys() - expected_fields
            raise SO101AdapterError(
                "SO-101 joint_position fields do not match the contract; "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        target_joints = _numbers(values["joint_positions_deg"], "joint_positions_deg", 5)
        target_gripper = _number(values["gripper_position"], "gripper_position")
        if not 0.0 <= target_gripper <= 100.0:
            raise SO101AdapterError("gripper_position must be in [0, 100]")
        current = self._read_positions()
        maximum_joint_step = max(abs(target - present) for target, present in zip(target_joints, current[:-1]))
        gripper_step = abs(target_gripper - current[-1])
        if self.config.step_limit_mode == "reject":
            if maximum_joint_step > self.config.max_joint_step_deg:
                raise SO101AdapterError(
                    f"joint step {maximum_joint_step:.6f} exceeds {self.config.max_joint_step_deg:.6f} degrees"
                )
            if gripper_step > self.config.max_gripper_step:
                raise SO101AdapterError(f"gripper step {gripper_step:.6f} exceeds {self.config.max_gripper_step:.6f}")
        return target_joints, target_gripper

    def stop(self) -> None:
        """Hold the measured pose; SO-101 exposes no separate stop primitive."""
        if self.read_only:
            raise SO101AdapterError("read-only SO-101 connection cannot command a hold")
        if not self._prepared:
            raise SO101AdapterError("SO-101 motors are not prepared; call prepare explicitly before stop")
        positions = self._read_positions()
        self.controller.send_action(dict(zip(SO101_POSITION_FEATURES, positions)))

    def close(self) -> None:
        if self.controller.is_connected:
            self.controller.disconnect()
        self._prepared = False


__all__ = [
    "SO101_ACTION_SPACE",
    "SO101_JOINTS",
    "SO101_MOTORS",
    "SO101_POSITION_FEATURES",
    "SO101Adapter",
    "SO101AdapterError",
]
