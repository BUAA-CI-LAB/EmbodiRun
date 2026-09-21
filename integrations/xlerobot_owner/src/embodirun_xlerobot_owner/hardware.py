"""Read-only-first synchronous hardware backend for the recovered XLeRobot SDK.

The backend deliberately does not import LeRobot or OpenCV at module import
time.  ``connect`` opens the two Feetech buses with ``handshake=False`` and
only performs reads.  In particular, it never calls the high-level robot
``connect``/``configure``/calibration helpers: those helpers write EEPROM or
control-table values on this SDK.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import inspect
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .motor_diagnostics import read_sts3215_diagnostics, read_sts3215_present_block

try:  # ``fcntl`` is present on the supported Linux/macOS hosts.
    import fcntl
except ImportError:  # pragma: no cover - only relevant to unsupported hosts
    fcntl = None


__all__ = [
    "HardwareDependencyError",
    "HardwareRobot",
    "HardwareRobotError",
    "HardwareSafetyError",
]


MODEL_NUMBER_STS3215 = 777
POSITION_MODE = 0
VELOCITY_MODE = 1
MAX_SIGNED_VELOCITY_RAW = 0x7FFF
MAX_ARM_VELOCITY_RAW = 3400
MAX_ARM_ACCELERATION_RAW = 254
MOTION_PROFILE_WRITE_ATTEMPTS = 3
MOTION_PROFILE_READBACK_DELAY_S = 0.01
WATCHDOG_TIMEOUT_S = 0.35

ARM_JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_NAMES = tuple(f"{side}_arm_{suffix}.pos" for side in ("left", "right") for suffix in ARM_JOINT_SUFFIXES)
ARM_NAMES = {side: tuple(f"{side}_arm_{suffix}" for suffix in ARM_JOINT_SUFFIXES) for side in ("left", "right")}
WHEEL_NAMES = ("base_left_wheel", "base_right_wheel")
HEAD_TILT_NAME = "head_motor_2"
POSITION_FIELDS = (
    "Present_Position",
    "Present_Velocity",
    "Moving",
    "Torque_Enable",
    "Operating_Mode",
    "Homing_Offset",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Goal_Position",
    "Goal_Velocity",
)
PREFLIGHT_FIELDS = (
    "Present_Position",
    "Present_Velocity",
    "Moving",
    "Torque_Enable",
    "Operating_Mode",
    "Homing_Offset",
    "Min_Position_Limit",
    "Max_Position_Limit",
)


# These names are intentionally placeholders.  They remain ``None`` until a
# bus is actually built, so importing this module is dependency-clean.  They
# also give tests an obvious seam when they do not have LeRobot installed.
FeetechMotorsBus: Any = None
Motor: Any = None
MotorCalibration: Any = None
MotorNormMode: Any = None


class HardwareRobotError(RuntimeError):
    """Base error for hardware-backend failures."""


class HardwareDependencyError(HardwareRobotError):
    """The configured optional hardware dependency cannot be loaded."""


class HardwareSafetyError(HardwareRobotError):
    """A safety preflight or motion guard refused an operation."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _read_json_calibration(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read calibration {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TypeError("calibration must be a mapping or JSON path")
    result: dict[str, Any] = dict(value)
    for wrapper in ("motors", "calibration"):
        nested = result.get(wrapper)
        if isinstance(nested, Mapping):
            result = dict(nested)
            break
    # A few LeRobot exports group the same canonical names below side keys.
    for side in ("left", "right"):
        nested = result.get(side)
        if isinstance(nested, Mapping):
            result.update(nested)
    return result


def _normalise_calibration_entry(name: str, value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        source = value
    else:
        source = {
            field: getattr(value, field, None)
            for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max")
        }
    fields: dict[str, int] = {}
    for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
        if field not in source or source[field] is None:
            raise ValueError(f"calibration for {name} is missing {field}")
        numeric = _finite(source[field], f"calibration.{name}.{field}")
        if numeric != int(numeric):
            raise ValueError(f"calibration.{name}.{field} must be an integer")
        fields[field] = int(numeric)
    if fields["id"] <= 0 or fields["id"] > 253:
        raise ValueError(f"calibration.{name}.id is outside the servo ID range")
    if fields["drive_mode"] not in (0, 1):
        raise ValueError(f"calibration.{name}.drive_mode must be 0 or 1")
    if not 0 <= fields["range_min"] < fields["range_max"] <= 4095:
        raise ValueError(f"calibration.{name} has an invalid raw position range")
    return fields


def _copy_result(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - SDK wrapper values may fail arbitrary deepcopy paths
        return value


def _load_sdk(sdk_src: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Load only the requested recovered SDK source, and only on demand."""

    global FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode
    if all(item is not None for item in (FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode)):
        try:
            from lerobot.motors.feetech import OperatingMode
        except ImportError as exc:  # pragma: no cover - only for test seams
            raise HardwareDependencyError("OperatingMode is unavailable in the configured SDK") from exc
        return FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode, OperatingMode

    if not sdk_src:
        raise HardwareDependencyError("sdk_src is required when motor ports are configured")
    source_path = Path(sdk_src).expanduser().resolve()
    if not source_path.is_dir():
        raise HardwareDependencyError(f"SDK source directory does not exist: {source_path}")
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)

    loaded = sys.modules.get("lerobot")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file and source_path not in Path(loaded_file).resolve().parents:
        for module_name in list(sys.modules):
            if module_name == "lerobot" or module_name.startswith("lerobot."):
                del sys.modules[module_name]

    try:
        from lerobot.motors import Motor as sdk_motor
        from lerobot.motors import MotorCalibration as sdk_calibration
        from lerobot.motors import MotorNormMode as sdk_norm_mode
        from lerobot.motors.feetech import FeetechMotorsBus as sdk_bus
        from lerobot.motors.feetech import OperatingMode
    except ImportError as exc:
        raise HardwareDependencyError(f"could not import FeetechMotorsBus from requested SDK {source_path}") from exc

    module = sys.modules.get("lerobot")
    module_file = getattr(module, "__file__", None)
    if not module_file or source_path not in Path(module_file).resolve().parents:
        raise HardwareDependencyError(f"lerobot resolved outside sdk_src: {module_file or '<unknown>'}")
    FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode = (
        sdk_bus,
        sdk_motor,
        sdk_calibration,
        sdk_norm_mode,
    )
    return FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode, OperatingMode


class _PortLock:
    """One non-blocking advisory lock for one serial-device path."""

    def __init__(self, port: str):
        if fcntl is None:
            raise HardwareDependencyError("fcntl is required for serial-port ownership locking")
        self.port = port
        encoded_path = os.path.realpath(port).encode("utf-8").hex()
        lock_root = Path(tempfile.gettempdir()) / "embodied-runtime-xlerobot-locks"
        components = [encoded_path[index : index + 64] for index in range(0, len(encoded_path), 64)]
        self.path = lock_root.joinpath(*components[:-1], f"{components[-1]}.lock")
        self.fd: int | None = None

    def acquire(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise HardwareRobotError(
                    f"serial port {self.port!r} is owned by another hardware robot instance"
                ) from exc
            raise HardwareRobotError(f"cannot acquire serial-port lock for {self.port!r}: {exc}") from exc

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


class _CameraWorker:
    """Bounded latest-frame capture; the frame is kept as original JPEG bytes."""

    def __init__(self, name: str, path: Any, stale_after_s: float):
        self.name = name
        self.path = path
        self.stale_after_ns = int(stale_after_s * 1e9)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: bytes | None = None
        self._timestamp_ns: int | None = None
        self._capture_mono_ns: int | None = None
        self._error: str | None = None
        self._error_timestamp_ns: int | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"xlerobot-camera-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._error_timestamp_ns = time.time_ns()

    def _run(self) -> None:
        capture: Any = None
        try:
            import cv2  # lazy optional dependency

            while not self._stop.is_set():
                try:
                    if capture is None:
                        # Reopen the stable by-path link, not an obsolete fd
                        # or a cached /dev/video number after USB reconnect.
                        capture = (
                            cv2.VideoCapture(self.path, cv2.CAP_V4L2)
                            if str(self.path).startswith("/dev/")
                            else cv2.VideoCapture(self.path)
                        )
                        if capture is None or not capture.isOpened():
                            raise RuntimeError(f"could not open {self.path!r}")
                        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        capture.set(cv2.CAP_PROP_FPS, 25)
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    capture_started_ns = time.monotonic_ns()
                    capture_started_wall_ns = time.time_ns()
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError("capture failed; reconnecting")
                    encoded_ok, encoded = cv2.imencode(".jpg", frame)
                    if not encoded_ok:
                        raise RuntimeError("JPEG encoding failed")
                    payload = encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)
                    with self._lock:
                        self._frame = payload
                        self._timestamp_ns = capture_started_wall_ns
                        self._capture_mono_ns = capture_started_ns
                        self._error = None
                        self._error_timestamp_ns = None
                except Exception as exc:  # noqa: BLE001 -- camera failure must not stop the other views
                    self._set_error(f"camera {self.name!r}: {type(exc).__name__}: {exc}")
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception as release_exc:  # noqa: BLE001 -- still discard the obsolete handle
                            self._set_error(f"camera {self.name!r} release failed: {release_exc}")
                        capture = None
                    if self._stop.wait(0.5):
                        break
                    continue
                self._stop.wait(0.01)
        except Exception as exc:  # noqa: BLE001 - camera backends fail with varied exceptions
            self._set_error(f"camera {self.name!r}: {type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:  # noqa: BLE001 - driver cleanup must remain bounded
                    self._set_error(f"camera {self.name!r} release failed: {exc}")

    def snapshot(self, *, with_age: bool = False):
        now_ns = time.time_ns()
        with self._lock:
            frame = self._frame
            timestamp_ns = self._timestamp_ns
            error = self._error
            error_timestamp_ns = self._error_timestamp_ns
            age_ns = None if self._capture_mono_ns is None else time.monotonic_ns() - self._capture_mono_ns
        stale = timestamp_ns is None or now_ns - timestamp_ns > self.stale_after_ns
        if stale:
            error = error or f"camera {self.name!r} frame is stale or unavailable"
        result = (frame if not stale else None, timestamp_ns, error, error_timestamp_ns, not stale)
        return (*result, age_ns) if with_age else result

    def close(self, timeout_s: float) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            return not self._thread.is_alive()
        return True


class HardwareRobot:
    """Synchronous read-only-first XLeRobot hardware boundary."""

    mode = "hardware"

    def __init__(self, config: dict):
        if not isinstance(config, Mapping):
            raise TypeError("HardwareRobot config must be a mapping")
        self.config = dict(config)
        self.mode = "hardware"
        self.armed = False

        enabled = self.config.get("enabled_arms", ["left", "right"])
        if not isinstance(enabled, (list, tuple, set)):
            raise TypeError("enabled_arms must be a list containing left and/or right")
        enabled_set = set(enabled)
        if not enabled_set.issubset({"left", "right"}):
            raise ValueError("enabled_arms may only contain left and right")
        self.enabled_arms = tuple(side for side in ("left", "right") if side in enabled_set)
        self.enable_head_tilt = bool(self.config.get("enable_head_tilt", False))
        if self.enable_head_tilt and not self.config.get("enable_base", False):
            raise ValueError("enable_head_tilt requires the base control scope")
        self._max_head_speed = _positive(self.config.get("max_head_speed_deg_s", 5), "max_head_speed_deg_s")
        if self._max_head_speed > 15:
            raise ValueError("max_head_speed_deg_s exceeds the camera profile's 15 deg/s limit")
        if type(self.config.get("head_tilt_up_sign", 1)) is not int or self.config.get("head_tilt_up_sign", 1) not in (
            -1,
            1,
        ):
            raise ValueError("head_tilt_up_sign must be -1 or 1")
        self._last_head_target_mono = 0.0

        ports = self.config.get("ports", {})
        if not isinstance(ports, Mapping):
            raise TypeError("ports must be a mapping")
        self._ports = {side: ports.get(side) for side in ("left", "right") if ports.get(side) not in (None, "")}
        cameras = self.config.get("cameras")
        if cameras is None:
            cameras = dict.fromkeys(("front", "left_wrist", "right_wrist"))
        if not isinstance(cameras, Mapping):
            raise TypeError("cameras must be a mapping of camera name to path")
        self._camera_paths = dict(cameras)
        if any(not isinstance(name, str) or not name.strip() for name in self._camera_paths):
            raise ValueError("camera names must be non-empty strings")

        self._calibration_error: str | None = None
        try:
            raw_calibration = _read_json_calibration(self.config.get("calibration"))
        except (TypeError, ValueError) as exc:
            raw_calibration = {}
            self._calibration_error = str(exc)
        self._calibration: dict[str, dict[str, int]] = {}
        self._calibration_errors: dict[str, str] = {}
        for name, value in raw_calibration.items():
            if not isinstance(name, str):
                continue
            motor_name = name.removesuffix(".pos")
            try:
                self._calibration[motor_name] = _normalise_calibration_entry(motor_name, value)
            except (TypeError, ValueError) as exc:
                self._calibration_errors[motor_name] = str(exc)

        self._allow_motion = bool(self.config.get("allow_motion", False))
        self._arm_velocity_raw: int | None = None
        self._arm_acceleration_raw: int | None = None
        self._motion_config_error: str | None = None
        if "arm_velocity_raw" not in self.config or self.config.get("arm_velocity_raw") is None:
            self._motion_config_error = "arm_velocity_raw must be explicitly configured before motion"
        else:
            try:
                requested = _finite(self.config.get("arm_velocity_raw"), "arm_velocity_raw")
                if requested <= 0:
                    raise ValueError("arm_velocity_raw must be greater than zero")
                if requested != int(requested) or requested > MAX_ARM_VELOCITY_RAW:
                    raise ValueError(f"arm_velocity_raw must be an integer within [1, {MAX_ARM_VELOCITY_RAW}]")
                self._arm_velocity_raw = int(requested)
            except (TypeError, ValueError) as exc:
                self._motion_config_error = str(exc)
        try:
            acceleration = _finite(
                self.config.get("arm_acceleration_raw", MAX_ARM_ACCELERATION_RAW),
                "arm_acceleration_raw",
            )
            if acceleration != int(acceleration) or not 0 <= acceleration <= MAX_ARM_ACCELERATION_RAW:
                raise ValueError(f"arm_acceleration_raw must be an integer within [0, {MAX_ARM_ACCELERATION_RAW}]")
            self._arm_acceleration_raw = int(acceleration)
        except (TypeError, ValueError) as exc:
            self._motion_config_error = str(exc)

        self._wheel_radius = _positive(self.config.get("wheel_radius", 0.05), "wheel_radius")
        self._wheelbase = _positive(self.config.get("wheelbase", 0.25), "wheelbase")
        self._max_joint_speed = _positive(self.config.get("max_joint_speed_deg_s", 15), "max_joint_speed_deg_s")
        self._max_gripper_speed = _positive(self.config.get("max_gripper_speed_pct_s", 25), "max_gripper_speed_pct_s")
        feedback_tolerance = _finite(
            self.config.get("position_feedback_tolerance_raw", 0),
            "position_feedback_tolerance_raw",
        )
        if feedback_tolerance != int(feedback_tolerance) or not 0 <= feedback_tolerance <= 64:
            raise ValueError("position_feedback_tolerance_raw must be an integer within [0, 64]")
        self._position_feedback_tolerance_raw = int(feedback_tolerance)
        self._max_linear = _positive(self.config.get("max_linear_m_s", 0.05), "max_linear_m_s")
        self._max_angular = _positive(self.config.get("max_angular_deg_s", 10), "max_angular_deg_s")
        self._read_interval_s = _finite(self.config.get("read_interval_s", 0.02), "read_interval_s")
        self._camera_stale_s = _positive(self.config.get("camera_stale_after_s", 0.5), "camera_stale_after_s")
        self._stop_timeout_s = _positive(self.config.get("stop_timeout_s", 0.4), "stop_timeout_s")
        self._stop_poll_s = _positive(self.config.get("stop_poll_interval_s", 0.02), "stop_poll_interval_s")
        self._cleanup_timeout_s = _positive(self.config.get("cleanup_timeout_s", 0.5), "cleanup_timeout_s")
        if self._read_interval_s < 0:
            raise ValueError("read_interval_s must not be negative")

        self._base_directions = self._parse_base_directions()
        self.metadata = {
            "source": "physical",
            "joint_unit": "degrees",
            "gripper_unit": "range_0_100",
            "joint_names": list(JOINT_NAMES) + ([HEAD_TILT_NAME + ".pos"] if self.enable_head_tilt else []),
            "joint_limits": self._joint_limits_metadata(),
            "camera_roles_confirmed": bool(self.config.get("camera_roles_confirmed", False)),
            "camera_names": list(self._camera_paths),
            "enabled_arms": list(self.enabled_arms),
            "enable_base": bool(self.config.get("enable_base", False)),
            "allow_motion": self._allow_motion,
            "arm_velocity_raw_configured": self._arm_velocity_raw is not None and self._motion_config_error is None,
            "arm_velocity_raw": {
                "configured": self._arm_velocity_raw is not None and self._motion_config_error is None,
                "value": self._arm_velocity_raw,
            },
            "arm_acceleration_raw": {
                "configured": self._arm_acceleration_raw is not None and self._motion_config_error is None,
                "value": self._arm_acceleration_raw,
            },
            "read_only_startup": True,
            "watchdog_timeout_s": WATCHDOG_TIMEOUT_S,
            "base_velocity_limits": {"linear_m_s": self._max_linear, "angular_deg_s": self._max_angular},
            "head_tilt": (
                {
                    "joint": HEAD_TILT_NAME + ".pos",
                    "scope": "base",
                    "max_speed_deg_s": self._max_head_speed,
                    "up_sign": self.config.get("head_tilt_up_sign", 1),
                }
                if self.enable_head_tilt
                else None
            ),
            "position_feedback_tolerance_raw": self._position_feedback_tolerance_raw,
        }

        self._io_lock = threading.RLock()
        self._buses: dict[str, Any] = {}
        self.buses = self._buses
        self._port_locks: dict[str, _PortLock] = {}
        self._bus_factory = self.config.get("_bus_factory", self.config.get("bus_factory"))
        self._connected = False
        self._closed = False
        self._stopped_by_us = False
        # A requested fault handoff starts blocked, even before connect/validation.
        self._inherited_stop_fault = "_resume_stop_fault" in self.config
        self._stop_uncertain = self._inherited_stop_fault
        self._fault_handoff: dict | None = None
        self._torque_ownership_uncertain = False
        self._owned_torque_names: set[str] = set()
        self._active_scopes: set[str] = set()
        self._stopped_scopes: set[str] = set()
        self._scope_command_mono: dict[str, float] = {}
        self.metadata["control_scopes"] = ["arms", "base"]
        self._held_raw: dict[str, int] = {}
        self._last_gripper_goal_raw: dict[str, int] = {}
        self._last_command_targets: dict[str, float] = {}
        self._last_target_mono = 0.0
        self._last_command_mono = 0.0
        self._last_stop: dict[str, Any] | None = None
        self._last_audit: dict[str, Any] = {}
        self._last_preflight: dict[str, Any] = {}
        self._cached_state: dict[str, Any] | None = None
        self._cached_raw: dict[str, Any] | None = None
        self._cached_wheel_present_blocks: dict[str, Any] = {}
        self._cached_errors: list[str] = []
        self._cached_state_timestamp_ns: int | None = None
        self._last_state_read_mono = float("-inf")

        self._camera_workers: dict[str, _CameraWorker] = {}
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        return self._connected and not self._closed

    @property
    def stop_unconfirmed(self) -> bool:
        return self._stop_uncertain

    @property
    def inherited_stop_feedback(self) -> dict | None:
        return _copy_result(self._fault_handoff["failed_stop"]) if self._fault_handoff else None

    def safety_state(self) -> dict:
        """Read-only in-memory ownership/fault evidence, not a stop confirmation."""
        with self._io_lock:
            return {
                "stop_unconfirmed": self._stop_uncertain,
                "torque_ownership_uncertain": self._torque_ownership_uncertain,
                "owned_motors": sorted(self._owned_torque_names),
                "held_goals": dict(self._held_raw),
                "gripper_goals": dict(self._last_gripper_goal_raw),
            }

    def _parse_base_directions(self) -> dict[str, int] | None:
        if not bool(self.config.get("enable_base", False)):
            return None
        raw = None
        for key in ("wheel_directions", "base_directions", "wheel_direction"):
            if key in self.config:
                raw = self.config[key]
                break
        if not isinstance(raw, Mapping):
            return None
        result: dict[str, int] = {}
        aliases = {
            "base_left_wheel": ("base_left_wheel", "left"),
            "base_right_wheel": ("base_right_wheel", "right"),
        }
        for wheel, names in aliases.items():
            found = next((raw[name] for name in names if name in raw), None)
            if isinstance(found, bool):
                return None
            try:
                direction = int(_finite(found, f"{wheel} direction"))
            except (TypeError, ValueError):
                return None
            if direction not in (-1, 1):
                return None
            result[wheel] = direction
        return result

    def _joint_limits_metadata(self) -> dict[str, list[float] | list[int] | None]:
        result: dict[str, list[float] | list[int] | None] = {}
        for side in ("left", "right"):
            for suffix, name in zip(ARM_JOINT_SUFFIXES, ARM_NAMES[side], strict=True):
                canonical = f"{name}.pos"
                calibration = self._calibration.get(name)
                if suffix == "gripper":
                    result[canonical] = [0, 100]
                elif calibration is None:
                    result[canonical] = None
                else:
                    mid = (calibration["range_min"] + calibration["range_max"]) / 2
                    result[canonical] = [
                        (calibration["range_min"] - mid) * 360.0 / 4095.0,
                        (calibration["range_max"] - mid) * 360.0 / 4095.0,
                    ]
        if self.enable_head_tilt:
            calibration = self._calibration.get(HEAD_TILT_NAME)
            half_range = (calibration["range_max"] - calibration["range_min"]) * 180 / 4095 if calibration else None
            result[HEAD_TILT_NAME + ".pos"] = [-half_range, half_range] if half_range is not None else None
        return result

    def _motor_specs(self, side: str, *, include_base: bool = False) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for motor_id, suffix in enumerate(ARM_JOINT_SUFFIXES, start=1):
            name = f"{side}_arm_{suffix}"
            specs[name] = {
                "id": motor_id,
                "norm_mode": "RANGE_0_100" if suffix == "gripper" else "DEGREES",
            }
        if side == "right" and include_base:
            specs["base_left_wheel"] = {"id": 9, "norm_mode": "RANGE_M100_100"}
            specs["base_right_wheel"] = {"id": 10, "norm_mode": "RANGE_M100_100"}
        if side == "left" and self.enable_head_tilt:
            specs[HEAD_TILT_NAME] = {"id": 8, "norm_mode": "DEGREES"}
        return specs

    def _motor_names_for_control(self, scope: str = "all") -> tuple[str, ...]:
        if scope not in ("all", "arms", "base"):
            raise ValueError("scope must be all, arms or base")
        names = () if scope == "base" else tuple(name for side in self.enabled_arms for name in ARM_NAMES[side])
        if scope != "arms" and bool(self.config.get("enable_base", False)):
            names += WHEEL_NAMES
            if self.enable_head_tilt:
                names += (HEAD_TILT_NAME,)
        return names

    @staticmethod
    def _motor_scope(name: str) -> str:
        return "base" if name in (*WHEEL_NAMES, HEAD_TILT_NAME) else "arms"

    @staticmethod
    def _motor_side(name: str) -> str:
        return "left" if name.startswith("left_") or name == HEAD_TILT_NAME else "right"

    def control_state(self) -> dict[str, bool]:
        with self._io_lock:
            return {scope: scope in self._active_scopes for scope in ("arms", "base")}

    def _build_bus(self, side: str, port: Any) -> Any:
        specs = self._motor_specs(side, include_base=bool(self.config.get("enable_base", False)))
        if callable(self._bus_factory):
            values = {
                "side": side,
                "port": port,
                "motors": specs,
                "motor_specs": specs,
                "calibration": {name: self._calibration[name] for name in specs if name in self._calibration},
            }
            try:
                signature = inspect.signature(self._bus_factory)
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
                )
                kwargs = (
                    values
                    if accepts_kwargs
                    else {name: values[name] for name in signature.parameters if name in values}
                )
                return self._bus_factory(**kwargs)
            except (TypeError, ValueError):
                return self._bus_factory(side, port, specs, values["calibration"])

        sdk_bus, sdk_motor, sdk_calibration, sdk_norm_mode, _ = _load_sdk(self.config.get("sdk_src"))
        motor_objects: dict[str, Any] = {}
        calibration_objects: dict[str, Any] = {}
        for name, spec in specs.items():
            norm_mode = getattr(sdk_norm_mode, spec["norm_mode"])
            motor_objects[name] = sdk_motor(spec["id"], "sts3215", norm_mode)
            calibration = self._calibration.get(name)
            if calibration is not None:
                calibration_objects[name] = sdk_calibration(**calibration)
        return sdk_bus(
            port=str(port),
            motors=motor_objects,
            calibration=calibration_objects,
        )

    def _acquire_port_locks(self) -> None:
        paths: dict[str, str] = {}
        for side, port in self._ports.items():
            canonical = os.path.realpath(str(port))
            if canonical in paths.values():
                raise HardwareRobotError(f"left and right motor buses cannot share serial port {port!r}")
            paths[side] = canonical
        acquired: list[_PortLock] = []
        try:
            for side, port in self._ports.items():
                lock = _PortLock(str(port))
                lock.acquire()
                self._port_locks[side] = lock
                acquired.append(lock)
        except Exception:
            for lock in acquired:
                lock.release()
            self._port_locks.clear()
            raise

    def _release_port_locks(self) -> None:
        for lock in tuple(self._port_locks.values()):
            with contextlib.suppress(OSError):
                lock.release()
        self._port_locks.clear()

    def _configure_local_bus(self, bus: Any) -> None:
        timeout_ms = self.config.get("sdk_timeout_ms")
        if timeout_ms is not None and callable(getattr(bus, "set_timeout", None)):
            bus.set_timeout(int(_positive(timeout_ms, "sdk_timeout_ms")))
        if callable(getattr(bus, "set_baudrate", None)):
            default_baudrate = getattr(bus, "default_baudrate", None)
            if default_baudrate is not None:
                bus.set_baudrate(default_baudrate)

    def connect(self) -> dict[str, Any]:
        """Connect cameras and/or motor buses without any register writes."""

        with self._io_lock:
            if self._closed:
                raise HardwareRobotError("hardware robot is closed")
            if self._connected:
                return {
                    "connected": True,
                    "read_only_startup": True,
                    "motor_sides": list(self._buses),
                    "camera_names": list(self._camera_paths),
                }
            self._acquire_port_locks()
            try:
                for side in ("left", "right"):
                    port = self._ports.get(side)
                    if port in (None, ""):
                        continue
                    bus = self._build_bus(side, port)
                    # This is the only SDK connect path: no Robot.connect call.
                    bus.connect(handshake=False)
                    self._configure_local_bus(bus)
                    self._buses[side] = bus
                self._connected = True
                self._audit_all()
                if self._inherited_stop_fault:
                    if "_resume_held_state" in self.config:
                        raise HardwareSafetyError("held and fault handoffs are mutually exclusive")
                    self.restore_stop_fault(self.config.pop("_resume_stop_fault"))
                resume = self.config.pop("_resume_held_state", None)
                if resume is not None:
                    self.metadata["holding_resume"] = self.restore_held_state(resume)
                self._start_cameras()
                self._start_watchdog()
                return {
                    "connected": True,
                    "read_only_startup": True,
                    "motor_sides": list(self._buses),
                    "camera_names": list(self._camera_paths),
                    "audit": _copy_result(self._last_audit),
                }
            except Exception:
                self._disconnect_buses_read_only()
                self._release_port_locks()
                self._connected = False
                raise

    def _read_register(self, bus: Any, field: str, name: str) -> tuple[Any, str | None]:
        try:
            return bus.read(field, name, normalize=False), None
        except Exception as exc:  # noqa: BLE001 - report arbitrary SDK read failures
            return None, f"{type(exc).__name__}: {exc}"

    def read_motor_diagnostics(self, name: str) -> dict[str, Any]:
        """Idle-only single-wheel diagnostic; cannot arm, stop or clear faults.

        Reject live scopes before serial traffic: even read-only SDK timeouts
        must not add a long critical section to a live motion watchdog.
        """
        with self._io_lock:
            if self._closed or not self._connected:
                raise HardwareRobotError("hardware robot is not connected")
            if self.armed or self._active_scopes:
                raise HardwareSafetyError("motor diagnostics require all control scopes inactive")
            if name not in WHEEL_NAMES or not self.config.get("enable_base", False):
                raise ValueError("motor diagnostics accept only enabled base wheels")
            bus = self._buses.get("right")
            if bus is None or not getattr(bus, "is_connected", False):
                raise HardwareRobotError("wheel motor port is not connected")
            spec = self._motor_specs("right", include_base=True)[name]
            motor = bus.motors.get(name)
            actual_id = motor.get("id") if isinstance(motor, Mapping) else getattr(motor, "id", None)
            if type(actual_id) is not int or actual_id != spec["id"]:
                raise HardwareSafetyError("diagnostic target ID does not match configured wheel")
            packet = getattr(bus, "packet_handler", None)
            port = getattr(bus, "port_handler", None)
            if port is None or not callable(getattr(packet, "readTxRx", None)):
                raise HardwareDependencyError("single-device diagnostic READ is unavailable")
            result = read_sts3215_diagnostics(packet, port, actual_id)
            return {**result, "motor": name, "motor_id": actual_id, "port": str(self._ports["right"])}

    def _snapshot_motor(self, side: str, bus: Any, name: str) -> dict[str, Any]:
        spec = self._motor_specs(side, include_base=True)[name]
        record: dict[str, Any] = {"id": spec["id"], "fields": {}, "errors": {}}
        try:
            record["model_number"] = bus.ping(spec["id"], num_retry=0, raise_on_error=False)
        except Exception as exc:  # noqa: BLE001 - report arbitrary SDK ping failures
            record["model_number"] = None
            record["errors"]["Model_Number"] = f"{type(exc).__name__}: {exc}"
        block_fields = set()
        packet = getattr(bus, "packet_handler", None)
        port = getattr(bus, "port_handler", None)
        if name in WHEEL_NAMES and port is not None and callable(getattr(packet, "readTxRx", None)):
            # The production STS adapter supports a single reply for position,
            # velocity and Moving. Never replace failed block data with a later
            # scalar read: that could turn a transport error into apparent zero.
            block_fields = {"Present_Position", "Present_Velocity", "Moving"}
            block = read_sts3215_present_block(packet, port, spec["id"])
            record["present_block"] = block
            for field in block_fields:
                if block["ok"]:
                    record["fields"][field] = block["fields"][field]
                else:
                    record["errors"][field] = "; ".join(block["errors"])
        for field in POSITION_FIELDS:
            if field in block_fields:
                continue
            value, error = self._read_register(bus, field, name)
            if error is None:
                record["fields"][field] = value
            else:
                record["errors"][field] = error
        return record

    def _audit_all(self) -> None:
        audit: dict[str, Any] = {"read_only": True, "buses": {}, "errors": []}
        for side, bus in self._buses.items():
            names = tuple(self._motor_specs(side, include_base=bool(self.config.get("enable_base"))))
            records = {name: self._snapshot_motor(side, bus, name) for name in names}
            audit["buses"][side] = records
        self._last_audit = audit

    def _wait_stopped_wheel(self, bus: Any, record: dict) -> list[str]:
        """Bounded read-only settling; every invalid reply is a hard failure."""
        trace = {
            "read_only": True,
            "required_consecutive_zero": 3,
            "settled": False,
            "samples": [record["present_block"]],
        }
        record["stationary_wait"] = trace
        consecutive = 0
        # At most 40 additional reads and 1.6 s of spacing. Never use during
        # active control: this holds the serial lock shared with the watchdog.
        for _ in range(40):
            time.sleep(0.04)
            block = read_sts3215_present_block(bus.packet_handler, bus.port_handler, record["id"])
            trace["samples"].append(block)
            if not block["ok"]:
                return ["wheel stationary wait transport/device failure: " + "; ".join(block["errors"])]
            fields = block["fields"]
            consecutive = consecutive + 1 if fields["Present_Velocity"] == 0 and fields["Moving"] == 0 else 0
            if consecutive == 3:
                trace["settled"] = True
                record["present_block"] = block
                record["fields"].update({k: fields[k] for k in ("Present_Position", "Present_Velocity", "Moving")})
                return []
        return ["wheel did not provide three consecutive strictly stationary samples"]

    def _preflight(
        self,
        expected_hold: dict[str, int] | None = None,
        *,
        scope: str = "all",
        expected_zero_wheels: dict[str, int] | None = None,
    ) -> tuple[dict[str, int], dict[str, dict[str, Any]], list[str]]:
        positions: dict[str, int] = {}
        records: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        if self._calibration_error:
            errors.append(self._calibration_error)
        errors.extend(self._calibration_errors.values())
        selected_names = self._motor_names_for_control(scope)
        if not selected_names:
            errors.append("no enabled arm or base target")
        if any(name in WHEEL_NAMES for name in selected_names) and self._base_directions is None:
            errors.append("enable_base requires explicit wheel_directions of -1 or 1")

        selected = [(self._motor_side(name), name) for name in selected_names]
        for side, name in selected:
            prior_error_count = len(errors)
            bus = self._buses.get(side)
            if bus is None:
                errors.append(f"{name}: motor port for {side} is not connected")
                continue
            record = self._snapshot_motor(side, bus, name)
            records[name] = record
            expected_id = self._motor_specs(side, include_base=True)[name]["id"]
            if record.get("model_number") != MODEL_NUMBER_STS3215:
                errors.append(f"{name}: model ID {record.get('model_number')!r}, expected {MODEL_NUMBER_STS3215}")
            calibration = self._calibration.get(name)
            if calibration is None:
                errors.append(f"{name}: saved calibration is missing")
            elif calibration["id"] != expected_id:
                errors.append(f"{name}: saved calibration ID {calibration['id']} does not match expected {expected_id}")
            fields = record["fields"]
            field_errors = record["errors"]
            required_field_errors = {
                field: message for field, message in field_errors.items() if field in PREFLIGHT_FIELDS
            }
            if required_field_errors:
                errors.extend(f"{name}.{field}: {message}" for field, message in required_field_errors.items())
                continue
            if calibration is not None:
                mismatches = [
                    ("Homing_Offset", "homing_offset"),
                    ("Min_Position_Limit", "range_min"),
                    ("Max_Position_Limit", "range_max"),
                ]
                different = [
                    f"{field} actual={fields[field]!r} expected={calibration[key]!r}"
                    for field, key in mismatches
                    if fields.get(field) != calibration[key]
                ]
                if different:
                    errors.append(f"{name}: EEPROM mismatch ({'; '.join(different)})")
            expected_mode = VELOCITY_MODE if name in WHEEL_NAMES else POSITION_MODE
            if fields.get("Operating_Mode") != expected_mode:
                mode_name = "velocity" if expected_mode == VELOCITY_MODE else "position"
                errors.append(f"{name}: Operating_Mode is not expected {mode_name} mode {expected_mode}")
            torque = fields.get("Torque_Enable")
            internally_owned = (
                name in self._owned_torque_names
                and (self._stopped_by_us or self._motor_scope(name) in self._stopped_scopes)
                and not self._stop_uncertain
            )
            if expected_hold is not None and name in expected_hold:
                expected = expected_hold.get(name)
                if fields.get("Goal_Position") != expected or "Goal_Position" in field_errors:
                    errors.append(f"{name}: held restart Goal_Position did not match")
                if torque != 1:
                    errors.append(f"{name}: held restart requires existing torque=1")
                if (
                    calibration is None
                    or expected is None
                    or not calibration["range_min"] <= expected <= calibration["range_max"]
                ):
                    errors.append(f"{name}: held restart target outside calibration")
                internally_owned = fields.get("Goal_Position") == expected and torque == 1
            if expected_zero_wheels is not None and name in expected_zero_wheels:
                if (
                    fields.get("Goal_Velocity") != 0
                    or "Goal_Velocity" in field_errors
                    or torque != expected_zero_wheels[name]
                ):
                    errors.append(f"{name}: stopped wheel handoff torque/zero goal changed")
                else:
                    internally_owned = True
            if torque not in (0, 1):
                errors.append(f"{name}: invalid Torque_Enable {torque!r}")
            elif torque == 1 and not internally_owned:
                errors.append(f"{name}: torque is already on unexpectedly")
            cold_wheel = (
                name in WHEEL_NAMES
                and torque == 0
                and fields.get("Goal_Velocity") == 0
                and name not in self._owned_torque_names
            )
            owned_stopped_wheel = internally_owned and torque == 1
            zero_goal_wheel = fields.get("Goal_Velocity") == 0
            if (
                name in WHEEL_NAMES
                and record.get("present_block", {}).get("ok")
                and (fields.get("Present_Velocity") != 0 or fields.get("Moving") != 0)
                and zero_goal_wheel
                and (owned_stopped_wheel or cold_wheel)
                and "Goal_Velocity" not in field_errors
                and not self._active_scopes
                and not self._stop_uncertain
                and not self._torque_ownership_uncertain
                and len(errors) == prior_error_count
            ):
                errors.extend(f"{name}: {error}" for error in self._wait_stopped_wheel(bus, record))
            try:
                present_position = int(fields["Present_Position"])
                present_velocity = int(fields["Present_Velocity"])
                moving = int(fields["Moving"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{name}: invalid present state: {exc}")
                continue
            if moving != 0 or present_velocity != 0:
                errors.append(
                    f"{name}: motor is moving during preflight (velocity={present_velocity}, moving={moving})"
                )
            if (
                name not in WHEEL_NAMES
                and calibration is not None
                and not (
                    calibration["range_min"] - self._position_feedback_tolerance_raw
                    <= present_position
                    <= calibration["range_max"] + self._position_feedback_tolerance_raw
                )
            ):
                errors.append(
                    f"{name}: present position {present_position} is outside saved range "
                    f"[{calibration['range_min']}, {calibration['range_max']}] plus "
                    f"feedback tolerance {self._position_feedback_tolerance_raw}"
                )
            positions[name] = present_position
        self._last_preflight = {
            "positions": dict(positions),
            "records": _copy_result(records),
            "errors": list(errors),
            "read_only": True,
        }
        return positions, records, errors

    def _validate_prior_idle(self, evidence: dict) -> None:
        """Require a fresh same-boot stopped service, never infer this from torque."""
        if not isinstance(evidence, dict) or evidence.get("source") != "physical":
            raise ValueError("prior idle service evidence required")
        if evidence.get("boot_id") != self._boot_id():
            raise ValueError("prior idle evidence belongs to another boot")
        age = time.monotonic() - _finite(evidence.get("created_monotonic_s"), "prior idle time")
        if not 0 <= age <= 120:
            raise ValueError("prior idle service evidence is expired")
        for key in ("status_before", "status_after"):
            status = evidence.get(key, {})
            confirmed = status.get("feedback")
            if confirmed is None:
                # No commands since a successful read-only held restart.
                confirmed = status.get("metadata", {}).get("holding_resume", {})
                if confirmed.get("restored") is not True or confirmed.get("register_writes") != 0:
                    raise ValueError("prior service has no confirmed stop or held restore")
            if (
                status.get("connected") is not True
                or status.get("armed") is not False
                or status.get("control_owner") is not None
                or status.get("control_state") != {"arms": False, "base": False}
                or status.get("stop_unconfirmed") is not False
                or status.get("recording") is not False
                or status.get("error")
                or status.get("observation_errors")
                or confirmed.get("stop_confirmed") is not True
                or confirmed.get("errors")
            ):
                raise ValueError("prior service is not unowned with confirmed stop")
        samples = evidence.get("samples", [])
        if not isinstance(samples, list) or len(samples) < 3:
            raise ValueError("three fresh stationary observations required")
        previous = -1
        for sample in samples:
            stamp = sample.get("state_timestamp_ns")
            raw = sample.get("raw", {})
            if (
                type(stamp) is not int
                or stamp <= previous
                or sample.get("state_cached") is not False
                or sample.get("errors")
                or set(raw) != set(self._motor_names_for_control("all"))
            ):
                raise ValueError("prior service observations incomplete or stale")
            if any(fields.get("Present_Velocity") != 0 or fields.get("Moving") != 0 for fields in raw.values()):
                raise ValueError("prior service did not observe strict stationary feedback")
            previous = stamp

    @staticmethod
    def _boot_id() -> str:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def restore_stop_fault(self, snapshot: dict) -> None:
        """Read-only adoption of a failed all-stop; never run startup stop/arm.

        This intentionally supports only a fully owned, powered robot whose
        last all-stop accepted every hold/zero write. Partial ownership or
        missing provenance requires separate recovery, not guessed goals.
        """
        from .stop_fault_handoff import validate_snapshot

        with self._io_lock:
            if (
                not self._inherited_stop_fault
                or not self._stop_uncertain
                or not self.connected
                or self.armed
                or self._owned_torque_names
            ):
                raise HardwareSafetyError("fault restore requires a new blocked connected instance")
            state = validate_snapshot(self, snapshot)
            if self._calibration_error or self._calibration_errors or self._motion_config_error:
                raise HardwareSafetyError("fault handoff configuration/calibration invalid")
            current = {}
            for name in self._motor_names_for_control():
                record = self._last_audit["buses"][self._motor_side(name)][name]
                fields, calibration = record["fields"], self._calibration.get(name)
                if (
                    record["errors"]
                    or record.get("model_number") != MODEL_NUMBER_STS3215
                    or calibration is None
                    or calibration["id"] != record["id"]
                ):
                    raise HardwareSafetyError(f"{name}: invalid fault handoff device/calibration read")
                expected = {
                    "Torque_Enable": 1,
                    "Operating_Mode": VELOCITY_MODE if name in WHEEL_NAMES else POSITION_MODE,
                    "Homing_Offset": calibration["homing_offset"],
                    "Min_Position_Limit": calibration["range_min"],
                    "Max_Position_Limit": calibration["range_max"],
                }
                expected["Goal_Velocity" if name in WHEEL_NAMES else "Goal_Position"] = (
                    0 if name in WHEEL_NAMES else state["held_goals"][name]
                )
                if any(type(fields.get(k)) is not int or fields[k] != v for k, v in expected.items()):
                    raise HardwareSafetyError(f"{name}: fault handoff mode/torque/goal/calibration changed")
                position = fields.get("Present_Position")
                if type(position) is not int or not 0 <= position <= 4095:
                    raise HardwareSafetyError(f"{name}: invalid current position")
                if name not in WHEEL_NAMES:
                    goal = state["held_goals"][name]
                    if not calibration["range_min"] <= goal <= calibration["range_max"]:
                        raise HardwareSafetyError(f"{name}: inherited goal outside calibration")
                    if not (
                        calibration["range_min"] - self._position_feedback_tolerance_raw
                        <= position
                        <= calibration["range_max"] + self._position_feedback_tolerance_raw
                    ):
                        raise HardwareSafetyError(f"{name}: current position outside calibration")
                # Nonzero velocity/Moving remain evidence of the inherited fault.
                if any(type(fields.get(k)) is not int for k in ("Present_Velocity", "Moving")):
                    raise HardwareSafetyError(f"{name}: invalid current motion feedback")
                current[name] = dict(fields)
            self._owned_torque_names = set(state["owned_motors"])
            self._held_raw = dict(state["held_goals"])
            self._last_gripper_goal_raw = dict(state["gripper_goals"])
            self._torque_ownership_uncertain = state["torque_ownership_uncertain"]
            self._fault_handoff = _copy_result(snapshot)
            self._last_stop = _copy_result(snapshot["failed_stop"])
            self.metadata["stop_fault_resume"] = {
                "restored": True,
                "armed": False,
                "register_writes": 0,
                "inherited_stop_unconfirmed": True,
                "initial_readback": current,
                "inherited_safety": self.safety_state(),
            }

    def restore_held_state(self, snapshot: dict) -> dict:
        """Explicit, fresh, stopped restart handoff; reads only, never auto-arms.

        Normal startup still rejects unexpected torque. This one-use CLI path
        also requires matching ports, joint set, goals, EEPROM and two fresh
        stationary samples before recognizing previously held torque as ours.
        """
        with self._io_lock:
            result = {"restored": False, "armed": False, "register_writes": 0, "errors": []}
            try:
                if (
                    self.armed
                    or self._owned_torque_names
                    or not self._connected
                    or self._stop_uncertain
                    or self._torque_ownership_uncertain
                ):
                    raise ValueError("held restore requires a new connected, unarmed instance")
                if not isinstance(snapshot, dict) or snapshot.get("stop_confirmed") is not True:
                    raise ValueError("confirmed stopped snapshot required")
                age = time.monotonic() - _finite(snapshot.get("created_monotonic_s"), "snapshot time")
                if not 0 <= age <= 120:
                    raise ValueError("held restart snapshot is expired or from another boot")
                if snapshot.get("source") != "physical" or snapshot.get("ports") != self._ports:
                    raise ValueError("held restart source/ports mismatch")
                if snapshot.get("enable_base") is not False:
                    raise ValueError("held restart must not resume active base control")
                wheels = snapshot.get("wheel_torque")
                if wheels is not None:
                    if (
                        not self.config.get("enable_base")
                        or not isinstance(wheels, dict)
                        or set(wheels) != set(WHEEL_NAMES)
                        or any(type(t) is not int or t not in (0, 1) for t in wheels.values())
                    ):
                        raise ValueError("wheel handoff must cover both configured wheels")
                    self._validate_prior_idle(snapshot.get("prior_idle"))
                names = tuple(n for n in self._motor_names_for_control("all") if n not in WHEEL_NAMES)
                goals = snapshot.get("goals")
                cold = snapshot.get("cold_motors", [])
                if (
                    not isinstance(cold, list)
                    or not all(isinstance(n, str) for n in cold)
                    or len(set(cold)) != len(cold)
                ):
                    raise ValueError("cold_motors must list unique joint names")
                if not isinstance(goals, dict) or set(goals) & set(cold) or set(goals) | set(cold) != set(names):
                    raise ValueError("held restart must cover exactly the controlled joints")
                if any(type(v) is not int or not 0 <= v <= 4095 for v in goals.values()):
                    raise ValueError("held restart goals must be integer raw positions")
                _, records, errors = self._preflight(expected_hold=goals, scope="all", expected_zero_wheels=wheels)
                result["stationary_wait"] = {
                    n: r["stationary_wait"] for n, r in records.items() if "stationary_wait" in r
                }
                if self.config.get("enable_base") and wheels is None:
                    # Adding previously unowned wheels is a read-only cold
                    # start, never an adoption of an active base controller.
                    for name, record in records.items():
                        if name not in WHEEL_NAMES:
                            continue
                        fields = record["fields"]
                        if fields.get("Torque_Enable") != 0 or fields.get("Goal_Velocity") != 0:
                            errors.append(f"{name}: held arm restart requires wheels torque-off with zero goal")
                if errors:
                    raise ValueError("; ".join(errors))
                stop = self._stop_locked("verified held restart", write_commands=False, scope="all")
                if not stop["stop_confirmed"]:
                    raise ValueError("held restart stationary feedback not confirmed")
                self._held_raw = dict(goals)
                self._last_gripper_goal_raw = {n: v for n, v in goals.items() if n.endswith("_gripper")}
                self._owned_torque_names.update(goals)
                if wheels is not None:
                    self._owned_torque_names.update(n for n, torque in wheels.items() if torque == 1)
                self._stopped_scopes.update(self._motor_scope(n) for n in names)
                if wheels is not None:
                    self._stopped_scopes.add("base")
                self._torque_ownership_uncertain = False
                self._stopped_by_us = True
                result.update(restored=True, stop_confirmed=True, motors=list(goals), cold_motors=cold)
                if wheels is not None:
                    result["wheel_torque"] = dict(wheels)
            except (TypeError, ValueError, KeyError) as exc:
                result["errors"] = [str(exc)]
            return result

    def _write(
        self,
        bus: Any,
        field: str,
        name: str,
        value: int,
        writes: list[dict[str, Any]],
    ) -> None:
        entry = {"field": field, "motor": name, "value": value, "status": "attempted"}
        writes.append(entry)
        bus.write(field, name, value, normalize=False)
        entry["status"] = "accepted"

    def _write_and_confirm_motion_profile(
        self,
        bus: Any,
        field: str,
        name: str,
        value: int,
        writes: list[dict[str, Any]],
    ) -> None:
        """Retry an idempotent volatile profile write until its readback matches."""

        actual: Any = None
        read_error: str | None = None
        for attempt in range(1, MOTION_PROFILE_WRITE_ATTEMPTS + 1):
            self._write(bus, field, name, value, writes)
            actual, read_error = self._read_register(bus, field, name)
            if read_error is None:
                try:
                    if int(actual) == value:
                        return
                except (TypeError, ValueError):
                    pass
            if attempt < MOTION_PROFILE_WRITE_ATTEMPTS:
                time.sleep(MOTION_PROFILE_READBACK_DELAY_S)
        detail = f"actual={actual!r}"
        if read_error is not None:
            detail += f", read_error={read_error}"
        raise HardwareSafetyError(
            f"{name}: {field} readback did not confirm {value} after "
            f"{MOTION_PROFILE_WRITE_ATTEMPTS} attempts ({detail})"
        )

    def _bus_for_name(self, name: str) -> Any:
        return self._buses[self._motor_side(name)]

    def _controlled_names_by_side(self, scope: str = "all") -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for name in self._motor_names_for_control(scope):
            side = self._motor_side(name)
            result[side] = result.get(side, ()) + (name,)
        return result

    def _best_effort_hold_locked(
        self,
        fallback_positions: Mapping[str, int],
        writes: list[dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        for side, names in self._controlled_names_by_side().items():
            bus = self._buses.get(side)
            if bus is None:
                continue
            for name in names:
                if name in WHEEL_NAMES:
                    try:
                        self._write(bus, "Goal_Velocity", name, 0, writes)
                    except Exception as exc:  # noqa: BLE001 - rollback must catch SDK failures
                        errors.append(f"{name}: rollback zero velocity failed: {exc}")
                    continue
                current, error = self._read_register(bus, "Present_Position", name)
                try:
                    hold = int(current) if error is None else int(fallback_positions[name])
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{name}: no safe rollback position: {exc}")
                    continue
                if name.endswith("_gripper"):
                    hold = self._last_gripper_goal_raw.get(name, hold)
                try:
                    self._write(bus, "Goal_Position", name, hold, writes)
                except Exception as exc:  # noqa: BLE001 - rollback must catch SDK failures
                    errors.append(f"{name}: rollback hold failed: {exc}")
        return errors

    def arm(self, scope: str = "all") -> dict[str, Any]:
        """Run a fresh read-only preflight, then explicitly enable selected joints."""

        with self._io_lock:
            controlled = self._motor_names_for_control(scope)
            requested = {self._motor_scope(n) for n in controlled}
            selected_arms = self.enabled_arms if scope != "base" else ()
            include_base = "base" in requested
            base = {
                "armed": bool(requested) and requested <= self._active_scopes,
                "status": "refused",
                "writes": [],
                "errors": [],
                "physical_outcome": "unknown",
            }
            if self._closed:
                base["errors"] = ["hardware robot is closed"]
                return base
            if not self._connected:
                base["errors"] = ["hardware robot is not connected"]
                return base
            if requested and requested <= self._active_scopes:
                base.update({"status": "already_armed", "armed": True})
                return base
            if requested & self._active_scopes:
                base["errors"] = ["requested control overlaps an active scope"]
                return base
            if not self._allow_motion:
                base["errors"] = ["allow_motion is false; motion remains disabled"]
                return base
            if self._motion_config_error:
                base["errors"] = [self._motion_config_error]
                return base
            if self._stop_uncertain:
                base["errors"] = ["previous stop was not confirmed stationary; explicit external recovery is required"]
                return base
            if self._torque_ownership_uncertain:
                base["errors"] = [
                    "torque ownership is uncertain after a partial enable; explicit external recovery is required"
                ]
                return base

            positions, records, errors = self._preflight(scope=scope)
            base["stationary_wait"] = {n: r["stationary_wait"] for n, r in records.items() if "stationary_wait" in r}
            if errors:
                base["errors"] = errors
                base["preflight"] = _copy_result(self._last_preflight)
                return base

            hold_positions = dict(positions)
            for side in selected_arms:
                for name in ARM_NAMES[side]:
                    calibration = self._calibration[name]
                    hold_positions[name] = max(
                        calibration["range_min"],
                        min(calibration["range_max"], positions[name]),
                    )
            if HEAD_TILT_NAME in controlled:
                calibration = self._calibration[HEAD_TILT_NAME]
                prior_hold = self._held_raw.get(HEAD_TILT_NAME, positions[HEAD_TILT_NAME])
                hold_positions[HEAD_TILT_NAME] = max(
                    calibration["range_min"], min(calibration["range_max"], prior_hold)
                )

            writes: list[dict[str, Any]] = []
            base["writes"] = writes
            torque_attempted = False
            try:
                for side in selected_arms:
                    for name in ARM_NAMES[side]:
                        self._write_and_confirm_motion_profile(
                            self._buses[side],
                            "Acceleration",
                            name,
                            self._arm_acceleration_raw or 0,
                            writes,
                        )
                        self._held_raw[name] = hold_positions[name]
                        if name.endswith("_gripper"):
                            self._last_gripper_goal_raw[name] = hold_positions[name]
                        self._write(
                            self._buses[side],
                            "Goal_Position",
                            name,
                            hold_positions[name],
                            writes,
                        )
                for side in selected_arms:
                    for name in ARM_NAMES[side]:
                        self._write_and_confirm_motion_profile(
                            self._buses[side],
                            "Goal_Velocity",
                            name,
                            self._arm_velocity_raw or 1,
                            writes,
                        )
                if include_base:
                    bus = self._buses["right"]
                    for name in WHEEL_NAMES:
                        self._write(bus, "Goal_Velocity", name, 0, writes)
                if HEAD_TILT_NAME in controlled:
                    # Install a current-position hold before enabling the camera servo.
                    bus = self._bus_for_name(HEAD_TILT_NAME)
                    self._write_and_confirm_motion_profile(bus, "Goal_Velocity", HEAD_TILT_NAME, 64, writes)
                    self._held_raw[HEAD_TILT_NAME] = hold_positions[HEAD_TILT_NAME]
                    self._write(bus, "Goal_Position", HEAD_TILT_NAME, hold_positions[HEAD_TILT_NAME], writes)
                for name in controlled:
                    # The SDK write may reach the servo and then raise (for
                    # example, on a lost reply). Treat the ownership as
                    # uncertain before attempting the write and fail closed.
                    torque_attempted = True
                    self._owned_torque_names.add(name)
                    self._write(self._bus_for_name(name), "Torque_Enable", name, 1, writes)
                for name in controlled:
                    value, error = self._read_register(self._bus_for_name(name), "Torque_Enable", name)
                    if error is not None or int(value) != 1:
                        raise HardwareSafetyError(f"{name}: Torque_Enable readback did not confirm enabled")
            except Exception as exc:  # noqa: BLE001 - any SDK write/read failure fails closed
                rollback_errors = self._best_effort_hold_locked({**self._held_raw, **hold_positions}, writes)
                self.armed = False
                self._active_scopes.clear()
                self._scope_command_mono.clear()
                self._stopped_by_us = False
                self._torque_ownership_uncertain = bool(torque_attempted)
                self._stop_uncertain = bool(torque_attempted or self._owned_torque_names)
                base.update(
                    {
                        "status": "error",
                        "armed": False,
                        "errors": [f"arm write failed: {type(exc).__name__}: {exc}", *rollback_errors],
                        "physical_outcome": "unknown",
                    }
                )
                return base

            self._active_scopes.update(requested)
            self._stopped_scopes.difference_update(requested)
            self.armed = bool(self._active_scopes)
            self._stopped_by_us = False
            self._stop_uncertain = False
            self._torque_ownership_uncertain = False
            self._last_command_targets.update(
                {
                    name: self._raw_to_joint_value(name, hold_positions[name])
                    for name in controlled
                    if name not in WHEEL_NAMES
                }
            )
            if selected_arms:
                self._last_target_mono = time.monotonic()
            if HEAD_TILT_NAME in controlled:
                self._last_head_target_mono = time.monotonic()
            self._last_command_mono = time.monotonic()
            self._scope_command_mono.update(dict.fromkeys(requested, self._last_command_mono))
            base.update(
                {
                    "status": "armed",
                    "armed": True,
                    "held_positions_raw": dict(hold_positions),
                    "held_positions": {
                        name + ".pos": self._raw_to_joint_value(name, value)
                        for name, value in hold_positions.items()
                        if name not in WHEEL_NAMES
                    },
                    "goal_velocity_raw": self._arm_velocity_raw,
                    "goal_acceleration_raw": self._arm_acceleration_raw,
                }
            )
            return base

    def _action_to_raw(self, name: str, value: Any) -> int:
        numeric = _finite(value, name)
        calibration = self._calibration.get(name)
        if calibration is None:
            raise HardwareSafetyError(f"{name}: saved calibration is missing")
        minimum, maximum = calibration["range_min"], calibration["range_max"]
        if name.endswith("_gripper"):
            if not 0 <= numeric <= 100:
                raise HardwareSafetyError(f"{name}: gripper value must be within [0, 100]")
            normalised = 100.0 - numeric if calibration["drive_mode"] else numeric
            raw = int((normalised / 100.0) * (maximum - minimum) + minimum)
        else:
            mid = (minimum + maximum) / 2.0
            raw = int((numeric * 4095.0 / 360.0) + mid)
        if not 0 <= raw <= 4095 or not minimum <= raw <= maximum:
            raise HardwareSafetyError(
                f"{name}: target raw position {raw} is outside saved range [{minimum}, {maximum}]"
            )
        return raw

    def _bound_position_rate(
        self,
        raw_positions: dict[str, int],
        applied: dict[str, float],
        now_mono: float,
    ) -> list[str]:
        """Bound valid targets against the previous target and elapsed time.

        Present_Position is deliberately read on every position command to
        validate fresh feedback and the saved range.  It is not used as the
        target reference: a gripper holding an object may legitimately lag its
        last goal.  Unchanged gripper goals therefore pass explicitly.
        """

        if not raw_positions:
            return []
        elapsed = max(0.0, now_mono - self._last_target_mono)
        errors: list[str] = []
        for name in tuple(raw_positions):
            bus = self._bus_for_name(name)
            present, read_error = self._read_register(bus, "Present_Position", name)
            if read_error is not None:
                errors.append(f"{name}: fresh Present_Position is unavailable: {read_error}")
                continue
            try:
                present_raw = int(present)
                calibration = self._calibration[name]
                if not (
                    calibration["range_min"] - self._position_feedback_tolerance_raw
                    <= present_raw
                    <= calibration["range_max"] + self._position_feedback_tolerance_raw
                ):
                    raise HardwareSafetyError(
                        f"{name}: present position {present_raw} is outside saved range "
                        f"[{calibration['range_min']}, {calibration['range_max']}] plus "
                        f"feedback tolerance {self._position_feedback_tolerance_raw}"
                    )
                actual = self._raw_to_joint_value(name, present_raw)
                target = float(applied[f"{name}.pos"])
                previous = self._last_command_targets.get(name, actual)
                speed = (
                    self._max_head_speed
                    if name == HEAD_TILT_NAME
                    else (self._max_gripper_speed if name.endswith("_gripper") else self._max_joint_speed)
                )
                target_elapsed = max(0, now_mono - self._last_head_target_mono) if name == HEAD_TILT_NAME else elapsed
                maximum_delta = speed * target_elapsed
                if name.endswith("_gripper") and math.isclose(target, previous, rel_tol=0.0, abs_tol=1e-6):
                    bounded = target
                else:
                    bounded = max(
                        previous - maximum_delta,
                        min(previous + maximum_delta, target),
                    )
                if bounded != target:
                    applied[f"{name}.pos"] = bounded
                    raw_positions[name] = self._action_to_raw(name, bounded)
            except (KeyError, TypeError, ValueError, HardwareSafetyError) as exc:
                errors.append(f"{name}: cannot rate-check target: {exc}")
        return errors

    def _body_to_wheel_raw(self, x: float, theta: float) -> dict[str, int]:
        theta_rad = theta * math.pi / 180.0
        left_speed = (x - theta_rad * self._wheelbase / 2.0) / self._wheel_radius
        right_speed = (x + theta_rad * self._wheelbase / 2.0) / self._wheel_radius
        raw_values = {
            "base_left_wheel": round(left_speed * 180.0 / math.pi * 4096.0 / 360.0),
            "base_right_wheel": round(right_speed * 180.0 / math.pi * 4096.0 / 360.0),
        }
        largest = max(abs(value) for value in raw_values.values())
        if largest > 3000:
            scale = 3000.0 / largest
            raw_values = {name: round(value * scale) for name, value in raw_values.items()}
        directions = self._base_directions or {}
        return {name: value * directions.get(name, 1) for name, value in raw_values.items()}

    def _wheel_raw_to_body(self, left_raw: int, right_raw: int) -> dict[str, float]:
        directions = self._base_directions or {}
        left_raw *= directions.get("base_left_wheel", 1)
        right_raw *= directions.get("base_right_wheel", 1)
        left_linear = (left_raw * 360.0 / 4096.0) * math.pi / 180.0 * self._wheel_radius
        right_linear = (right_raw * 360.0 / 4096.0) * math.pi / 180.0 * self._wheel_radius
        return {
            "x.vel": (left_linear + right_linear) / 2.0,
            "theta.vel": ((right_linear - left_linear) / self._wheelbase) * 180.0 / math.pi,
        }

    def _command_rejected(self, errors: list[str]) -> dict[str, Any]:
        return {
            "accepted": False,
            "command_accepted": False,
            "status": "rejected",
            "applied_action": {},
            "sent_timestamp_ns": time.time_ns(),
            "physical_outcome": "unknown",
            "errors": errors,
            "writes": [],
        }

    def command(self, action: dict) -> dict[str, Any]:
        """Validate a named absolute action completely before issuing any write."""

        with self._io_lock:
            if self._closed:
                return self._command_rejected(["hardware robot is closed"])
            if not self.armed:
                return self._command_rejected(["hardware robot is not armed"])
            if not isinstance(action, Mapping) or not action:
                return self._command_rejected(["action must be a non-empty mapping"])

            position_names = (*JOINT_NAMES, HEAD_TILT_NAME + ".pos") if self.enable_head_tilt else JOINT_NAMES
            allowed = set(position_names) | {"x.vel", "theta.vel"}
            unknown = sorted(set(action).difference(allowed))
            if unknown:
                return self._command_rejected([f"unknown action keys: {unknown}"])
            errors: list[str] = []
            position_keys = [name for name in position_names if name in action]
            for side in ("left", "right"):
                side_keys = [name for name in position_keys if name.startswith(f"{side}_arm_")]
                if not side_keys:
                    continue
                if side not in self.enabled_arms:
                    errors.append(f"{side} arm is not enabled")
                missing = [f"{name}.pos" for name in ARM_NAMES[side] if f"{name}.pos" not in action]
                if missing:
                    errors.append(f"{side} arm action is missing keys: {missing}")
            has_base = "x.vel" in action or "theta.vel" in action
            command_scopes = {self._motor_scope(name.removesuffix(".pos")) for name in position_keys}
            if has_base:
                command_scopes.add("base")
            if not command_scopes <= self._active_scopes:
                errors.append("action includes a control scope that is not armed")
            if has_base:
                if not bool(self.config.get("enable_base", False)):
                    errors.append("base motion is disabled")
                if "x.vel" not in action or "theta.vel" not in action:
                    errors.append("base action requires both x.vel and theta.vel")
                if self._base_directions is None:
                    errors.append("base action requires explicit wheel_directions")

            raw_positions: dict[str, int] = {}
            applied: dict[str, float] = {}
            for name in position_keys:
                try:
                    raw_positions[name.removesuffix(".pos")] = self._action_to_raw(
                        name.removesuffix(".pos"), action[name]
                    )
                    applied[name] = _finite(action[name], name)
                except (TypeError, ValueError, HardwareSafetyError) as exc:
                    errors.append(str(exc))

            command_start_mono = time.monotonic()
            if not errors:
                errors.extend(self._bound_position_rate(raw_positions, applied, command_start_mono))

            wheel_raw: dict[str, int] = {}
            if has_base and not errors:
                try:
                    x = max(-self._max_linear, min(self._max_linear, _finite(action["x.vel"], "x.vel")))
                    theta = max(
                        -self._max_angular,
                        min(self._max_angular, _finite(action["theta.vel"], "theta.vel")),
                    )
                    applied["x.vel"], applied["theta.vel"] = x, theta
                    wheel_raw = self._body_to_wheel_raw(x, theta)
                except (TypeError, ValueError, HardwareSafetyError) as exc:
                    errors.append(str(exc))
            if errors:
                return self._command_rejected(errors)

            writes: list[dict[str, Any]] = []
            try:
                for side in self.enabled_arms:
                    bus = self._buses[side]
                    for name in ARM_NAMES[side]:
                        if name in raw_positions:
                            self._write(bus, "Goal_Position", name, raw_positions[name], writes)
                if HEAD_TILT_NAME in raw_positions:
                    self._write(
                        self._bus_for_name(HEAD_TILT_NAME),
                        "Goal_Position",
                        HEAD_TILT_NAME,
                        raw_positions[HEAD_TILT_NAME],
                        writes,
                    )
                if wheel_raw:
                    bus = self._buses["right"]
                    for name, value in wheel_raw.items():
                        self._write(bus, "Goal_Velocity", name, value, writes)
            except Exception as exc:  # noqa: BLE001 - any SDK write failure fails closed
                rollback_errors = self._best_effort_hold_locked(self._held_raw, writes)
                self.armed = False
                self._active_scopes.clear()
                self._scope_command_mono.clear()
                self._stop_uncertain = bool(self._owned_torque_names)
                return {
                    "accepted": False,
                    "command_accepted": False,
                    "status": "error",
                    "applied_action": dict(applied),
                    "sent_timestamp_ns": time.time_ns(),
                    "physical_outcome": "unknown",
                    "errors": [f"command write failed: {type(exc).__name__}: {exc}", *rollback_errors],
                    "writes": writes,
                }

            for name, raw in raw_positions.items():
                self._held_raw[name] = raw
                if name.endswith("_gripper"):
                    self._last_gripper_goal_raw[name] = raw
                self._last_command_targets[name] = applied[f"{name}.pos"]
            if set(raw_positions) - {HEAD_TILT_NAME}:
                self._last_target_mono = command_start_mono
            if HEAD_TILT_NAME in raw_positions:
                self._last_head_target_mono = command_start_mono
            self._last_command_mono = time.monotonic()
            for scope in command_scopes:
                self._scope_command_mono[scope] = self._last_command_mono
            return {
                "accepted": True,
                "command_accepted": True,
                "status": "accepted",
                "applied_action": dict(applied),
                "raw_action": {**raw_positions, **wheel_raw},
                "sent_timestamp_ns": time.time_ns(),
                "physical_outcome": "unknown",
                "errors": [],
                "writes": writes,
            }

    def _stop_locked(
        self, reason: str, *, write_commands: bool = True, scope: str = "all", recover_inherited_fault: bool = False
    ) -> dict[str, Any]:
        controlled = self._motor_names_for_control(scope)
        scopes = {self._motor_scope(n) for n in controlled}
        self._active_scopes.difference_update(scopes)
        for selected in scopes:
            self._scope_command_mono.pop(selected, None)
        self.armed = bool(self._active_scopes)
        if not self.armed:
            self._last_command_mono = 0.0
        if "arms" in scopes:
            self._last_target_mono = 0.0
        writes: list[dict[str, Any]] = []
        errors: list[str] = []
        fallback = dict(self._held_raw)
        feedback_available = True
        for side, names in self._controlled_names_by_side(scope).items():
            bus = self._buses.get(side)
            if bus is None:
                errors.append(f"{side}: motor bus is unavailable")
                continue
            for name in names:
                if not write_commands:
                    continue
                if name in WHEEL_NAMES:
                    try:
                        self._write(bus, "Goal_Velocity", name, 0, writes)
                    except Exception as exc:  # noqa: BLE001 - stop must report SDK write failures
                        errors.append(f"{name}: stop zero write failed: {exc}")
                    continue
                present, read_error = self._read_register(bus, "Present_Position", name)
                if read_error is not None:
                    feedback_available = False
                    errors.append(f"{name}.Present_Position: stop feedback unavailable: {read_error}")
                try:
                    current = int(present) if read_error is None else int(fallback[name])
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{name}: stop has no usable current position: {exc}")
                    continue
                if name.endswith("_gripper"):
                    hold = self._last_gripper_goal_raw.get(name, current)
                else:
                    calibration = self._calibration.get(name)
                    hold = (
                        max(
                            calibration["range_min"],
                            min(calibration["range_max"], current),
                        )
                        if calibration is not None
                        else current
                    )
                try:
                    self._write(bus, "Goal_Position", name, hold, writes)
                    self._held_raw[name] = hold
                except Exception as exc:  # noqa: BLE001 - stop must report SDK write failures
                    errors.append(f"{name}: stop hold write failed: {exc}")

        consecutive = 0
        stationary = False
        samples: list[dict[str, Any]] = []
        confirmation_timeout_s = self._stop_timeout_s
        if (
            write_commands
            and not errors
            and not self._active_scopes
            and reason in ("operator", "operator_retry")
            and any(name in WHEEL_NAMES for name in controlled)
        ):
            confirmation_timeout_s = 1.5
        deadline = time.monotonic() + confirmation_timeout_s
        while time.monotonic() <= deadline:
            sample: dict[str, Any] = {"timestamp_ns": time.time_ns(), "motors": {}}
            sample_ok = True
            for name in controlled:
                bus = self._bus_for_name(name)
                packet, port = getattr(bus, "packet_handler", None), getattr(bus, "port_handler", None)
                if name in WHEEL_NAMES and port is not None and callable(getattr(packet, "readTxRx", None)):
                    block = read_sts3215_present_block(
                        packet, port, self._motor_specs("right", include_base=True)[name]["id"]
                    )
                    sample.setdefault("wheel_present_blocks", {})[name] = block
                    velocity = block["fields"].get("Present_Velocity")
                    moving = block["fields"].get("Moving")
                    velocity_error = moving_error = None if block["ok"] else "; ".join(block["errors"])
                else:
                    velocity, velocity_error = self._read_register(bus, "Present_Velocity", name)
                    moving, moving_error = self._read_register(bus, "Moving", name)
                sample["motors"][name] = {
                    "Present_Velocity": velocity,
                    "Moving": moving,
                    "errors": [error for error in (velocity_error, moving_error) if error],
                }
                if velocity_error or moving_error:
                    sample_ok = False
                    feedback_available = False
                else:
                    try:
                        if int(velocity) != 0 or int(moving) != 0:
                            sample_ok = False
                    except (TypeError, ValueError):
                        sample_ok = False
                        feedback_available = False
            samples.append(sample)
            if sample_ok and controlled:
                consecutive += 1
                if consecutive >= 2:
                    stationary = True
                    break
            else:
                consecutive = 0
            time.sleep(min(self._stop_poll_s, max(0.0, deadline - time.monotonic())))

        writes_succeeded = (not write_commands and not writes) or (
            write_commands and bool(writes) and all(entry.get("status") == "accepted" for entry in writes)
        )
        stop_confirmed = bool(controlled and stationary and feedback_available and not errors and writes_succeeded)
        command_accepted = bool(write_commands and not errors)
        if (
            self._inherited_stop_fault
            and recover_inherited_fault
            and scope == "all"
            and write_commands
            and reason in ("operator", "operator_retry")
            and stop_confirmed
        ):
            self._inherited_stop_fault = False
        self._stop_uncertain = (
            self._inherited_stop_fault or (self._stop_uncertain and scope != "all") or not stop_confirmed
        )
        self._stopped_by_us = stop_confirmed and not self.armed
        if stop_confirmed:
            self._stopped_scopes.update(scopes)
        self._last_stop = {
            "reason": reason,
            "command_accepted": command_accepted,
            "stationary_confirmed": stationary,
            "stop_confirmed": stop_confirmed,
            "feedback_available": feedback_available,
            "physical_outcome": "stopped" if stop_confirmed else "unknown",
            "samples": samples,
        }
        status = "confirmed" if stop_confirmed else "unknown" if not errors else "error"
        return {
            "status": status,
            "scope": scope,
            "reason": reason,
            "accepted": command_accepted,
            "command_accepted": command_accepted,
            "stationary_confirmed": stationary,
            "stop_confirmed": stop_confirmed,
            "observed_stationary": stationary,
            "feedback_available": feedback_available,
            "physical_outcome": "stopped" if stop_confirmed else "unknown",
            "sent_timestamp_ns": time.time_ns(),
            "errors": errors,
            "writes": writes,
            "samples": samples,
        }

    def stop_all(self) -> dict[str, Any]:
        """Explicit production all-stop; the only inherited-stop-fault recovery."""
        return self.stop("all", _recover_inherited_fault=True)

    def stop(self, scope: str = "all", *, _recover_inherited_fault: bool = False) -> dict[str, Any]:
        with self._io_lock:
            controlled = self._motor_names_for_control(scope)
            if self._closed:
                return {
                    "status": "error",
                    "accepted": False,
                    "command_accepted": False,
                    "stationary_confirmed": False,
                    "stop_confirmed": False,
                    "physical_outcome": "unknown",
                    "errors": ["hardware robot is closed"],
                    "writes": [],
                }
            if not self._connected:
                return {
                    "status": "error",
                    "accepted": False,
                    "command_accepted": False,
                    "stationary_confirmed": False,
                    "stop_confirmed": False,
                    "physical_outcome": "unknown",
                    "errors": ["hardware robot is not connected"],
                    "writes": [],
                }
            owned = bool(set(controlled) & self._owned_torque_names)
            active = any(self._motor_scope(n) in self._active_scopes for n in controlled)
            reason = "operator" if active else "operator_retry" if owned else "no_control"
            result = self._stop_locked(
                reason, write_commands=owned, scope=scope, recover_inherited_fault=_recover_inherited_fault
            )
            if not result["stop_confirmed"] and self._active_scopes:
                result["whole_robot_fallback"] = self._stop_locked("scoped stop unconfirmed")
            return result

    def _start_cameras(self) -> None:
        for name, path in self._camera_paths.items():
            if path in (None, ""):
                continue
            worker = _CameraWorker(name, path, self._camera_stale_s)
            worker.start()
            self._camera_workers[name] = worker

    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="xlerobot-hardware-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(0.025):
            with self._io_lock:
                if self._closed:
                    return
                expired = [
                    scope
                    for scope in self._active_scopes
                    if time.monotonic() - self._scope_command_mono.get(scope, 0) > WATCHDOG_TIMEOUT_S
                ]
                for scope in expired:
                    try:
                        self._last_stop = self._stop_locked("watchdog", scope=scope)
                        if not self._last_stop["stop_confirmed"] and self._active_scopes:
                            self._last_stop = self._stop_locked("watchdog stop unconfirmed")
                    except Exception as exc:  # noqa: BLE001 - watchdog must fail closed
                        self.armed = False
                        self._active_scopes.clear()
                        self._scope_command_mono.clear()
                        self._stop_uncertain = True
                        self._last_stop = {
                            "reason": "watchdog",
                            "status": "error",
                            "command_accepted": False,
                            "stationary_confirmed": False,
                            "physical_outcome": "unknown",
                            "errors": [f"watchdog stop failed: {type(exc).__name__}: {exc}"],
                        }

    def read(self) -> tuple[dict[str, Any], dict[str, bytes]]:
        """Return a current bounded observation and only fresh camera JPEGs."""

        with self._io_lock:
            if self._closed:
                raise HardwareRobotError("hardware robot is closed")
            if not self._connected:
                raise HardwareRobotError("hardware robot is not connected")
            source_timestamp_ns = time.time_ns()
            now_mono = time.monotonic()
            if self._cached_state is not None and now_mono - self._last_state_read_mono < self._read_interval_s:
                state = _copy_result(self._cached_state)
                raw = _copy_result(self._cached_raw or {})
                wheel_present_blocks = _copy_result(self._cached_wheel_present_blocks)
                errors = list(self._cached_errors)
                state_timestamp_ns = self._cached_state_timestamp_ns
                state_cached = True
            else:
                state: dict[str, Any] = {}
                raw = {}
                wheel_present_blocks = {}
                errors = []
                state_timestamp_ns: int | None = None
                motor_timestamp_ns = time.time_ns()
                wheel_values: dict[str, int] = {}
                for side, bus in self._buses.items():
                    specs = self._motor_specs(side, include_base=bool(self.config.get("enable_base")))
                    for name in specs:
                        if name in WHEEL_NAMES:
                            fields = ("Present_Velocity", "Moving", "Present_Position")
                        else:
                            fields = ("Present_Position", "Present_Velocity", "Moving")
                        raw[name] = {}
                        packet = getattr(bus, "packet_handler", None)
                        port = getattr(bus, "port_handler", None)
                        block = None
                        if name in WHEEL_NAMES and port is not None and callable(getattr(packet, "readTxRx", None)):
                            # Same validated single reply as wheel preflight.
                            # No settling/retry here: reads also run while
                            # control is active and share the watchdog lock.
                            block = read_sts3215_present_block(packet, port, specs[name]["id"])
                            wheel_present_blocks[name] = block
                        for field in fields:
                            if block is not None:
                                value = block["fields"].get(field)
                                error = None if block["ok"] else "; ".join(block["errors"])
                            else:
                                value, error = self._read_register(bus, field, name)
                            if error is not None:
                                raw[name][field] = {"error": error}
                                errors.append(f"{name}.{field}: {error}")
                            else:
                                raw[name][field] = value
                        present = raw[name].get("Present_Position")
                        if isinstance(present, Mapping):
                            present = None
                        if name not in WHEEL_NAMES and present is not None:
                            calibration = self._calibration.get(name)
                            if calibration is None:
                                errors.append(f"{name}: saved calibration is missing")
                            else:
                                try:
                                    present_raw = int(present)
                                    state[name + ".pos"] = self._raw_to_joint_value(name, present_raw)
                                    if not (
                                        calibration["range_min"] - self._position_feedback_tolerance_raw
                                        <= present_raw
                                        <= calibration["range_max"] + self._position_feedback_tolerance_raw
                                    ):
                                        errors.append(
                                            f"{name}: present position {present_raw} is outside saved range "
                                            f"[{calibration['range_min']}, {calibration['range_max']}] plus "
                                            f"feedback tolerance {self._position_feedback_tolerance_raw}"
                                        )
                                    state_timestamp_ns = motor_timestamp_ns
                                except (TypeError, ValueError, HardwareSafetyError) as exc:
                                    errors.append(f"{name}: cannot convert present position: {exc}")
                        elif name in WHEEL_NAMES:
                            velocity = raw[name].get("Present_Velocity")
                            if not isinstance(velocity, Mapping) and velocity is not None:
                                try:
                                    wheel_values[name] = int(velocity)
                                except (TypeError, ValueError):
                                    errors.append(f"{name}: invalid Present_Velocity {velocity!r}")
                if bool(self.config.get("enable_base", False)):
                    if set(wheel_values) == set(WHEEL_NAMES) and self._base_directions is not None:
                        state.update(
                            self._wheel_raw_to_body(wheel_values["base_left_wheel"], wheel_values["base_right_wheel"])
                        )
                        state_timestamp_ns = state_timestamp_ns or motor_timestamp_ns
                    elif self._base_directions is None:
                        errors.append("base velocity cannot be measured without explicit wheel_directions")
                self._cached_state = _copy_result(state)
                self._cached_raw = _copy_result(raw)
                self._cached_wheel_present_blocks = _copy_result(wheel_present_blocks)
                self._cached_errors = list(errors)
                self._cached_state_timestamp_ns = state_timestamp_ns
                self._last_state_read_mono = now_mono
                state_cached = False

            cameras: dict[str, bytes] = {}
            camera_timestamps_ns: dict[str, int | None] = {}
            camera_ages_ns: dict[str, int | None] = {}
            camera_status: dict[str, dict[str, Any]] = {}
            for name in self._camera_paths:
                worker = self._camera_workers.get(name)
                if worker is None:
                    camera_timestamps_ns[name] = None
                    camera_status[name] = {
                        "fresh": False,
                        "timestamp_ns": None,
                        "error": "camera path is not configured",
                        "error_timestamp_ns": None,
                    }
                    continue
                if isinstance(worker, _CameraWorker):
                    frame, timestamp_ns, error, error_timestamp_ns, fresh, age_ns = worker.snapshot(with_age=True)
                    camera_ages_ns[name] = age_ns
                else:
                    frame, timestamp_ns, error, error_timestamp_ns, fresh = worker.snapshot()
                camera_timestamps_ns[name] = timestamp_ns
                if frame is not None and fresh:
                    cameras[name] = frame
                if error or not fresh:
                    errors.append(error or f"camera {name!r} frame is stale or unavailable")
                camera_status[name] = {
                    "fresh": fresh,
                    "timestamp_ns": timestamp_ns,
                    "error": error,
                    "error_timestamp_ns": error_timestamp_ns,
                }

            # Passive feedback never clears a latched stop fault.  A confirmed
            # stop also requires the owner's earlier active stop procedure.
            wheel_stationary = (
                bool(self.config.get("enable_base"))
                and all(
                    raw.get(name, {}).get("Present_Velocity") == 0 and raw.get(name, {}).get("Moving") == 0
                    for name in WHEEL_NAMES
                )
                and not errors
            )
            base_stopped = wheel_stationary and "base" not in self._active_scopes
            base_stop_confirmed = (
                base_stopped
                and "base" in self._stopped_scopes
                and not self._stop_uncertain
                and not self._torque_ownership_uncertain
            )
            observation = {
                "state_age_ns": max(0, int((time.monotonic() - self._last_state_read_mono) * 1e9)),
                "safety": {
                    "base_control_ready": bool(
                        self._allow_motion
                        and not self._motion_config_error
                        and not self._stop_uncertain
                        and not self._torque_ownership_uncertain
                        and not errors
                        and self._base_directions is not None
                    ),
                    "stopped": base_stopped,
                    "stop_confirmed": base_stop_confirmed,
                    "arms_stop_confirmed": (
                        "arms" in self._stopped_scopes
                        and "arms" not in self._active_scopes
                        and not self._stop_uncertain
                        and not self._torque_ownership_uncertain
                        and not errors
                    ),
                    "stop_unconfirmed": self._stop_uncertain,
                },
                "navigation": {"zero_velocity": wheel_stationary},
                "state": state,
                "source_timestamp_ns": source_timestamp_ns,
                "state_timestamp_ns": state_timestamp_ns,
                "camera_timestamps_ns": camera_timestamps_ns,
                "camera_ages_ns": camera_ages_ns,
                "camera_status": camera_status,
                "raw": raw,
                "raw_fields": raw,
                "wheel_present_blocks": wheel_present_blocks,
                "errors": errors,
                "metadata": _copy_result(self.metadata),
                "state_cached": state_cached,
                "control_state": self.control_state(),
            }
            return observation, cameras

    def _raw_to_joint_value(self, name: str, raw: int) -> float:
        calibration = self._calibration.get(name)
        if calibration is None:
            raise HardwareSafetyError(f"{name}: saved calibration is missing")
        if name.endswith("_gripper"):
            value = (raw - calibration["range_min"]) * 100.0 / (calibration["range_max"] - calibration["range_min"])
            return 100.0 - value if calibration["drive_mode"] else value
        mid = (calibration["range_min"] + calibration["range_max"]) / 2.0
        return (raw - mid) * 360.0 / 4095.0

    def _disconnect_buses_read_only(self) -> list[str]:
        errors: list[str] = []
        for side, bus in tuple(self._buses.items()):
            try:
                if getattr(bus, "is_connected", True):
                    bus.disconnect(disable_torque=False)
            except Exception as exc:  # noqa: BLE001 - disconnect must report SDK failures
                errors.append(f"{side}: disconnect failed: {type(exc).__name__}: {exc}")
                handler = getattr(bus, "port_handler", None)
                close_port = getattr(handler, "closePort", None)
                if callable(close_port):
                    try:
                        close_port()
                    except Exception as close_exc:  # noqa: BLE001 - port cleanup is best effort
                        errors.append(f"{side}: port close failed: {close_exc}")
        self._buses.clear()
        return errors

    def close(self) -> dict[str, Any]:
        with self._io_lock:
            if self._closed:
                return {"closed": True, "errors": []}
            errors: list[str] = []
            self._watchdog_stop.set()
            try:
                if self.armed:
                    try:
                        stop_result = self._stop_locked("close")
                        if stop_result.get("errors"):
                            errors.extend(str(item) for item in stop_result["errors"])
                        if stop_result.get("stop_confirmed") is not True:
                            errors.append("close stop was not confirmed stationary")
                    except Exception as exc:  # noqa: BLE001 - close must continue cleanup
                        self.armed = False
                        self._stop_uncertain = True
                        errors.append(f"close stop failed: {type(exc).__name__}: {exc}")
            finally:
                try:
                    errors.extend(self._disconnect_buses_read_only())
                except Exception as exc:  # noqa: BLE001 - close must continue cleanup
                    errors.append(f"close disconnect failed: {type(exc).__name__}: {exc}")
                self._connected = False
                self._closed = True
                self.armed = False
                self._release_port_locks()
            watchdog = self._watchdog_thread
            workers = tuple(self._camera_workers.values())
            self._camera_workers.clear()

        if watchdog is not None:
            watchdog.join(self._cleanup_timeout_s)
            if watchdog.is_alive():
                errors.append("watchdog thread did not stop within cleanup timeout")
        for worker in workers:
            if not worker.close(self._cleanup_timeout_s):
                errors.append(f"camera {worker.name!r} thread did not stop within cleanup timeout")
        return {"closed": True, "errors": errors, "torque_disabled": False}
