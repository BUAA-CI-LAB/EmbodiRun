"""Named, dependency-light SO101 planar correction adapter.

This is the robot-specific bridge for the public Agent session.  It maps the
reviewer's declared ``so101_shoulder_plane`` / ``m_deg`` waypoints into named
12-joint public action payloads.  It does not connect to a robot, read a
camera, clamp an unknown binding, or import the old managed runtime.  A caller
must provide the exact Deploy feature names and a fresh observation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from embodirun.client import Observation


class SO101CorrectionError(ValueError):
    """A correction cannot be mapped using the explicitly configured bridge."""


class SO101PlanarCorrectionMapper:
    """Map bounded shoulder-plane corrections to 12 named degree targets.

    The bridge uses the same nominal two-link shoulder/elbow model as the
    existing SO101 integration.  The Deploy owner remains responsible for its
    own binding limits, current-state admission, feedback, and execution.

    ``control_hz`` is required because Deploy's public executor advances an
    action sequence at one fixed rate and does not interpret arbitrary
    ``RobotAction.timestamp_s`` values as per-waypoint motion durations.  A
    correction is rejected unless its declared ``duration_s`` equals one
    public control step.  Supply ``action_encoder=BiSO101ActionEncoder()`` for
    the structured dual-arm adapter; without it this bridge intentionally
    returns the explicitly configured flat feature mapping.
    """

    _l1 = 0.1159
    _l2 = 0.1350
    _offset1 = math.atan2(0.028, 0.11257)
    _offset2 = math.atan2(0.0052, 0.1349) + _offset1

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        calibration: Mapping[str, Mapping[str, Any]],
        control_hz: float,
        max_waypoint_duration_s: float = 30.0,
        max_joint_delta_deg: float | None = None,
        action_encoder: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        names = tuple(feature_names)
        if len(names) != 12 or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must contain twelve non-empty names")
        if len(set(names)) != 12:
            raise ValueError("feature_names must be unique")
        if not isinstance(calibration, Mapping) or set(calibration) != set(names):
            raise ValueError("calibration must declare exactly the configured SO101 feature names")
        validated_calibration: dict[str, dict[str, Any]] = {}
        for name in names:
            item = calibration[name]
            if not isinstance(item, Mapping):
                raise ValueError(f"calibration.{name} must be a mapping")
            for field in ("range_min", "range_max", "drive_mode"):
                if field not in item or isinstance(item[field], bool):
                    raise ValueError(f"calibration.{name}.{field} is required")
            range_min = self._number(item["range_min"], f"calibration.{name}.range_min")
            range_max = self._number(item["range_max"], f"calibration.{name}.range_max")
            drive_mode = item["drive_mode"]
            if drive_mode not in (0, 1) or not 0 <= range_min < range_max <= 4095:
                raise ValueError(f"calibration.{name} has invalid range or drive mode")
            validated_calibration[name] = {
                **dict(item),
                "range_min": range_min,
                "range_max": range_max,
                "drive_mode": drive_mode,
            }
        if (
            isinstance(max_waypoint_duration_s, bool)
            or not isinstance(max_waypoint_duration_s, (int, float))
            or not math.isfinite(float(max_waypoint_duration_s))
            or max_waypoint_duration_s <= 0
        ):
            raise ValueError("max_waypoint_duration_s must be finite and positive")
        if (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not math.isfinite(float(control_hz))
            or control_hz <= 0
        ):
            raise ValueError("control_hz must be finite and positive")
        if max_joint_delta_deg is not None and (
            isinstance(max_joint_delta_deg, bool)
            or not isinstance(max_joint_delta_deg, (int, float))
            or not math.isfinite(float(max_joint_delta_deg))
            or max_joint_delta_deg <= 0
        ):
            raise ValueError("max_joint_delta_deg must be finite and positive")
        if action_encoder is not None and not callable(action_encoder):
            raise TypeError("action_encoder must be callable")
        encoder_names = getattr(action_encoder, "feature_names", None)
        if encoder_names is not None and tuple(encoder_names) != names:
            raise ValueError("action_encoder feature_names must match SO101 feature_names")
        self.feature_names = names
        self.calibration = validated_calibration
        self.control_hz = float(control_hz)
        self.max_waypoint_duration_s = float(max_waypoint_duration_s)
        self.max_joint_delta_deg = float(max_joint_delta_deg) if max_joint_delta_deg is not None else None
        self.action_encoder = action_encoder

    @staticmethod
    def _number(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise SO101CorrectionError(f"{label} must be finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise SO101CorrectionError(f"{label} must be finite") from error
        if not math.isfinite(number):
            raise SO101CorrectionError(f"{label} must be finite")
        return number

    @staticmethod
    def _state(observation: Observation, names: Sequence[str]) -> list[float]:
        value = observation.robot
        if isinstance(value, Mapping):
            # Public SO101 adapters expose named position fields.  Do not
            # guess a field order when a binding has not declared one.
            try:
                values = [value[name] for name in names]
            except KeyError as error:
                raise SO101CorrectionError("observation does not contain all configured SO101 features") from error
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 12:
                raise SO101CorrectionError("observation robot state must contain 12 values")
            values = list(value)
        else:
            raise SO101CorrectionError("observation has no usable SO101 robot state")
        result = [SO101PlanarCorrectionMapper._number(item, "observation state") for item in values]
        if not all(-180.0 <= item <= 180.0 for item in result[:5] + result[6:11]):
            raise SO101CorrectionError("observation arm joints must be in degree bounds")
        if not all(0.0 <= result[index] <= 100.0 for index in (5, 11)):
            raise SO101CorrectionError("observation grippers must be in [0,100]")
        return result

    def _inverse(self, reach_m: float, height_m: float, reference: tuple[float, float]) -> tuple[float, float]:
        radius = math.hypot(reach_m, height_m)
        if not abs(self._l1 - self._l2) + 1e-5 < radius < self._l1 + self._l2 - 1e-5:
            raise SO101CorrectionError("SO101 target is outside the reachable workspace")
        cosine = (radius * radius - self._l1 * self._l1 - self._l2 * self._l2) / (2 * self._l1 * self._l2)
        theta2 = math.acos(max(-1.0, min(1.0, cosine)))
        candidates = []
        for bend in (theta2, -theta2):
            theta1 = math.atan2(height_m, reach_m) + math.atan2(
                self._l2 * math.sin(bend), self._l1 + self._l2 * math.cos(bend)
            )
            candidates.append(
                (
                    90.0 - math.degrees(theta1 + self._offset1),
                    math.degrees(bend + self._offset2) - 90.0,
                )
            )
        return min(candidates, key=lambda candidate: math.dist(candidate, reference))

    def _forward(self, shoulder: float, elbow: float) -> tuple[float, float]:
        theta1 = math.radians(90.0 - shoulder) - self._offset1
        theta2 = math.radians(elbow + 90.0) - self._offset2
        return (
            self._l1 * math.cos(theta1) + self._l2 * math.cos(theta1 - theta2),
            self._l1 * math.sin(theta1) + self._l2 * math.sin(theta1 - theta2),
        )

    def preview(self, rows: Sequence[Sequence[float]]) -> dict[str, Any]:
        """Return the nominal planar preview expected by ``AstraCodexReviewer``."""

        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise SO101CorrectionError("preview rows must be a sequence")
        left: list[list[float]] = []
        right: list[list[float]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Sequence) or len(row) != 12:
                raise SO101CorrectionError(f"preview row {index} must contain 12 values")
            values = [self._number(item, f"preview[{index}]") for item in row]
            left.append(list(self._forward(values[1], values[2])))
            right.append(list(self._forward(values[7], values[8])))
        return {
            "frame": "so101_shoulder_plane",
            "left": left,
            "right": right,
            "source": "configured SO101 planar geometry; no collision prediction",
        }

    def _within_calibration(self, row: Sequence[float]) -> None:
        for index, name in enumerate(self.feature_names):
            item = self.calibration[name]
            if index in (5, 11):
                if not 0.0 <= row[index] <= 100.0:
                    raise SO101CorrectionError(f"{name} gripper is outside [0,100]")
                continue
            # Deploy exposes the calibrated arm position in degrees.  The
            # raw observed range gives a finite symmetric degree envelope;
            # the owner still performs the final binding validation.
            limit = (float(item["range_max"]) - float(item["range_min"])) * 180.0 / 4095.0
            if not -limit <= row[index] <= limit:
                raise SO101CorrectionError(f"{name} target is outside calibration")

    def __call__(
        self,
        corrections: Sequence[Mapping[str, Any]],
        *,
        observation: Observation,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(corrections, (str, bytes))
            or not isinstance(corrections, Sequence)
            or not 1 <= len(corrections) <= 5
        ):
            raise SO101CorrectionError("corrections must contain 1..5 waypoints")
        current = self._state(observation, self.feature_names)
        result: list[dict[str, Any]] = []
        for index, waypoint in enumerate(corrections):
            if not isinstance(waypoint, Mapping) or set(waypoint) != {
                "left",
                "right",
                "frame",
                "units",
                "duration_s",
            }:
                raise SO101CorrectionError(f"correction {index} has invalid fields")
            if waypoint["frame"] != "so101_shoulder_plane" or waypoint["units"] != "m_deg":
                raise SO101CorrectionError(f"correction {index} has unsupported frame or units")
            duration = self._number(waypoint["duration_s"], "duration_s")
            if duration <= 0 or duration > self.max_waypoint_duration_s:
                raise SO101CorrectionError("duration_s is outside the configured bound")
            step_duration = 1.0 / self.control_hz
            if not math.isclose(duration, step_duration, rel_tol=1e-6, abs_tol=1e-6):
                raise SO101CorrectionError(
                    "duration_s must equal the configured public control_hz step; "
                    "per-waypoint timing is not supported by ControlClient.execute"
                )
            row = list(current)
            active = False
            for side, base in (("left", 0), ("right", 6)):
                target = waypoint[side]
                if target is None:
                    continue
                active = True
                if not isinstance(target, Mapping) or set(target) != {
                    "reach_m",
                    "height_m",
                    "pan_deg",
                    "wrist_flex_deg",
                    "wrist_roll_deg",
                    "gripper",
                }:
                    raise SO101CorrectionError(f"{side} correction fields are incomplete")
                reach = self._number(target["reach_m"], f"{side}.reach_m")
                height = self._number(target["height_m"], f"{side}.height_m")
                shoulder, elbow = self._inverse(reach, height, (current[base + 1], current[base + 2]))
                row[base] = self._number(target["pan_deg"], f"{side}.pan_deg")
                row[base + 1] = shoulder
                row[base + 2] = elbow
                row[base + 3] = self._number(target["wrist_flex_deg"], f"{side}.wrist_flex_deg")
                row[base + 4] = self._number(target["wrist_roll_deg"], f"{side}.wrist_roll_deg")
                row[base + 5] = self._number(target["gripper"], f"{side}.gripper")
                if not 0.0 <= row[base + 5] <= 100.0:
                    raise SO101CorrectionError(f"{side}.gripper must be in [0,100]")
            if not active:
                raise SO101CorrectionError("correction must target at least one arm")
            if self.max_joint_delta_deg is not None and any(
                abs(row[index] - current[index]) > self.max_joint_delta_deg
                for index in tuple(range(5)) + tuple(range(6, 11))
            ):
                raise SO101CorrectionError("correction exceeds the configured joint delta")
            self._within_calibration(row)
            timestamp_s = index * step_duration
            action_metadata = {
                "action_semantics": "biso101_so101_v1",
                "action_layout": "left6_right6",
                "action_encoding": "absolute",
                "joint_position_unit": "degrees",
                "correction": True,
                "control_hz": self.control_hz,
                "duration_s": duration,
                "duration_semantics": "public_control_hz_step",
            }
            if self.action_encoder is None:
                result.append(
                    {
                        "timestamp_s": timestamp_s,
                        "values": dict(zip(self.feature_names, row)),
                        "metadata": action_metadata,
                    }
                )
            else:
                encoded = self.action_encoder(tuple(row), timestamp_s=timestamp_s, metadata=action_metadata)
                if not isinstance(encoded, Mapping):
                    raise SO101CorrectionError("action_encoder must return a public action mapping")
                action = dict(encoded)
                declared_timestamp = action.get("timestamp_s", timestamp_s)
                if (
                    isinstance(declared_timestamp, bool)
                    or not isinstance(declared_timestamp, (int, float))
                    or not math.isfinite(float(declared_timestamp))
                    or not math.isclose(float(declared_timestamp), timestamp_s, abs_tol=1e-9)
                ):
                    raise SO101CorrectionError("action_encoder timestamp_s must match the public schedule")
                if not isinstance(action.get("values"), Mapping):
                    raise SO101CorrectionError("action_encoder must return public values")
                encoded_metadata = action.get("metadata", {})
                if not isinstance(encoded_metadata, Mapping):
                    raise SO101CorrectionError("action_encoder metadata must be a mapping")
                merged_metadata = dict(action_metadata)
                merged_metadata.update(dict(encoded_metadata))
                action["timestamp_s"] = timestamp_s
                action["metadata"] = merged_metadata
                result.append(action)
            current = row
        return result


__all__ = ["SO101CorrectionError", "SO101PlanarCorrectionMapper"]
