"""Relative Quest controller mapping using XLeRobot's planar SO101 kinematics.

The five arm joints cannot follow arbitrary six-dimensional wrist poses. As in
XLeRobot's VR example, reach/height use two-link IK, lateral motion uses pan,
and wrist pitch/roll are controlled separately. No startup zero motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_NAMES = tuple(f"{side}_arm_{joint}.pos" for side in ("left", "right") for joint in JOINTS)
CAMERAS = ("front", "left_wrist", "right_wrist")


def finite(value: Any, name: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} requires {length} values")
    return tuple(finite(v, name) for v in value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_delta(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def wrist_angles(q: tuple[float, ...]) -> tuple[float, float]:
    x, y, z, w = q
    return (
        math.degrees(math.asin(clamp(2 * (w * x - y * z), -1, 1))),
        math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (x * x + z * z))),
    )


def relative_wrist_angles(q: tuple[float, ...], reference: tuple[float, ...]) -> tuple[float, float]:
    """Pitch/roll in the gripped controller's frame, not differences of world Euler angles."""
    x, y, z, w = q
    rx, ry, rz, rw = reference
    return wrist_angles(
        (
            rw * x - rx * w - ry * z + rz * y,
            rw * y + rx * z - ry * w - rz * x,
            rw * z - rx * y + ry * x - rz * w,
            rw * w + rx * x + ry * y + rz * z,
        )
    )


@dataclass(frozen=True)
class Controller:
    position: tuple[float, ...]
    orientation: tuple[float, ...]
    tracked: bool
    grip: bool
    trigger: float
    thumbstick: tuple[float, ...]
    gripper_direction: str = "close"

    @classmethod
    def parse(cls, data: Any) -> Controller:
        if not isinstance(data, dict):
            raise TypeError("both controller objects are required")
        if type(data.get("tracked")) is not bool or type(data.get("grip")) is not bool:
            raise ValueError("tracked and grip must be booleans")
        position = vector(data.get("position"), 3, "position")
        q = vector(data.get("orientation"), 4, "orientation")
        norm = math.sqrt(sum(v * v for v in q))
        if not 0.8 < norm < 1.2:
            raise ValueError("invalid controller quaternion")
        trigger = finite(data.get("trigger"), "trigger")
        stick = vector(data.get("thumbstick"), 2, "thumbstick")
        direction = data.get("gripper_direction", "close")
        if direction not in ("close", "open"):
            raise ValueError("gripper_direction must be close or open")
        if not 0 <= trigger <= 1 or any(abs(v) > 1 for v in stick):
            raise ValueError("controller axes outside range")
        return cls(
            position,
            tuple(v / norm for v in q),
            data["tracked"],
            data["grip"],
            trigger,
            stick,
            direction,
        )

    @property
    def neutral(self) -> bool:
        return self.tracked and not self.grip and self.trigger < 0.1 and max(map(abs, self.thumbstick)) < 0.15


@dataclass(frozen=True)
class InputFrame:
    seq: int
    timestamp_ms: float
    controllers: dict[str, Controller]

    @classmethod
    def parse(cls, data: Any) -> InputFrame:
        if not isinstance(data, dict) or data.get("type") != "input":
            raise ValueError("input message required")
        seq = data.get("seq")
        if type(seq) is not int or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        timestamp = finite(data.get("timestamp_ms"), "timestamp_ms")
        if timestamp < 0:
            raise ValueError("timestamp_ms must be nonnegative")
        controllers = data.get("controllers")
        if not isinstance(controllers, dict):
            raise TypeError("controllers missing")
        return cls(
            seq,
            timestamp,
            {side: Controller.parse(controllers.get(side)) for side in ("left", "right")},
        )

    @property
    def neutral(self) -> bool:
        return all(c.neutral for c in self.controllers.values())

    @property
    def grip_activation_ready(self) -> bool:
        return any(c.grip for c in self.controllers.values()) and all(
            c.tracked and c.trigger < 0.1 and max(map(abs, c.thumbstick)) < 0.15 for c in self.controllers.values()
        )


class InputDelayed(ValueError):
    """An ordered packet that is too old to execute, not a broken session."""


class InputClock:
    """Reject reordered or buffered browser input without trusting wall clocks."""

    def __init__(self, max_age_s: float = 0.25):
        self.max_age_s = max_age_s
        self.seq = -1
        self.client_time = -1.0
        self.offset: float | None = None

    def accept(self, frame: InputFrame, now: float) -> float:
        client_s = frame.timestamp_ms / 1000
        if frame.seq <= self.seq or client_s <= self.client_time:
            raise ValueError("stale or reordered controller input")
        offset = now - client_s
        if self.offset is None:
            self.offset = offset
        # Keep the smallest observed offset; this bounds added network queueing.
        if offset < self.offset - self.max_age_s:
            raise ValueError("controller clock changed; reconnect")
        self.offset = min(offset, self.offset)
        self.seq, self.client_time = frame.seq, client_s
        if offset - self.offset > self.max_age_s:
            raise InputDelayed("controller input delayed in network")
        return client_s + self.offset


class SO101Kinematics:
    """Degree conventions and link geometry from XLeRobot SO101Robot.py.

    FK here is the algebraic inverse of its IK; the upstream FK expression
    uses a different elbow sign. Reach tests guard against that discrepancy.
    """

    l1, l2 = 0.1159, 0.1350
    offset1 = math.atan2(0.028, 0.11257)
    offset2 = math.atan2(0.0052, 0.1349) + offset1

    def forward(self, shoulder: float, elbow: float) -> tuple[float, float]:
        t1 = math.radians(90 - shoulder) - self.offset1
        t2 = math.radians(elbow + 90) - self.offset2
        return (
            self.l1 * math.cos(t1) + self.l2 * math.cos(t1 - t2),
            self.l1 * math.sin(t1) + self.l2 * math.sin(t1 - t2),
        )

    def inverse(self, x: float, y: float, *, reference: tuple[float, float] | None = None) -> tuple[float, float]:
        radius = math.hypot(x, y)
        if not abs(self.l1 - self.l2) + 1e-5 < radius < self.l1 + self.l2 - 1e-5:
            raise ValueError("SO101 target outside reachable workspace")
        t2 = math.acos(clamp((radius**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2), -1, 1))
        candidates = []
        for bend in (t2, -t2):
            t1 = math.atan2(y, x) + math.atan2(self.l2 * math.sin(bend), self.l1 + self.l2 * math.cos(bend))
            angles = (90 - math.degrees(t1 + self.offset1), math.degrees(bend + self.offset2) - 90)
            if reference is None:
                return angles
            candidates.append(tuple(ref + angle_delta(a, ref) for a, ref in zip(angles, reference)))
        return min(candidates, key=lambda angles: math.dist(angles, reference))


@dataclass
class MappingConfig:
    position_scale: float = 0.5
    pan_deg_per_m: float = 120.0
    thumbstick_pan_deg_s: float = 15.0
    wrist_scale: float = 1.0
    max_joint_speed_deg_s: float = 15.0
    max_gripper_speed_pct_s: float = 25.0
    max_linear_m_s: float = 0.05
    max_angular_deg_s: float = 10.0
    max_controller_jump_m: float = 0.12
    max_wrist_jump_deg: float = 45.0
    enable_base: bool = False
    enabled_arms: tuple[str, ...] = ("left", "right")
    gripper_open_pct: float = 100.0
    gripper_closed_pct: float = 0.0
    # Explicit per-side configuration; don't infer mirrored mounting signs.
    pan_signs: dict[str, float] = field(default_factory=lambda: {"left": 1.0, "right": 1.0})

    def __post_init__(self) -> None:
        for name in (
            "position_scale",
            "pan_deg_per_m",
            "thumbstick_pan_deg_s",
            "wrist_scale",
            "max_joint_speed_deg_s",
            "max_gripper_speed_pct_s",
            "max_linear_m_s",
            "max_angular_deg_s",
            "max_controller_jump_m",
            "max_wrist_jump_deg",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.enabled_arms or set(self.enabled_arms) - {"left", "right"}:
            raise ValueError("enabled_arms must contain left and/or right")
        if type(self.enable_base) is not bool:
            raise ValueError("enable_base must be boolean")
        if set(self.pan_signs) != {"left", "right"} or any(v not in (-1, 1) for v in self.pan_signs.values()):
            raise ValueError("pan_signs must explicitly be +/-1 for each arm")
        for name in ("gripper_open_pct", "gripper_closed_pct"):
            if not 0 <= finite(getattr(self, name), name) <= 100:
                raise ValueError(f"{name} must be 0..100")


class QuestMapper:
    def __init__(self, config: MappingConfig | None = None):
        self.config = config or MappingConfig()
        self.kinematics = SO101Kinematics()
        self.anchors: dict[str, tuple[Controller, dict[str, float]]] = {}
        self.previous: dict[str, Controller] = {}
        self.last_targets: dict[str, float] = {}
        self.control_mode = "arms"
        self.drive_hold: dict[str, float] = {}
        self.motion_reference: dict[str, Controller] = {}
        self.trigger_ready: dict[str, bool] = {}
        self.diagnostics: dict[str, dict] = {}

    def set_control_mode(self, mode: str) -> None:
        if mode not in ("arms", "drive"):
            raise ValueError("control mode must be arms or drive")
        if mode == "drive" and not self.config.enable_base:
            raise ValueError("base mapping is disabled")
        if mode != self.control_mode:
            self.reset()
            self.control_mode = mode

    def reset(self) -> None:
        self.anchors.clear()
        self.previous.clear()
        self.last_targets.clear()
        self.drive_hold.clear()
        self.motion_reference.clear()
        self.trigger_ready.clear()
        self.diagnostics.clear()

    def accept_applied_action(self, applied: dict[str, float]) -> None:
        # Follow the robot's rate-limited, quantized command receipt, not an
        # unsent target that would otherwise accumulate while the hand stops.
        for name, value in applied.items():
            if name in self.last_targets:
                self.last_targets[name] = finite(value, name)

    def _arm_step(self, names, joints, delta, rotation, dt, limits, side):
        """Keep planar IK coherent without freezing independent pan/wrist axes.

        Position has priority over wrist attitude on this five-DOF arm. Shoulder
        and elbow still share one Cartesian fraction (no joint-wise IK clipping).
        Unreachable/excess motion is consumed, never queued for a later jump.
        """
        cfg = self.config
        x, y = self.kinematics.forward(joints[names[1]], joints[names[2]])
        speed_step = cfg.max_joint_speed_deg_s * dt
        joint_limited, speed_limited = set(), set()
        workspace_limited = False

        def candidate(fraction, report=False):
            nonlocal workspace_limited
            try:
                if delta[1] == 0 and delta[2] == 0:
                    shoulder, elbow = joints[names[1]], joints[names[2]]
                else:
                    shoulder, elbow = self.kinematics.inverse(
                        x - delta[2] * cfg.position_scale * fraction,
                        y + delta[1] * cfg.position_scale * fraction,
                        reference=(joints[names[1]], joints[names[2]]),
                    )
                valid = True
                for name, value in zip(names[1:3], (shoulder, elbow)):
                    low, high = limits.get(name, (-180, 180))
                    if not low <= value <= high:
                        valid = False
                        if report:
                            joint_limited.add(name)
                    if abs(value - joints[name]) > speed_step + 1e-9:
                        valid = False
                        if report:
                            speed_limited.add(name)
                return {names[1]: shoulder, names[2]: elbow} if valid else None
            except ValueError:
                if report:
                    workspace_limited = True
                return None

        fraction = 1.0
        result = candidate(fraction, report=True)
        while result is None and fraction > 1 / 16384:
            fraction *= 0.5
            result = candidate(fraction)
        if result is None:
            result = {name: joints[name] for name in names[1:3]}
            fraction = 0.0
        elif fraction < 1:
            low, high = fraction, min(1.0, fraction * 2)
            for _ in range(10):
                middle = (low + high) / 2
                refined = candidate(middle)
                if refined is None:
                    high = middle
                else:
                    low, result = middle, refined
            fraction = low

        fractions = [fraction]

        def independent_step(name, target):
            low, high = limits.get(name, (-180, 180))
            if not low <= target <= high:
                joint_limited.add(name)
            if abs(target - joints[name]) > speed_step + 1e-9:
                speed_limited.add(name)
            value = clamp(target, max(low, joints[name] - speed_step), min(high, joints[name] + speed_step))
            if abs(target - joints[name]) > 1e-9:
                fractions.append(abs((value - joints[name]) / (target - joints[name])))
            result[name] = value

        independent_step(names[0], joints[names[0]] + delta[0] * cfg.pan_deg_per_m * cfg.pan_signs[side])
        independent_step(
            names[3],
            joints[names[3]]
            + rotation[0] * cfg.wrist_scale
            - (result[names[1]] - joints[names[1]])
            - (result[names[2]] - joints[names[2]]),
        )
        independent_step(names[4], joints[names[4]] + rotation[1] * cfg.wrist_scale)
        return (
            result,
            min(fractions),
            {
                "position_fraction": round(fraction, 4),
                "joint_limit_joints": sorted(joint_limited),
                "speed_limited_joints": sorted(speed_limited),
                "workspace_limited": workspace_limited,
            },
        )

    def map(
        self,
        frame: InputFrame,
        state: dict[str, float],
        dt: float,
        limits: dict[str, list[float]] | None = None,
    ) -> dict[str, float]:
        dt = clamp(finite(dt, "dt"), 0.0, 0.1)
        cfg = self.config
        if not all(c.tracked for c in frame.controllers.values()):
            self.reset()
            raise ValueError("controller tracking lost")
        current = {name: finite(state[name], name) for name in JOINT_NAMES}
        if self.control_mode == "drive":
            if not cfg.enable_base:
                raise ValueError("base mapping is disabled")
            # Hold the pose captured on enable, not a moving reference that
            # follows servo sag. Controller poses/triggers do not move arms.
            if not self.drive_hold:
                self.drive_hold = dict(current)
            result = dict(self.drive_hold)
            controller = frame.controllers["right"]
            stick = controller.thumbstick
            result["x.vel"] = -stick[1] * cfg.max_linear_m_s if controller.grip and abs(stick[1]) > 0.15 else 0.0
            result["theta.vel"] = -stick[0] * cfg.max_angular_deg_s if controller.grip and abs(stick[0]) > 0.15 else 0.0
            self.last_targets = dict(result)
            return result
        # A hold is a fixed commanded pose, not the measured (possibly sagging)
        # pose copied back into Goal_Position at every sample.
        result = {name: self.last_targets.get(name, value) for name, value in current.items()}
        self.diagnostics = {}
        for side in ("left", "right"):
            controller = frame.controllers[side]
            names = [f"{side}_arm_{j}.pos" for j in JOINTS]
            if side not in cfg.enabled_arms or not controller.grip:
                self.anchors.pop(side, None)
                self.previous.pop(side, None)
                self.motion_reference.pop(side, None)
                self.trigger_ready.pop(side, None)
                self.diagnostics[side] = {"active": False}
                continue
            if side not in self.anchors:
                self.anchors[side] = controller, {name: current[name] for name in names}
                self.previous[side] = controller
                self.motion_reference[side] = controller
                self.trigger_ready[side] = controller.trigger < 0.1
                result.update({name: current[name] for name in names[:5]})
                self.diagnostics[side] = {
                    "active": True,
                    "trigger_ready": self.trigger_ready[side],
                    "gripper_direction": controller.gripper_direction,
                }
                continue
            previous = self.previous[side]
            displacement = math.dist(controller.position, previous.position)
            rotation_step = 2 * math.degrees(
                math.acos(clamp(abs(sum(a * b for a, b in zip(controller.orientation, previous.orientation))), 0, 1))
            )
            if displacement > cfg.max_controller_jump_m or rotation_step > cfg.max_wrist_jump_deg:
                self.reset()
                raise ValueError("controller pose jumped; stop and re-arm")
            self.previous[side] = controller
            reference = self.motion_reference[side]
            raw_delta = tuple(a - b for a, b in zip(controller.position, reference.position))
            delta = tuple(value if abs(value) >= 0.001 else 0.0 for value in raw_delta)
            # Own Grip + horizontal stick drives the bottom pan joint. While
            # deflected it overrides lateral hand motion, not height/reach or
            # wrist pose. Consume lateral hand motion during the override.
            stick = controller.thumbstick[0]
            stick = math.copysign((abs(stick) - 0.15) / 0.85, stick) if abs(stick) > 0.15 else 0.0
            if stick:
                delta = (stick * cfg.thumbstick_pan_deg_s * dt / cfg.pan_deg_per_m, delta[1], delta[2])
            raw_rotation = relative_wrist_angles(controller.orientation, reference.orientation)
            rotation = tuple(value if abs(value) >= 0.5 else 0.0 for value in raw_rotation)
            # Per-axis deadband: a large lateral move must not activate tiny
            # depth noise at a shoulder limit. Retain unconsumed components so
            # slow deliberate height/depth and wrist motion still accumulate.
            moving, rotating = any(delta), any(rotation)
            fraction = 1.0
            restrictions = {}
            if moving or rotating:
                arm, fraction, restrictions = self._arm_step(
                    names,
                    result,
                    delta,
                    rotation,
                    dt,
                    limits or {},
                    side,
                )
                result.update(arm)
                self.motion_reference[side] = replace(
                    reference,
                    position=tuple(controller.position[i] if delta[i] else reference.position[i] for i in range(3)),
                    orientation=controller.orientation if rotating else reference.orientation,
                )

            # Trigger is a hold-to-run gripper command. Y/B chooses direction;
            # releasing holds, and a direction change requires a fresh trigger
            # squeeze so switching modes cannot reverse an already-held claw.
            grip_name = names[-1]
            held = result[grip_name]
            if controller.gripper_direction != previous.gripper_direction:
                self.trigger_ready[side] = False
            if controller.trigger < 0.1:
                self.trigger_ready[side] = True
            if self.trigger_ready[side] and controller.trigger >= 0.1:
                low, high = (limits or {}).get(grip_name, (0, 100))
                target = clamp(
                    cfg.gripper_open_pct if controller.gripper_direction == "open" else cfg.gripper_closed_pct,
                    low,
                    high,
                )
                step = cfg.max_gripper_speed_pct_s * dt * controller.trigger
                result[grip_name] = clamp(target, held - step, held + step)
            self.diagnostics[side] = {
                "active": True,
                "limited": fraction < 0.999,
                "motion_fraction": round(fraction, 4),
                "trigger_ready": self.trigger_ready[side],
                "gripper_direction": controller.gripper_direction,
                "pan_source": "thumbstick" if stick else "pose",
                **restrictions,
            }
        # Arm manipulation never drives the base, even if the installation
        # has explicitly enabled wheel control.
        result["x.vel"] = 0.0
        result["theta.vel"] = 0.0
        self.last_targets = dict(result)
        return result
