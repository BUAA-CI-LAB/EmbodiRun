"""Robot-independent planar state, limits, and velocity commands."""

from __future__ import annotations

from dataclasses import dataclass, field

from embodirun.types import Metadata
from embodirun.utils import Pose2D

from ._validation import finite, metadata, sequence


def _positive(value: object, name: str) -> float:
    result = finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded(value: object, name: str, maximum: float) -> float:
    result = finite(value, name)
    if abs(result) > maximum:
        raise ValueError(f"{name} must be within +/-{maximum}")
    return result


@dataclass(frozen=True, slots=True)
class PlanarVelocityLimits:
    max_abs_vx_mps: float = 0.35
    max_abs_vy_mps: float = 0.35
    max_abs_yaw_rate_rps: float = 0.7

    def __post_init__(self) -> None:
        for name in ("max_abs_vx_mps", "max_abs_vy_mps", "max_abs_yaw_rate_rps"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


DEFAULT_PLANAR_VELOCITY_LIMITS = PlanarVelocityLimits()


@dataclass(frozen=True, slots=True)
class PlanarVelocityCommand:
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float
    limits: PlanarVelocityLimits = DEFAULT_PLANAR_VELOCITY_LIMITS

    def __post_init__(self) -> None:
        if not isinstance(self.limits, PlanarVelocityLimits):
            raise TypeError("limits must be PlanarVelocityLimits")
        object.__setattr__(
            self,
            "vx_mps",
            _bounded(self.vx_mps, "vx_mps", self.limits.max_abs_vx_mps),
        )
        object.__setattr__(
            self,
            "vy_mps",
            _bounded(self.vy_mps, "vy_mps", self.limits.max_abs_vy_mps),
        )
        object.__setattr__(
            self,
            "yaw_rate_rps",
            _bounded(
                self.yaw_rate_rps,
                "yaw_rate_rps",
                self.limits.max_abs_yaw_rate_rps,
            ),
        )

    @classmethod
    def stopped(
        cls,
        limits: PlanarVelocityLimits = DEFAULT_PLANAR_VELOCITY_LIMITS,
    ) -> PlanarVelocityCommand:
        return cls(0.0, 0.0, 0.0, limits)

    @property
    def moving(self) -> bool:
        return any(abs(value) > 1e-6 for value in (self.vx_mps, self.vy_mps, self.yaw_rate_rps))


@dataclass(frozen=True, slots=True)
class MobileBaseState:
    pose: Pose2D
    forward_velocity_mps: float
    lateral_velocity_mps: float
    yaw_rate_rps: float
    sequence: int
    received_at_s: float | None
    active_velocity_lease_id: str | None = None
    raw: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose2D):
            raise TypeError("pose must be a Pose2D")
        for name in (
            "forward_velocity_mps",
            "lateral_velocity_mps",
            "yaw_rate_rps",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "sequence", sequence(self.sequence))
        if self.received_at_s is not None:
            object.__setattr__(
                self,
                "received_at_s",
                finite(self.received_at_s, "received_at_s", minimum=0.0),
            )
        if self.active_velocity_lease_id is not None and (
            not isinstance(self.active_velocity_lease_id, str) or not self.active_velocity_lease_id.strip()
        ):
            raise ValueError("active_velocity_lease_id must be a non-empty string or None")
        object.__setattr__(self, "raw", metadata(self.raw, "raw"))

    def as_observation_metadata(self) -> dict[str, object]:
        result: dict[str, object] = dict(self.raw)
        result.update(
            {
                "position": [self.pose.x_m, self.pose.y_m],
                "yaw": self.pose.yaw_rad,
                "velocity": [self.forward_velocity_mps, self.lateral_velocity_mps],
                "yaw_rate": self.yaw_rate_rps,
                "sequence": self.sequence,
            }
        )
        if self.received_at_s is not None:
            result["received_at_unix"] = self.received_at_s
        if self.active_velocity_lease_id is not None:
            result["active_velocity_lease_id"] = self.active_velocity_lease_id
        return result


__all__ = [
    "DEFAULT_PLANAR_VELOCITY_LIMITS",
    "MobileBaseState",
    "PlanarVelocityCommand",
    "PlanarVelocityLimits",
]
