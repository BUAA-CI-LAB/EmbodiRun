"""Configuration for the unit-preserving simulated policy-vector robot.

The vector deliberately has no physical-unit interpretation here.  A model
checkpoint may use a dataset-native representation that has not been proven
to be degrees or a normalized motor range, so this adapter keeps those values
unchanged and labels every observation accordingly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

POLICY_VECTOR_SIZE = 6
STATE_FIELD = "state_native"
UNVERIFIED_UNITS = "dataset_native_unverified"


def finite_vector(value: object, name: str) -> tuple[float, ...]:
    """Validate one six-value vector without applying a unit conversion."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if len(value) != POLICY_VECTOR_SIZE:
        raise ValueError(f"{name} must contain {POLICY_VECTOR_SIZE} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PolicyVectorConfig:
    """Initial state for an in-memory six-value simulated robot."""

    robot_id: str
    initial_state_native: tuple[float, ...] = (0.0,) * POLICY_VECTOR_SIZE

    @classmethod
    def from_mapping(
        cls,
        robot_id: str,
        value: Mapping[str, Any],
    ) -> PolicyVectorConfig:
        """Build config from Host ``robot_options`` using one explicit field."""

        if not isinstance(value, Mapping):
            raise TypeError("policy-vector configuration options must be a mapping")
        options = dict(value)
        unknown = sorted(set(options) - {"initial_state_native"})
        if unknown:
            raise ValueError("unknown policy-vector configuration fields: " + ", ".join(unknown))
        return cls(
            robot_id=robot_id,
            initial_state_native=finite_vector(
                options.get("initial_state_native", (0.0,) * POLICY_VECTOR_SIZE),
                "initial_state_native",
            ),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        # Re-run validation for direct dataclass construction as well as YAML.
        object.__setattr__(
            self,
            "initial_state_native",
            finite_vector(self.initial_state_native, "initial_state_native"),
        )


__all__ = [
    "POLICY_VECTOR_SIZE",
    "STATE_FIELD",
    "UNVERIFIED_UNITS",
    "PolicyVectorConfig",
    "finite_vector",
]
