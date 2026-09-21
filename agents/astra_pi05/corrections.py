"""Typed injection seam for robot-specific Astra correction mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

from embodirun.client import Observation

PublicAction: TypeAlias = Mapping[str, Any]
JointRows: TypeAlias = Sequence[Sequence[float]]
MappedCorrections: TypeAlias = PublicAction | JointRows


class CorrectionMapper(Protocol):
    """Map declared SO101 shoulder-plane corrections to public actions.

    Implementations belong to a robot-specific integration.  The algorithm
    layer does not infer kinematics.  A mapper may return public action
    dictionaries directly, or finite 12-value rows when the caller also
    supplies explicit ``feature_names`` to the cooperative loop.
    """

    def __call__(
        self,
        corrections: Sequence[Mapping[str, Any]],
        *,
        observation: Observation,
    ) -> Sequence[PublicAction] | JointRows | Mapping[str, Any]:
        """Return 1..5 already mapped action payloads or explicitly named rows."""


__all__ = [
    "CorrectionMapper",
    "JointRows",
    "MappedCorrections",
    "PublicAction",
]
