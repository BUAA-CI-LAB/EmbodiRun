"""Canonical XLeRobot external-owner action contract.

One ``action_space`` identifier covers the whole XLeRobot external owner, but
the mobile base and the SO-101 arms use different units:

* base: ``x.vel`` in metres per second and ``theta.vel`` in degrees per second
* arms: every ``*.pos`` joint in degrees, with the gripper in the native
  ``range_0_100`` command range

The contract is a pure data definition so the recipe, the proxy adapter, and
tests share one source of truth.  Unknown action keys, missing unit entries,
and a key that does not belong to the selected scope are rejected; callers
must never invent a unit for an action they did not author.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

XLEROBOT_ACTION_SPACE = "lerobot.xlerobot.external_owner.v1"

_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: Canonical unit for every public arm field.
ARM_UNITS: dict[str, str] = {
    f"{side}_arm_{joint}.pos": ("range_0_100" if joint == "gripper" else "degrees")
    for side in ("left", "right")
    for joint in _ARM_JOINTS
}

#: Canonical unit for every public base field.
BASE_UNITS: dict[str, str] = {
    "x.vel": "metres-per-sec",
    "theta.vel": "angular-degrees-per-sec",
}

#: Units the owner runtime has historically accepted for each canonical name.
UNIT_ALIASES: dict[str, frozenset[str]] = {
    "degrees": frozenset({"degrees"}),
    "range_0_100": frozenset({"range_0_100", "percent"}),
    "metres-per-sec": frozenset({"metres-per-sec", "m/s"}),
    "angular-degrees-per-sec": frozenset({"angular-degrees-per-sec", "deg/s"}),
}

ACTION_SCOPES = ("base", "arms")


def arm_keys() -> frozenset[str]:
    return frozenset(ARM_UNITS)


def base_keys() -> frozenset[str]:
    return frozenset(BASE_UNITS)


def canonical_unit(key: str) -> str | None:
    """Return the canonical unit for one public XLeRobot action field."""

    return ARM_UNITS.get(key, BASE_UNITS.get(key))


def canonical_units(values: Mapping[str, Any]) -> dict[str, str]:
    """Return the canonical unit mapping for a well-formed action payload."""

    units: dict[str, str] = {}
    for key in values:
        unit = canonical_unit(key)
        if unit is None:
            raise ValueError(f"unknown XLeRobot action field: {key}")
        units[key] = unit
    if not units:
        raise ValueError("XLeRobot action must not be empty")
    return units


def scope_for_values(values: Mapping[str, Any]) -> str:
    """Classify one action payload as ``base``, ``arms``, or ``mixed``.

    ``mixed`` means the payload addresses both scopes in a single action; it
    cannot be routed to either scoped owner and must be split by the caller.
    """

    keys = set(values)
    if not keys:
        raise ValueError("XLeRobot action must not be empty")
    unknown = sorted(keys - arm_keys() - base_keys())
    if unknown:
        raise ValueError("unknown XLeRobot action field(s): " + ", ".join(unknown))
    in_arms = any(key in ARM_UNITS for key in keys)
    in_base = any(key in BASE_UNITS for key in keys)
    if in_arms and in_base:
        return "mixed"
    return "arms" if in_arms else "base"


def validate_action(
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    scope: str | None = None,
) -> None:
    """Validate one XLeRobot action against the canonical contract.

    ``scope`` selects the owner scope that will receive the action.  When it is
    omitted the payload's own scope is used, so a caller cannot silently send a
    base action to an arms owner (or the reverse).
    """

    if not isinstance(values, Mapping) or not values:
        raise ValueError("XLeRobot action values must be a non-empty mapping")
    if not isinstance(metadata, Mapping):
        raise ValueError("XLeRobot action metadata must be a mapping")
    if not all(isinstance(key, str) for key in values):
        raise ValueError("XLeRobot action field names must be strings")
    if metadata.get("action_space") != XLEROBOT_ACTION_SPACE:
        raise ValueError("unsupported XLeRobot action_space")
    payload_scope = scope_for_values(values)
    if payload_scope == "mixed":
        raise ValueError("XLeRobot action mixes base and arms fields; split it per scope")
    if scope is not None:
        if scope not in ACTION_SCOPES:
            raise ValueError(f"unknown XLeRobot control scope: {scope}")
        if scope == "base":
            if set(values) != base_keys():
                if any(key in BASE_UNITS for key in values):
                    raise ValueError("base action requires both x.vel and theta.vel")
                raise ValueError("action contains keys outside the selected XLeRobot scope")
        elif payload_scope != "arms":
            raise ValueError("action contains keys outside the selected XLeRobot scope")
    units = metadata.get("units")
    if not isinstance(units, Mapping):
        raise ValueError("XLeRobot action metadata must declare units")
    for key in values:
        expected = canonical_unit(key)
        aliases = UNIT_ALIASES[expected]
        if units.get(key) not in aliases:
            raise ValueError(f"unsupported unit for {key}: expected {expected}")


def stamped_metadata(
    values: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    """Return metadata with the canonical action space and units filled in.

    This is the authoring helper for recipe-owned actions (a calibrated
    handover pose or a checked-in route fixture).  It is deliberately separate
    from :func:`validate_action`: externally supplied actions must declare
    their own units instead of having them guessed here.
    """

    if scope is not None and scope not in ACTION_SCOPES:
        raise ValueError(f"unknown XLeRobot control scope: {scope}")
    result: dict[str, Any] = dict(metadata or {})
    declared = result.get("units")
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("XLeRobot action metadata units must be a mapping")
        result["units"] = dict(declared)
    else:
        result["units"] = canonical_units(values)
    result["action_space"] = XLEROBOT_ACTION_SPACE
    return result


def unsupported_shape_reason(value: Any) -> str | None:
    """Describe why ``value`` is not a canonical XLeRobot action payload."""

    if not isinstance(value, Mapping):
        return "an XLeRobot action must be an object of named public fields"
    if not value:
        return "an XLeRobot action must not be empty"
    unknown = sorted(str(key) for key in value if canonical_unit(str(key)) is None)
    if unknown:
        return "unknown XLeRobot action field(s): " + ", ".join(unknown)
    return None


__all__ = [
    "ACTION_SCOPES",
    "ARM_UNITS",
    "BASE_UNITS",
    "UNIT_ALIASES",
    "XLEROBOT_ACTION_SPACE",
    "arm_keys",
    "base_keys",
    "canonical_unit",
    "canonical_units",
    "scope_for_values",
    "stamped_metadata",
    "unsupported_shape_reason",
    "validate_action",
]
