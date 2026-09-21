"""Explicit evidence normalization for the XLeRobot snack delivery recipe.

The public Control observation does not itself publish XLeRobot-specific
safety or task evidence.  This module is the one boundary that turns an
owner's observation payload into the canonical evidence names the recipe
state machine reasons about:

``base_control_ready``, ``stopped``, ``stop_confirmed``, ``grasp_confirmed``,
``route_arrived``, ``zero_velocity``.

Rules encoded here:

* only an explicit JSON ``true`` confirms an evidence name; a missing field,
  ``false``, ``"true"``, or any other value fails closed;
* the mapping is fully configurable per deployment, so an owner adapter keeps
  its own feedback names;
* a name may be resolved by an explicit operator confirmation instead of
  sensor evidence, but that is always labelled ``manual_confirmation`` and is
  never recorded as owner/sensor confirmation;
* ``grasp_confirmed`` is only enabled for manual confirmation when the
  deployment config says so explicitly.

The boundary is injectable: any object implementing
:class:`EvidenceNormalizer` can replace :class:`ConfigEvidenceNormalizer`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

OWNER_SOURCE = "owner_observation"
MANUAL_SOURCE = "manual_confirmation"
MISSING_SOURCE = "missing"

#: Canonical evidence names understood by the recipe state machine.
EVIDENCE_NAMES = (
    "base_control_ready",
    "arms_stop_confirmed",
    "stopped",
    "stop_confirmed",
    "grasp_confirmed",
    "route_arrived",
    "zero_velocity",
)

#: Default observation path for each canonical name.
DEFAULT_PATHS: dict[str, tuple[str, ...]] = {
    "base_control_ready": ("safety", "base_control_ready"),
    "arms_stop_confirmed": ("safety", "arms_stop_confirmed"),
    "stopped": ("safety", "stopped"),
    "stop_confirmed": ("safety", "stop_confirmed"),
    "grasp_confirmed": ("task_evidence", "grasp_confirmed"),
    "route_arrived": ("navigation", "arrived"),
    "zero_velocity": ("navigation", "zero_velocity"),
}

#: Historical flat config keys accepted for compatibility.
LEGACY_PATH_KEYS: dict[str, str] = {
    "ready_for_base_path": "base_control_ready",
    "stopped_path": "stopped",
    "stop_confirmed_path": "stop_confirmed",
    "grasp_confirmed_path": "grasp_confirmed",
}


class EvidenceError(RuntimeError):
    """The evidence configuration or observation payload is unusable."""


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    """One normalized evidence decision with its provenance."""

    name: str
    confirmed: bool
    source: str
    path: tuple[str, ...]
    value: Any = None
    present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confirmed": self.confirmed,
            "source": self.source,
            "path": list(self.path),
            "value": self.value,
            "present": self.present,
        }


class EvidenceNormalizer(Protocol):
    """Injectable evidence boundary used by the recipe state machine."""

    def evaluate(self, name: str, payload: Mapping[str, Any]) -> EvidenceValue: ...

    def manual_confirmation_allowed(self, name: str) -> bool: ...


def _lookup(payload: Mapping[str, Any], path: Sequence[str]) -> tuple[bool, Any]:
    current: Any = payload
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return False, None
        current = current[component]
    return True, current


def lookup_path(payload: Mapping[str, Any], path: Sequence[str]) -> tuple[bool, Any]:
    """Return ``(present, value)`` for one field path in an observation."""

    return _lookup(payload, path)


@dataclass(frozen=True, slots=True)
class ConfigEvidenceNormalizer:
    """Map a deployment's observation fields onto canonical evidence names."""

    paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    manual_confirmation: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ConfigEvidenceNormalizer:
        raw = config.get("evidence", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise EvidenceError("evidence must be an object")
        paths: dict[str, tuple[str, ...]] = {}
        declared = raw.get("paths", {})
        if not isinstance(declared, Mapping):
            raise EvidenceError("evidence.paths must be an object")
        for name, value in declared.items():
            if name not in EVIDENCE_NAMES:
                raise EvidenceError(f"evidence.paths.{name} is not a known evidence name")
            paths[name] = _path(value, f"evidence.paths.{name}")
        for legacy_name, name in LEGACY_PATH_KEYS.items():
            if legacy_name in raw:
                if name in paths:
                    raise EvidenceError(f"evidence declares {name} twice ({legacy_name} and paths)")
                paths[name] = _path(raw[legacy_name], f"evidence.{legacy_name}")
        manual = raw.get("manual_confirmation", ())
        if manual is None:
            manual = ()
        if isinstance(manual, (str, bytes)) or not isinstance(manual, Sequence):
            raise EvidenceError("evidence.manual_confirmation must be a list of names")
        allowed: set[str] = set()
        for name in manual:
            if name not in {"grasp_confirmed", "route_arrived"}:
                raise EvidenceError("manual confirmation is allowed only for grasp_confirmed or route_arrived")
            if name not in EVIDENCE_NAMES:
                raise EvidenceError(f"evidence.manual_confirmation contains unknown name {name!r}")
            allowed.add(name)
        return cls(paths=paths, manual_confirmation=frozenset(allowed))

    def path_for(self, name: str) -> tuple[str, ...]:
        if name not in EVIDENCE_NAMES:
            raise EvidenceError(f"unknown evidence name: {name}")
        return tuple(self.paths.get(name, DEFAULT_PATHS[name]))

    def evaluate(self, name: str, payload: Mapping[str, Any]) -> EvidenceValue:
        if not isinstance(payload, Mapping):
            raise EvidenceError("evidence payload must be an observation object")
        path = self.path_for(name)
        present, value = _lookup(payload, path)
        confirmed = present and value is True
        source = OWNER_SOURCE if confirmed or present else MISSING_SOURCE
        return EvidenceValue(
            name=name,
            confirmed=confirmed,
            source=source,
            path=path,
            value=value,
            present=present,
        )

    def manual_confirmation_allowed(self, name: str) -> bool:
        return name in self.manual_confirmation


def _path(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise EvidenceError(f"{location} must be a non-empty list of field names")
    if any(not isinstance(item, str) or not item for item in value):
        raise EvidenceError(f"{location} must contain non-empty field names")
    return tuple(value)


def requirement_list(config: Mapping[str, Any], key: str, default: Sequence[str]) -> tuple[str, ...]:
    """Read one list of canonical evidence names from the evidence section."""

    raw = config.get("evidence", {})
    if not isinstance(raw, Mapping):
        raise EvidenceError("evidence must be an object")
    value = raw.get(key, list(default))
    if value is None:
        value = list(default)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceError(f"evidence.{key} must be a list of evidence names")
    result: list[str] = []
    for name in value:
        if name not in EVIDENCE_NAMES:
            raise EvidenceError(f"evidence.{key} contains unknown name {name!r}")
        result.append(name)
    return tuple(result)


__all__ = [
    "DEFAULT_PATHS",
    "EVIDENCE_NAMES",
    "LEGACY_PATH_KEYS",
    "MANUAL_SOURCE",
    "MISSING_SOURCE",
    "OWNER_SOURCE",
    "ConfigEvidenceNormalizer",
    "EvidenceError",
    "EvidenceNormalizer",
    "EvidenceValue",
    "lookup_path",
    "requirement_list",
]
