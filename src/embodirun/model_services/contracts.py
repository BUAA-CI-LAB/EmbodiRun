"""Model-independent values exchanged with inference services."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must match {IDENTIFIER.pattern}")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ImagePayload:
    """One encoded camera image as it crosses the inference boundary.

    The bytes are already encoded, so the transport stays model-neutral and the payload can
    be written straight to HTTP or WirelessComm. ``name`` identifies the camera, and adapter
    configuration is what maps that name onto a checkpoint image feature.
    """

    name: str
    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        _identifier(self.name, "image.name")
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("image.mime_type must be image/jpeg or image/png")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("image.data must contain encoded image bytes")


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """One inference request for a session at a given step.

    The fields are deliberately model-neutral: an instruction, a flat state mapping, the
    camera images, and the session and step identity the service needs to order requests
    idempotently. ``reset`` starts a new episode for a recurrent policy, and ``metadata``
    carries whatever a specific adapter needs without widening the contract.
    """

    session_id: str
    request_id: str
    step_id: int
    instruction: str
    state: Mapping[str, Any]
    images: Sequence[ImagePayload]
    reset: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.session_id, "session_id")
        _identifier(self.request_id, "request_id")
        if isinstance(self.step_id, bool) or not isinstance(self.step_id, int):
            raise TypeError("step_id must be an integer")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        object.__setattr__(self, "state", _mapping(self.state, "state"))
        images = tuple(self.images)
        if not images or any(not isinstance(image, ImagePayload) for image in images):
            raise ValueError("images must contain at least one ImagePayload")
        object.__setattr__(self, "images", images)
        if not isinstance(self.reset, bool):
            raise TypeError("reset must be a boolean")
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """One action returned by a policy service.

    ``kind`` names the action and ``values`` carries its parameters, so the runtime can
    validate and route an action without knowing which model produced it. An accepted
    action is a request rather than a completed motion: execution and its safety checks
    belong to the runtime. ``from_payload`` builds one from the wire representation.
    """

    kind: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.kind, "action.kind")
        object.__setattr__(self, "values", _mapping(self.values, "action.values"))

    @classmethod
    def from_payload(cls, payload: object) -> PolicyAction:
        value = _mapping(payload, "action")
        kind = value.get("type")
        parameters = value.get("values", {})
        return cls(
            kind=_identifier(kind, "action.type"),
            values=_mapping(parameters, "action.values"),
        )


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """One step response from a policy service.

    It carries the action chunk together with everything needed to account for it: the
    ``request_id`` it answers, the session and the ``session_revision`` it advances, the
    action space the chunk is expressed in, per-stage ``timing``, and the
    ``policy_revision`` that produced it. ``from_payload`` builds one from the wire
    representation.
    """

    request_id: str
    session_id: str
    step_id: int
    session_revision: int
    action_space: str
    actions: Sequence[PolicyAction]
    timing: Mapping[str, float] = field(default_factory=dict)
    policy_revision: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        _identifier(self.session_id, "session_id")
        _identifier(self.action_space, "action_space")
        for name in ("step_id", "session_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        actions = tuple(self.actions)
        if not actions or any(not isinstance(action, PolicyAction) for action in actions):
            raise ValueError("actions must contain at least one PolicyAction")
        object.__setattr__(self, "actions", actions)
        timing: dict[str, float] = {}
        for key, value in _mapping(self.timing, "timing").items():
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"timing.{key} must be finite and non-negative")
            timing[str(key)] = number
        object.__setattr__(self, "timing", timing)

    @classmethod
    def from_payload(cls, payload: object) -> PolicyResult:
        value = _mapping(payload, "step response")
        actions = value.get("actions")
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
            raise TypeError("step response actions must be a list")
        return cls(
            request_id=_identifier(value.get("request_id"), "request_id"),
            session_id=_identifier(value.get("session_id"), "session_id"),
            step_id=int(value.get("step_id")),
            session_revision=int(value.get("session_revision")),
            action_space=_identifier(value.get("action_space"), "action_space"),
            actions=tuple(PolicyAction.from_payload(action) for action in actions),
            timing=_mapping(value.get("timing", {}), "timing"),
            policy_revision=(str(value["policy_revision"]) if value.get("policy_revision") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class Session:
    """Identity and committed revision of one inference session.

    A session groups the steps of one episode so a recurrent policy can carry state between
    calls. ``revision`` is the revision the service reports for that committed state, which
    is what lets a client tell whether it is looking at the state it expects.
    """

    session_id: str
    revision: int

    def __post_init__(self) -> None:
        _identifier(self.session_id, "session_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")


__all__ = [
    "ImagePayload",
    "PolicyObservation",
    "PolicyAction",
    "PolicyResult",
    "Session",
]
