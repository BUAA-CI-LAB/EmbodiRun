"""Backend- and protocol-independent inference client contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .contracts import PolicyObservation, PolicyResult, Session


class InferenceClient(Protocol):
    """Manage policy sessions without exposing a backend or protocol."""

    def health(self) -> Mapping[str, Any]: ...

    def capabilities(self) -> Mapping[str, Any]: ...

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session: ...

    def step(self, observation: PolicyObservation) -> PolicyResult: ...

    def reset(self, session_id: str, *, request_id: str) -> Session: ...

    def close(self, session_id: str) -> None: ...


# Preserve the established public name while bindings migrate to the more
# transport-neutral interface name.
PolicyClient = InferenceClient
"""Backward-compatible alias for :class:`InferenceClient`."""


__all__ = ["InferenceClient", "PolicyClient"]
