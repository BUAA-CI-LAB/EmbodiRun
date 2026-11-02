"""Language-conditioned navigation policy request."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import NavigationContractError
from .observation import NavigationObservation


@dataclass(frozen=True, slots=True)
class NavigationRequest:
    instruction: str
    observation: NavigationObservation

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise NavigationContractError("instruction must be a non-empty string")
        if len(self.instruction) > 10_000:
            raise NavigationContractError("instruction exceeds 10000 characters")
        if not isinstance(self.observation, NavigationObservation):
            raise NavigationContractError("observation must be a NavigationObservation")


__all__ = ["NavigationRequest"]
