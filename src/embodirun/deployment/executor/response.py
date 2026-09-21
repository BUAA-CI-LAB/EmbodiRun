"""Transport-independent HTTP response values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class JsonHttpResponse:
    status: int
    payload: Any


__all__ = ["JsonHttpResponse"]
