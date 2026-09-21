"""Small type aliases shared across runtime domains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

TensorTree: TypeAlias = Any
Metadata: TypeAlias = Mapping[str, Any]
Shape: TypeAlias = Sequence[int | str]

__all__ = ["Metadata", "Shape", "TensorTree"]
