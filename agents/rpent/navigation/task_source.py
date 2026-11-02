"""Literal task references; goal boundaries and order remain model decisions."""

from __future__ import annotations

import re


def indexed_task(task: str) -> str:
    """Number whitespace-delimited words without interpreting task semantics."""
    return "\n".join(f"{index}: {word.group()}" for index, word in enumerate(re.finditer(r"\S+", task), 1))


def select_task_spans(task: str, spans: list[list[int]]) -> list[str]:
    """Copy model-selected inclusive word ranges, preserving interior whitespace.

    Ranges are one-based. Their supplied order, including repeated visits, is
    preserved. Validation never invents a boundary, drops an invalid range, or
    substitutes the whole instruction when the model supplied an invalid plan.
    """
    words = list(re.finditer(r"\S+", task))
    if not isinstance(spans, list) or not 1 <= len(spans) <= 16:
        raise ValueError("spans must contain 1-16 [first_word,last_word] ranges")
    goals = []
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(index) is not int for index in span)
            or not 1 <= span[0] <= span[1] <= len(words)
        ):
            raise ValueError(f"each span must select existing words with 1 <= first <= last <= {len(words)}")
        goals.append(task[words[span[0] - 1].start() : words[span[1] - 1].end()])
    return goals
