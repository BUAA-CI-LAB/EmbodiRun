"""Concurrent node operation helper used by deployment operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Generic, TypeVar

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ParallelResults(Generic[ResultT]):
    values: dict[str, ResultT]
    errors: dict[str, Exception]


def run_on_nodes(node_ids: Iterable[str], operation: Callable[[str], ResultT]) -> ParallelResults[ResultT]:
    """Run one independent operation per node and wait for every result."""
    nodes = tuple(sorted(node_ids))
    if not nodes:
        return ParallelResults({}, {})
    values: dict[str, ResultT] = {}
    errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=len(nodes), thread_name_prefix="embodirun-node") as pool:
        futures = {pool.submit(operation, node_id): node_id for node_id in nodes}
        for future in as_completed(futures):
            node_id = futures[future]
            try:
                values[node_id] = future.result()
            except Exception as error:  # noqa: BLE001 - preserve per-node failures
                errors[node_id] = error
    return ParallelResults(values, errors)


__all__ = ["ParallelResults", "run_on_nodes"]
