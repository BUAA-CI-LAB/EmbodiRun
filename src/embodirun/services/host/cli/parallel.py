"""Compatibility exports for canonical deployment operation concurrency."""

from embodirun.deployment.operations._parallel import ParallelResults, run_on_nodes

__all__ = ["ParallelResults", "run_on_nodes"]
