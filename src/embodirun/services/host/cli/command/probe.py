"""Probe configured nodes without changing them."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from embodirun.deployment.config import NodeConfig
from embodirun.deployment.probe import probe_node

from ..context import CommandContext
from ..parallel import run_on_nodes


@dataclass(frozen=True, slots=True)
class NodeProbeResult:
    """Connectivity and base-tool status for one configured node."""

    node_id: str
    node_type: str
    connection: str
    address: str
    reachable: bool
    latency_ms: float
    init_ready: bool = False
    platform: str | None = None
    machine: str | None = None
    python: str | None = None
    python_version: str | None = None
    git: str | None = None
    uv: str | None = None
    error: str | None = None


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "probe",
        help="check connectivity and base tools on every configured node",
    )
    parser.set_defaults(command_handler=run)


def run(_args: argparse.Namespace, context: CommandContext) -> int:
    progress = context.progress
    progress.begin("probe", context.deployment.name)
    for node_id in sorted(context.config.nodes):
        progress.add_node(node_id, total=1)
    probes = _probe_nodes(context)
    reachable = all(probe.reachable for probe in probes)
    progress.finish(success=reachable)
    for probe in probes:
        if probe.reachable:
            readiness = "init ready" if probe.init_ready else "missing init tools"
            progress.message(
                f"{probe.node_id}: {probe.address} · {probe.platform}/{probe.machine} · "
                f"Python {probe.python_version or 'not found'} · {readiness}"
            )
    return 0 if reachable else 1


def _probe_nodes(
    context: CommandContext,
) -> tuple[NodeProbeResult, ...]:
    def probe(node_id: str) -> NodeProbeResult:
        node = context.config.nodes[node_id]
        started = time.monotonic()
        context.progress.update(node_id, "Connecting and probing")
        try:
            with context.executor(node_id) as executor:
                result = probe_node(executor)
        except (OSError, RuntimeError, ValueError) as error:
            context.progress.fail(node_id, error)
            return NodeProbeResult(
                node_id=node_id,
                node_type=node.kind,
                connection=node.connection.kind,
                address=_node_address(node),
                reachable=False,
                latency_ms=_elapsed_ms(started),
                error=str(error),
            )
        init_ready = all((result.python, result.python_version, result.git, result.uv))
        latency_ms = _elapsed_ms(started)
        context.progress.advance(node_id)
        context.progress.succeed(
            node_id,
            detail=f"Reachable in {latency_ms:g} ms",
        )
        return NodeProbeResult(
            node_id=node_id,
            node_type=node.kind,
            connection=node.connection.kind,
            address=_node_address(node),
            reachable=True,
            latency_ms=latency_ms,
            init_ready=init_ready,
            platform=result.platform,
            machine=result.machine,
            python=result.python,
            python_version=result.python_version,
            git=result.git,
            uv=result.uv,
        )

    results = run_on_nodes(context.config.nodes, probe)
    if results.errors:
        raise RuntimeError("probe worker failed without a node result")
    return tuple(results.values[node_id] for node_id in sorted(results.values))


def _node_address(node: NodeConfig) -> str:
    connection = node.connection
    if connection.kind == "local":
        return "local"
    return f"{connection.host}:{connection.port}"


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 3)


__all__ = ["NodeProbeResult", "register", "run"]
