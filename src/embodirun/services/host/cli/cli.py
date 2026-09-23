"""Integrate independently implemented deployment subcommands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from embodirun.deployment.config import ConfigError, NodeConfig, load_config
from embodirun.deployment.environment import EnvironmentError
from embodirun.deployment.executor import Executor, executor_for
from embodirun.deployment.plan import ServiceError, build_plan

from .command import control, down, init, probe, run, sync, up, validate
from .context import CommandContext, state_path
from .progress import ConsoleProgressReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodirun")
    parser.add_argument("--config", required=True, help="deployment YAML path")
    parser.add_argument(
        "--state-dir",
        type=Path,
        help=("local state directory (default: ~/.local/state/embodirun, kept for compatibility)"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate.register(commands)
    probe.register(commands)
    init.register(commands)
    sync.register(commands)
    up.register(commands)
    down.register(commands)
    run.register(commands)
    control.register(commands)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    executor_factory: Callable[[NodeConfig], Executor] = executor_for,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        deployment = build_plan(config)
    except (ConfigError, EnvironmentError, ServiceError, RuntimeError) as error:
        print(f"embodirun: error: {error}", file=sys.stderr)
        return 2

    context = CommandContext(
        config=config,
        deployment=deployment,
        state_path=state_path(args.state_dir, deployment.name),
        executor_factory=executor_factory,
        progress=ConsoleProgressReporter(),
    )
    try:
        return args.command_handler(args, context)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"embodirun: error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
