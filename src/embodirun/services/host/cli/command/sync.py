"""CLI registration for synchronizing local Deploy source."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from embodirun.deployment.operations.sync import (
    NodeSync,
    SyncError,
    synchronize_deploy,
)

from ..context import CommandContext


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "sync",
        help="synchronize local Deploy code without restarting inference",
    )
    parser.add_argument(
        "--target",
        choices=("deploy",),
        default="deploy",
        help="project to synchronize (default: deploy)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="local EmbodiRun repository root (default: current directory)",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    return synchronize_deploy(args.source, context)


__all__ = ["NodeSync", "SyncError", "register", "run"]
