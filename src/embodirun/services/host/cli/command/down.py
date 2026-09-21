"""CLI registration for stopping deployment services."""

from __future__ import annotations

import argparse
from typing import Any

from embodirun.deployment.operations.down import (
    DownError,
    NodeDownResult,
    stop_services,
)

from ..context import CommandContext


def register(commands: Any) -> None:
    parser = commands.add_parser("down", help="stop initialized services")
    parser.add_argument(
        "--target",
        choices=("all", "control", "simulation", "model"),
        default="all",
        help="service kind to stop (default: all)",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    return stop_services(context, args.target)


__all__ = ["DownError", "NodeDownResult", "register", "run"]
