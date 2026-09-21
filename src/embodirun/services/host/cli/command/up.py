"""CLI registration for starting deployment services."""

from __future__ import annotations

import argparse
from typing import Any

from embodirun.deployment.operations.up import (
    DEFAULT_WAIT_TIMEOUT_S,
    NodeUpResult,
    UpError,
    start_services,
)

from ..context import CommandContext


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "up",
        help="start services and wait until they are healthy",
    )
    parser.add_argument(
        "--wait-timeout",
        type=_positive_seconds,
        default=DEFAULT_WAIT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"maximum service readiness wait (default: {DEFAULT_WAIT_TIMEOUT_S:g})",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    return start_services(context, args.wait_timeout)


def _positive_seconds(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not result > 0 or not result < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


__all__ = ["DEFAULT_WAIT_TIMEOUT_S", "NodeUpResult", "UpError", "register", "run"]
