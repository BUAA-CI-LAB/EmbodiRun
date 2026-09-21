"""CLI registration for deployment initialization operations."""

from __future__ import annotations

import argparse
from typing import Any

from embodirun.deployment.operations.init import (
    DEFAULT_MANAGED_ROOT,
    DEPLOY_REPOSITORY,
    INFERENCE_REPOSITORY,
    InitError,
    NodeInitialization,
    initialize,
)

from ..context import CommandContext


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "init",
        help="probe nodes and synchronize locked deployment environments",
    )
    parser.add_argument(
        "--root",
        default=".local/share/rlinf-deploy",
        help="managed directory on each node, relative to its home by default",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    return initialize(context, args.root)


__all__ = [
    "DEFAULT_MANAGED_ROOT",
    "DEPLOY_REPOSITORY",
    "INFERENCE_REPOSITORY",
    "InitError",
    "NodeInitialization",
    "register",
    "run",
]
