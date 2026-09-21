"""Report successful configuration and deployment-plan validation."""

from __future__ import annotations

import argparse
from typing import Any

from ..context import CommandContext


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "validate",
        help="validate configuration and deployment relationships",
    )
    parser.set_defaults(command_handler=run)


def run(_args: argparse.Namespace, context: CommandContext) -> int:
    print(f"configuration is valid: {context.deployment.name}")
    return 0


__all__ = ["register", "run"]
