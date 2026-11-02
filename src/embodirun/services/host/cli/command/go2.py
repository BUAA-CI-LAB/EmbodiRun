"""Route Go2 policy bindings through the installed EmbodiRun command."""

from __future__ import annotations

import argparse
from typing import Any


def register(commands: Any) -> None:
    """Register the robot/model path without importing hardware dependencies."""
    parser = commands.add_parser("go2", help="Go2 navigation and edge services")
    bindings = parser.add_subparsers(dest="binding", required=True)
    for name, help_text in (("activevln", "ActiveVLN navigation"), ("edge", "Go2 hardware owner deployment")):
        parser = bindings.add_parser(name, add_help=False, help=help_text)
        parser.add_argument("binding_args", nargs=argparse.REMAINDER)


def run(args: argparse.Namespace) -> int:
    """Invoke the binding without requiring deployment YAML."""
    if args.binding == "edge":
        from embodirun.deployment.go2.cli import main
    else:
        from embodirun.bindings.unitree.go2.activevln.cli import main

    return main(args.binding_args)
