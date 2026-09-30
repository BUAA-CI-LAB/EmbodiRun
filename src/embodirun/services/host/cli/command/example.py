"""Expose the source checkout's shared example launcher through embodirun."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def register(commands: Any) -> None:
    parser = commands.add_parser("example", help="prepare and run a recipe from a source checkout")
    parser.add_argument(
        "example_args", nargs=argparse.REMAINDER, help="init TEMPLATE [--output DIR], or MANIFEST COMMAND [options]"
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace) -> int:
    """Use exactly the launcher used by examples/run.sh, without a second dispatcher."""
    candidates = [Path(__file__).resolve().parents[6], Path.cwd()]
    if source := os.environ.get("EMBODIRUN_SOURCE_ROOT"):
        candidates.insert(0, Path(source).expanduser())
    for root in candidates:
        if (root / "examples/runner.py").is_file() and (root / "pyproject.toml").is_file():
            sys.path.insert(0, str(root))
            from examples.runner import main

            return main(args.example_args)
    raise RuntimeError("Recipe sources are missing. Install from a checkout or set EMBODIRUN_SOURCE_ROOT to it.")
