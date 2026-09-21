"""Public entry point for the deployment command-line interface."""

from .cli import build_parser, main

__all__ = ["build_parser", "main"]
