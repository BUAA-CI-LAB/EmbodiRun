"""Argument and environment parsers for the Go2 camera CLI."""

from __future__ import annotations

import argparse
import re
from typing import Mapping, Optional, Tuple  # noqa: UP035

ENV_PREFIX = "GO2_CAMERA_"
_PROFILE_PATTERN = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)@(?P<fps>\d+)$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def port_number(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return parsed


def camera_profile(value: str) -> Tuple[int, int, int]:  # noqa: UP006
    """Parse a compact ``WIDTHxHEIGHT@FPS`` profile."""

    match = _PROFILE_PATTERN.fullmatch(value.strip().lower())
    if match is None:
        raise argparse.ArgumentTypeError("must use WIDTHxHEIGHT@FPS, for example 640x360@15")
    width = int(match.group("width"))
    height = int(match.group("height"))
    fps = int(match.group("fps"))
    if width <= 0 or height <= 0 or fps <= 0:
        raise argparse.ArgumentTypeError("profile dimensions and FPS must be positive")
    return width, height, fps


def environment_bool(environment: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, on/off, or 1/0")


def environment_value(
    environment: Mapping[str, str],
    name: str,
    default: Optional[str] = None,  # noqa: UP045
) -> Optional[str]:  # noqa: UP045
    return environment.get(f"{ENV_PREFIX}{name}", default)


def add_boolean_option(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: Optional[str] = None,  # noqa: UP045
) -> None:
    """Add ``--flag``/``--no-flag`` without Python 3.9-only argparse APIs."""

    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{destination: default})


__all__ = [
    "ENV_PREFIX",
    "add_boolean_option",
    "camera_profile",
    "environment_bool",
    "environment_value",
    "port_number",
    "positive_float",
    "positive_int",
]
