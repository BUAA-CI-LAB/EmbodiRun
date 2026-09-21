"""Command-line entry point for the robot-resident Go2 control service."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence

from .config import MIN_EXTERNAL_TOKEN_CHARS, ControlServerConfig
from .executor import ActionExecutor
from .http import valid_api_token_for_bind
from .sdk2 import UnitreeTransport, ensure_cyclonedds_library_dir
from .server import serve_control_api
from .transport import DryRunTransport, RobotTransport


def _env_bool(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def build_parser(environ: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """Build a parser whose deployment defaults can all come from env vars."""

    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        description="Bounded HTTP control service for a Unitree Go2",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "live"),
        default=env.get("GO2_CONTROL_MODE", "dry-run"),
        help="transport mode (env: GO2_CONTROL_MODE)",
    )
    parser.add_argument(
        "--host",
        default=env.get("GO2_CONTROL_HOST", "127.0.0.1"),
        help="HTTP bind host/IP (env: GO2_CONTROL_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=env.get("GO2_CONTROL_PORT", "8080"),
        help="HTTP bind port (env: GO2_CONTROL_PORT)",
    )
    parser.add_argument(
        "--interface",
        default=env.get("GO2_DDS_INTERFACE"),
        help="DDS network interface; required in live mode (env: GO2_DDS_INTERFACE)",
    )
    parser.add_argument(
        "--state-topic",
        default=env.get("GO2_STATE_TOPIC", "rt/lf/sportmodestate"),
        help="SportModeState DDS topic (env: GO2_STATE_TOPIC)",
    )
    parser.add_argument(
        "--cyclonedds-lib-dir",
        default=env.get("GO2_CYCLONEDDS_LIB_DIR"),
        help=("directory containing the compatible libddsc.so.0; required in live mode (env: GO2_CYCLONEDDS_LIB_DIR)"),
    )
    parser.add_argument(
        "--token",
        default=env.get("GO2_API_TOKEN"),
        help="bearer token; prefer GO2_API_TOKEN to avoid process-list exposure",
    )
    operator_ready = parser.add_mutually_exclusive_group()
    operator_ready.add_argument(
        "--operator-ready",
        dest="operator_ready",
        action="store_true",
        help=(
            "attest that the robot is standing and its area is clear; required "
            "for live motion (env: GO2_OPERATOR_READY)"
        ),
    )
    operator_ready.add_argument(
        "--no-operator-ready",
        dest="operator_ready",
        action="store_false",
        help="revoke an operator-ready value inherited from the environment",
    )
    parser.set_defaults(operator_ready=_env_bool(env, "GO2_OPERATOR_READY"))
    parser.add_argument(
        "--rpc-timeout",
        type=float,
        default=env.get("GO2_SDK_RPC_TIMEOUT", "2.0"),
        help="SDK2 RPC timeout in seconds (env: GO2_SDK_RPC_TIMEOUT)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ControlServerConfig:
    return ControlServerConfig(
        mode=args.mode,
        host=args.host,
        port=args.port,
        interface=args.interface,
        state_topic=args.state_topic,
        cyclonedds_lib_dir=args.cyclonedds_lib_dir,
        token=args.token,
        operator_ready=args.operator_ready,
        rpc_timeout_s=args.rpc_timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Start the service; suitable for ``python -m ...control``."""

    try:
        parser = build_parser()
        arguments = parser.parse_args(argv)
        config = config_from_args(arguments)
    except (TypeError, ValueError) as exc:
        print(f"invalid Go2 control configuration: {exc}", file=sys.stderr)
        return 2

    if not valid_api_token_for_bind(config.host, config.token):
        print(
            f"--token/GO2_API_TOKEN must contain at least "
            f"{MIN_EXTERNAL_TOKEN_CHARS} non-whitespace characters when binding "
            "beyond localhost",
            file=sys.stderr,
        )
        return 2

    required_dds_lib_dir = None
    if config.mode == "live":
        try:
            assert config.cyclonedds_lib_dir is not None
            required_dds_lib_dir = ensure_cyclonedds_library_dir(
                config.cyclonedds_lib_dir,
                reexec_args=None if argv is None else list(argv),
            )
        except Exception as exc:  # noqa: BLE001 - process boundary reports config failure
            print(f"invalid CycloneDDS library directory: {exc}", file=sys.stderr)
            return 2

    try:
        transport: RobotTransport
        if config.mode == "live":
            assert config.interface is not None
            transport = UnitreeTransport(
                config.interface,
                config.state_topic,
                rpc_timeout_s=config.rpc_timeout_s,
                required_dds_lib_dir=required_dds_lib_dir,
            )
        else:
            transport = DryRunTransport()
    except Exception as exc:  # noqa: BLE001 - process boundary reports SDK failure
        print(f"failed to initialize transport: {exc}", file=sys.stderr)
        return 1

    executor = ActionExecutor(
        transport,
        operator_motion_ready=config.operator_ready or config.mode == "dry-run",
    )
    try:
        serve_control_api(config, executor)
    except Exception as exc:  # noqa: BLE001 - process boundary reports server failure
        print(f"Go2 control server failed: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["build_parser", "config_from_args", "main"]
