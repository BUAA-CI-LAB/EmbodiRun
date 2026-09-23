"""Command-line interface for deploying the Go2 edge agent over SSH."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from .deployer import Go2AgentDeployer
from .options import (
    CameraStartOptions,
    ControlStartOptions,
    ServiceSelection,
    StartOptions,
)
from .transport import ParamikoTransport, RemoteTransport, SshConnection

TransportFactory = Callable[[SshConnection], RemoteTransport]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and manage repository-owned services on a Unitree Go2 computer."
    )
    parser.add_argument("--host", required=True, help="SSH hostname or IP address")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--username", required=True, help="SSH username")
    parser.add_argument(
        "--remote-root",
        default=".local/share/embodirun-go2-agent",
        help="remote installation root, absolute or relative to the SSH user's home",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument(
        "--accept-new-host-key",
        action="store_true",
        help="trust and persist a previously unseen SSH host key",
    )
    parser.add_argument("--identity-file", type=Path, help="SSH private-key path")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument(
        "--password",
        help="SSH password (prefer --password-env or --ask-password to avoid shell history)",
    )
    auth.add_argument("--password-env", metavar="NAME", help="environment variable with password")
    auth.add_argument("--ask-password", action="store_true", help="prompt without echo")

    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="verify SSH and the remote Python runtime")
    probe.add_argument("--python", default="python3", help="remote Python executable")

    install = commands.add_parser("install", help="upload the current embodirun source")
    install.add_argument(
        "--package-root",
        type=Path,
        help="override the local embodirun package root (primarily for packaging tests)",
    )

    start = commands.add_parser("start", help="start control and/or camera services")
    _add_service_selection(start)
    start.add_argument("--python", default="python3", help="remote Python executable")
    start.add_argument("--control-mode", choices=("live", "dry-run"), default="live")
    start.add_argument("--interface", default="eno2", help="Unitree DDS network interface")
    start.add_argument("--state-topic", default="rt/lf/sportmodestate")
    start.add_argument(
        "--cyclonedds-lib-dir",
        help="remote directory containing the compatible libddsc.so.0",
    )
    start.add_argument(
        "--control-bind",
        help="control bind address (default: the SSH --host value)",
    )
    start.add_argument("--control-port", type=int, default=8080)
    start.add_argument("--api-token", help="control API bearer token")
    start.add_argument(
        "--api-token-env",
        metavar="NAME",
        default="GO2_API_TOKEN",
        help="token environment variable (default: GO2_API_TOKEN)",
    )
    start.add_argument(
        "--operator-ready",
        action="store_true",
        help="explicitly arm live posture/motion commands; omitted by default",
    )
    start.add_argument("--camera-backend", choices=("realsense", "v4l2"), default="realsense")
    start.add_argument("--realsense-serial")
    start.add_argument(
        "--depth-scale",
        type=float,
        help="verified meters per RealSense raw depth unit",
    )
    start.add_argument(
        "--camera-bind",
        help="camera bind address (default: the SSH --host value)",
    )
    start.add_argument("--camera-port", type=int, default=8765)
    start.add_argument("--camera-token", help="camera API bearer token")
    start.add_argument(
        "--camera-token-env",
        metavar="NAME",
        default="GO2_CAMERA_TOKEN",
        help="token environment variable (default: GO2_CAMERA_TOKEN)",
    )
    start.add_argument("--camera-width", type=int, default=640)
    start.add_argument("--camera-height", type=int, default=360)
    start.add_argument("--camera-fps", type=int, default=15)
    start.add_argument("--jpeg-fps", type=float, default=5.0)
    start.add_argument("--camera-device", default="/dev/video4")

    status = commands.add_parser("status", help="show PID-backed service status")
    _add_service_selection(status)
    stop = commands.add_parser("stop", help="stop control and/or camera services")
    _add_service_selection(stop)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    password_prompt: Callable[[str], str] = getpass.getpass,
    transport_factory: TransportFactory = ParamikoTransport,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    environment = os.environ if environ is None else environ

    password = _resolve_password(args, environment, password_prompt)
    secrets = [password]

    transport: RemoteTransport | None = None
    try:
        connection = SshConnection(
            host=args.host,
            port=args.ssh_port,
            username=args.username,
            password=password,
            identity_file=args.identity_file,
            connect_timeout_s=args.connect_timeout,
            command_timeout_s=args.command_timeout,
            accept_new_host_key=args.accept_new_host_key,
        )
        transport = transport_factory(connection)
        deployer = Go2AgentDeployer(transport, args.remote_root)
        result: dict[str, object]
        if args.command == "probe":
            result = {"ok": True, "remote": deployer.probe(args.python)}
        elif args.command == "install":
            result = {"ok": True, "install": deployer.install(args.package_root)}
        elif args.command == "start":
            api_token = _resolve_named_secret(args.api_token, args.api_token_env, environment)
            camera_token = _resolve_named_secret(
                args.camera_token,
                args.camera_token_env,
                environment,
            )
            secrets.extend([api_token, camera_token])
            options = _start_options(args, api_token=api_token, camera_token=camera_token)
            result = {"ok": True, **deployer.start(options)}
        elif args.command == "status":
            result = {"ok": True, **deployer.status(args.service)}
        elif args.command == "stop":
            result = {"ok": True, **deployer.stop(args.service)}
        else:  # pragma: no cover - protected by argparse
            raise RuntimeError(f"unsupported command: {args.command}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        message = _redact_many(str(exc), secrets)
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if transport is not None:
            transport.close()

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _add_service_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--service", choices=("all", "control", "camera"), default="all")


def _resolve_password(
    args: argparse.Namespace,
    environment: Mapping[str, str],
    password_prompt: Callable[[str], str],
) -> str | None:
    if args.password is not None:
        return args.password
    if args.ask_password:
        return password_prompt(f"SSH password for {args.username}@{args.host}: ")
    variable = args.password_env or "GO2_SSH_PASSWORD"
    return environment.get(variable)


def _resolve_named_secret(
    literal: str | None,
    variable: str | None,
    environment: Mapping[str, str],
) -> str | None:
    if literal is not None:
        return literal
    if variable:
        return environment.get(variable)
    return None


def _start_options(
    args: argparse.Namespace,
    *,
    api_token: str | None,
    camera_token: str | None,
) -> StartOptions:
    service = cast(ServiceSelection, args.service)
    control = None
    if service in {"all", "control"}:
        control = ControlStartOptions(
            mode=args.control_mode,
            interface=args.interface,
            state_topic=args.state_topic,
            cyclonedds_lib_dir=args.cyclonedds_lib_dir,
            bind=args.control_bind or args.host,
            port=args.control_port,
            api_token=api_token,
            operator_ready=args.operator_ready,
        )
    camera = None
    if service in {"all", "camera"}:
        camera = CameraStartOptions(
            backend=args.camera_backend,
            realsense_serial=args.realsense_serial,
            depth_scale=args.depth_scale,
            bind=args.camera_bind or args.host,
            port=args.camera_port,
            camera_token=camera_token,
            width=args.camera_width,
            height=args.camera_height,
            camera_fps=args.camera_fps,
            jpeg_fps=args.jpeg_fps,
            device=args.camera_device,
        )
    return StartOptions(
        python=args.python,
        services=service,
        control=control,
        camera=camera,
    )


def _redact_many(text: str, secrets: Sequence[str | None]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
