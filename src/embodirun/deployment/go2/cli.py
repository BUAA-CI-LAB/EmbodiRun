"""Command-line interface for deploying the Go2 edge agent over SSH."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .deployer import Go2OwnerDeployer
from .transport import ParamikoTransport, RemoteTransport, SshConnection

TransportFactory = Callable[[SshConnection], RemoteTransport]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodirun go2 edge", description="Install and manage the unified Go2 hardware owner."
    )
    parser.add_argument("--host", required=True, help="SSH hostname or IP address")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--username", required=True, help="SSH username")
    parser.add_argument(
        "--remote-root",
        default=".local/share/embodirun-go2-owner",
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

    install = commands.add_parser("install", help="upload the optional Go2 owner source")
    install.add_argument(
        "--package-root",
        type=Path,
        help="override the local embodirun_go2_owner package root (primarily for packaging tests)",
    )

    start = commands.add_parser("start", help="start the single Go2 hardware owner")
    start.add_argument("--config", type=Path, required=True, help="local owner JSON configuration")
    start.add_argument("--python", default="python3", help="owner Python (3.8+; vendor SDK environment)")
    start.add_argument("--sdk-pythonpath", help="remote SDK source directory, if not installed")
    start.add_argument("--operator-ready", action="store_true", help="explicitly arm motion at process startup")
    start.add_argument("--supervisor-python", default="python3.10", help="supervisor Python (3.10+)")
    for name in ("status", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--supervisor-python", default="python3.10")

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
        deployer = Go2OwnerDeployer(transport, args.remote_root)
        result: dict[str, object]
        if args.command == "probe":
            result = {"ok": True, "remote": deployer.probe(args.python)}
        elif args.command == "install":
            result = {"ok": True, "install": deployer.install(args.package_root)}
        elif args.command == "start":
            settings = json.loads(args.config.read_text())
            if not isinstance(settings, dict) or not isinstance(settings.get("server", {}), dict):
                raise ValueError("owner configuration and server must be objects")
            token_name = settings.get("server", {}).get("token_env", "GO2_API_TOKEN")
            if not isinstance(token_name, str):
                raise ValueError("server.token_env must name an environment variable")
            token = environment.get(token_name)
            secrets.append(token)
            result = {
                "ok": True,
                **deployer.start(
                    args.config,
                    python=args.python,
                    supervisor_python=args.supervisor_python,
                    sdk_pythonpath=args.sdk_pythonpath,
                    operator_ready=args.operator_ready,
                    token=token,
                ),
            }
        elif args.command == "status":
            result = {"ok": True, **deployer.status(args.supervisor_python)}
        elif args.command == "stop":
            result = {"ok": True, **deployer.stop(args.supervisor_python)}
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


def _redact_many(text: str, secrets: Sequence[str | None]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
