"""Launch a local Quest station or an AGX observation/control endpoint."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path


def prepare(directory: Path, hosts: list[str]) -> dict:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = {name: directory / name for name in ("cert.pem", "key.pem", "browser-token", "robot-token")}
    if any(path.exists() for path in paths.values()):
        raise ValueError("credentials already exist; reuse them or select a new directory")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names = []
    for host in dict.fromkeys(["localhost", "127.0.0.1", *hosts]):
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Quest XLeRobot local station")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )
    values = {
        "key.pem": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        "cert.pem": cert.public_bytes(serialization.Encoding.PEM),
        "browser-token": secrets.token_urlsafe(24).encode() + b"\n",
        "robot-token": secrets.token_urlsafe(32).encode() + b"\n",
    }
    for name, content in values.items():
        fd = os.open(paths[name], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
    return {name: str(path.resolve()) for name, path in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare", help="generate local HTTPS certificate and private access tokens")
    setup.add_argument("--directory", type=Path, required=True)
    setup.add_argument("--host", action="append", default=[])
    serve = sub.add_parser("serve", help="serve Mac browser UI or AGX endpoint")
    serve.add_argument("--mode", choices=("demo", "remote", "hardware"), default="demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8443)
    serve.add_argument("--token-file", type=Path, required=True)
    serve.add_argument(
        "--browser-no-token-cidr",
        help="allow browser access without pairing from this private subnet only; robot API still requires its token",
    )
    serve.add_argument("--ssl-cert", type=Path)
    serve.add_argument("--ssl-key", type=Path)
    serve.add_argument("--hardware-config", type=Path)
    resume = serve.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-held-state", type=Path, help="one-use stopped arm handoff; no torque writes or automatic arm"
    )
    resume.add_argument(
        "--resume-stop-fault", type=Path, help="one-use blocked fault handoff; reads only, never clears a fault"
    )
    serve.add_argument("--mapping-config", type=Path)
    serve.add_argument("--robot-url")
    serve.add_argument("--robot-scope", choices=("all", "arms", "base"), default="all")
    serve.add_argument("--robot-token-file", type=Path)
    serve.add_argument(
        "--leader-command",
        type=Path,
        help="fixed local dual-leader launcher exposed as an explicit browser button",
    )
    serve.add_argument("--output", type=Path, default=Path("artifacts/quest/episodes"))
    serve.add_argument("--fps", type=float, default=20)
    serve.add_argument("--pair", action="store_true", help="print a single-use 6-digit browser code (10 minutes)")
    serve.add_argument("--robot-api", action="store_true", help="expose authenticated robot endpoint")
    validate = sub.add_parser("validate", help="inspect episode completeness and training eligibility")
    validate.add_argument("episode", type=Path)
    export = sub.add_parser("export", help="export stationary dual-arm episodes through LeRobotDataset")
    export.add_argument("episodes", nargs="+", type=Path)
    export.add_argument("--repo-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--fps", type=float, default=20)
    export.add_argument(
        "--allow-demo",
        action="store_true",
        help="explicit synthetic dataset for software tests only",
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.directory, args.host), indent=2))
        return 0
    if args.command == "validate":
        from .recording import validate_episode

        result = validate_episode(args.episode)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("trainable") else 1
    if args.command == "export":
        from .recording import export_lerobot

        result = export_lerobot(
            args.episodes,
            repo_id=args.repo_id,
            output=args.output,
            fps=args.fps,
            allow_demo=args.allow_demo,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    from aiohttp import web

    from .control import MappingConfig
    from .robot import DemoRobot, RemoteRobot
    from .server import Platform, create_app

    token = args.token_file.read_text().strip()
    if (args.resume_held_state or args.resume_stop_fault) and args.mode != "hardware":
        parser.error("restart handoffs are hardware-only")
    tls = None
    if args.ssl_cert and args.ssl_key:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(args.ssl_cert, args.ssl_key)
    elif args.ssl_cert or args.ssl_key:
        parser.error("--ssl-cert and --ssl-key must be supplied together")
    if args.host not in ("localhost", "127.0.0.1", "::1") and tls is None:
        parser.error("LAN browser serving requires HTTPS; prepare a local certificate first")
    if args.mode == "demo":
        robot = DemoRobot()
    elif args.mode == "remote":
        if not args.robot_url or not args.robot_token_file:
            parser.error("remote mode requires --robot-url and --robot-token-file")
        robot = RemoteRobot(args.robot_url, args.robot_token_file.read_text().strip(), scope=args.robot_scope)
    else:
        if not args.hardware_config:
            parser.error("hardware mode requires --hardware-config")
        from .hardware import HardwareRobot

        config = json.loads(args.hardware_config.read_text())
        snapshot = args.resume_stop_fault or args.resume_held_state
        if snapshot:
            # Claim the one-use file before validation; failed attempts remain
            # available for audit but cannot be accidentally replayed.
            used = snapshot.with_suffix(snapshot.suffix + ".used")
            if used.exists():
                parser.error("restart snapshot was already claimed")
            snapshot.rename(used)
            config["_resume_stop_fault" if args.resume_stop_fault else "_resume_held_state"] = json.loads(
                used.read_text()
            )
        robot = HardwareRobot(config)
    if args.leader_command:
        if args.mode != "remote" or args.robot_scope != "base":
            parser.error("--leader-command requires remote mode with --robot-scope base")
        if not args.leader_command.is_file():
            parser.error(f"leader command does not exist: {args.leader_command}")
        if not os.access(args.leader_command, os.X_OK):
            parser.error(f"leader command is not executable: {args.leader_command}")
    mapping = MappingConfig(**json.loads(args.mapping_config.read_text())) if args.mapping_config else MappingConfig()
    platform = Platform(
        robot,
        token,
        output=args.output,
        fps=args.fps,
        mapping=mapping,
        robot_api=args.robot_api,
        browser_no_token_cidr=args.browser_no_token_cidr,
        leader_command=(str(args.leader_command.resolve()), "--no-run-prompt") if args.leader_command else None,
    )
    if args.pair:
        print(
            f"Browser pairing code: {platform.issue_pairing_code()} (one use; 10 minutes)",
            flush=True,
        )
    web.run_app(create_app(platform), host=args.host, port=args.port, ssl_context=tls, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
