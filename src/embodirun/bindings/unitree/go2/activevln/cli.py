"""CLI for Go2 RealSense -> onboard NX ActiveVLN -> full-chunk motion."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from embodirun.application.navigation.async_chunk_session import AsyncChunkNavigationSession, OverlapConfig
from embodirun.application.navigation.chunk_session import ChunkNavigationSession, ChunkSessionConfig
from embodirun.model_services.backends.embodiinfer.compat import connect_client
from embodirun.robots.unitree.go2.camera import Go2CameraClient
from embodirun.robots.unitree.go2.client import Go2ControlClient

from .interactive import interact
from .policy import ActiveVLNPolicy, check_capabilities


@dataclass(frozen=True)
class NavigationConfig:
    """Connection placeholders and bounded motion settings, with secrets in env vars."""

    robot_url: str = ""
    robot_token_env: str | None = None
    camera_url: str = ""
    control_url: str = ""
    inference_url: str = ""
    robot_id: str = "go2"
    camera_token_env: str = "GO2_CAMERA_TOKEN"
    control_token_env: str = "GO2_API_TOKEN"
    inference_token_env: str = "ACTIVEVLN_BENCHMARK_TOKEN"
    inference_timeout_s: float = 60.0
    camera_timeout_s: float = 2.0
    control_timeout_s: float = 0.5
    mode: str = "overlap"
    session: ChunkSessionConfig = field(default_factory=ChunkSessionConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)

    @classmethod
    def load(cls, path: Path) -> NavigationConfig:
        """Read strict JSON without connecting, installing, or changing the system."""
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("configuration must be a JSON object")
        if not isinstance(value.get("robot_url", ""), str):
            raise ValueError("robot_url must be a URL or an empty placeholder")
        if value.get("robot_url"):
            token_name = value.get("robot_token_env") or "GO2_API_TOKEN"
            for key in ("camera_token_env", "control_token_env"):
                if value.get(key) and value[key] != token_name:
                    raise ValueError(f"{key} conflicts with robot_token_env")
                value[key] = token_name
            for endpoint in ("camera_url", "control_url"):
                if not isinstance(value.get(endpoint, ""), str):
                    raise ValueError(f"{endpoint} must be a URL or an empty placeholder")
                if value.get(endpoint) and value[endpoint].rstrip("/") != value["robot_url"].rstrip("/"):
                    raise ValueError(f"{endpoint} conflicts with robot_url")
                value[endpoint] = value["robot_url"]
        session = ChunkSessionConfig(**value.pop("session", {}))
        overlap = OverlapConfig(**value.pop("overlap", {}))
        config = cls(**value, session=session, overlap=overlap)
        for name in ("camera_token_env", "control_token_env", "inference_token_env"):
            if not isinstance(getattr(config, name), str) or not getattr(config, name):
                raise ValueError(f"{name} must name an environment variable")
        if config.mode not in ("chunk", "overlap"):
            raise ValueError("mode must be chunk or overlap")
        for name in ("inference_timeout_s", "camera_timeout_s", "control_timeout_s"):
            number = getattr(config, name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("robot_url", "camera_url", "control_url", "inference_url"):
            value = getattr(config, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a URL or an empty placeholder")
            parsed = urlsplit(value)
            if value and (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in ("", "/")
            ):
                raise ValueError(f"{name} must be an HTTP service root without credentials")
        return config

    def require_endpoints(self) -> None:
        """Fail before any network access while a deployment address is blank."""
        missing = [name for name in ("camera_url", "control_url", "inference_url") if not getattr(self, name)]
        if missing:
            raise ValueError("fill deployment endpoints first: " + ", ".join(missing))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodirun go2 activevln", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("plan", "inspect configuration offline; blank endpoints are allowed"),
        ("check", "read camera, robot state and model capabilities without motion"),
        ("run", "run navigation; defaults to one observation/inference without motion"),
        ("interactive", "accept successive instructions using the running model and owner services"),
    ):
        command = sub.add_parser(name, help=description)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--mode", choices=("chunk", "overlap"), help="override synchronous or overlapping execution"
        )
        command.add_argument(
            "--overlap-lead-s", type=float, help="override seconds before motion ends to start inference"
        )
        if name == "run":
            command.add_argument("--instruction", required=True)
        if name in {"run", "interactive"}:
            command.add_argument("--execute", action="store_true", help="enable motion on an already armed owner")
            command.add_argument("--output-jsonl", type=Path, help="create a new event log; never overwrite")
            command.add_argument("--max-chunks", type=int, help="override the per-instruction chunk limit")
            command.add_argument("--max-runtime-s", type=float, help="override the per-instruction time limit")
    sub.add_parser("edge", help="forward to the existing Go2 probe/install/start/status/stop CLI")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Connect existing edge/model services and run navigation with explicit motion opt-in."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "edge":
        from embodirun.deployment.go2.cli import main as edge_main

        return edge_main(arguments[1:])
    args = _parser().parse_args(arguments)
    try:
        config = NavigationConfig.load(args.config)
        if args.mode is not None:
            config = replace(config, mode=args.mode)
        if args.overlap_lead_s is not None:
            config = replace(config, overlap=replace(config.overlap, lead_s=args.overlap_lead_s))
        if args.command in {"run", "interactive"}:
            overrides = {
                key: getattr(args, key) for key in ("max_chunks", "max_runtime_s") if getattr(args, key) is not None
            }
            config = replace(config, session=replace(config.session, **overrides))
        if args.command == "plan":
            print(json.dumps({"configuration": asdict(config), "motion_enabled": False}, indent=2))
            return 0
        config.require_endpoints()
        if args.command == "run" and not args.instruction.strip():
            raise ValueError("instruction must be non-empty")
        client = connect_client(
            config.inference_url,
            os.environ.get(config.inference_token_env),
            timeout_s=config.inference_timeout_s,
        )
        capabilities = check_capabilities(client, overlap=config.mode == "overlap")
        camera = Go2CameraClient(
            config.camera_url,
            token=os.environ.get(config.camera_token_env),
            timeout_s=config.camera_timeout_s,
            require_depth=False,
        )
        base = Go2ControlClient(
            config.control_url,
            token=os.environ.get(config.control_token_env),
            timeout_s=config.control_timeout_s,
        )
        if args.command == "check":
            state = base.state()
            observation = camera.capture(episode_id="go2-check", reset=True)
            print(
                json.dumps(
                    {
                        "camera_sequence": observation.sequence,
                        "depth_available": observation.depth is not None,
                        "state_sequence": state.sequence,
                        "active_lease": state.active_velocity_lease_id,
                        "inference": capabilities,
                        "motion_enabled": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        with ExitStack() as stack:
            output = None
            output_lock = threading.Lock()
            if args.output_jsonl:
                args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
                output = stack.enter_context(args.output_jsonl.open("x", encoding="utf-8"))

            def emit(event: dict[str, object]) -> None:
                line = json.dumps(event, ensure_ascii=False, allow_nan=False)
                with output_lock:
                    print(line, flush=True)
                    if output is not None:
                        output.write(line + "\n")
                        output.flush()

            def interrupt(_signum: int, _frame: object) -> None:
                # Termination must exit the process and disarm, not return to
                # the interactive prompt as an operator's Ctrl+C would.
                raise SystemExit(128 + signal.SIGTERM)

            old_handler = signal.signal(signal.SIGTERM, interrupt)
            stack.callback(signal.signal, signal.SIGTERM, old_handler)
            interactive = args.command == "interactive"
            if interactive and args.execute:
                base.preflight(require_release=True, require_renewal=config.mode == "overlap")

            def run_task(instruction: str) -> dict[str, object]:
                if interactive and args.execute:
                    base.preflight(require_release=True, require_renewal=config.mode == "overlap")
                episode_id = f"go2-{uuid.uuid4().hex}"

                def task_emit(event: dict[str, object]) -> None:
                    emit({**event, "episode_id": episode_id})

                task_emit(
                    {
                        "event": "navigation_started",
                        "instruction": instruction,
                        "execute": args.execute,
                        "mode": config.mode,
                    }
                )
                policy = ActiveVLNPolicy(client, robot_id=config.robot_id, event_sink=task_emit)
                try:
                    if config.mode == "overlap":
                        session = AsyncChunkNavigationSession(
                            policy, camera, base, config=config.session, overlap=config.overlap, event_sink=task_emit
                        )
                    else:
                        session = ChunkNavigationSession(
                            policy, camera, base, config=config.session, event_sink=task_emit
                        )
                    return session.run(
                        instruction,
                        execute=args.execute,
                        episode_id=episode_id,
                        keep_motion_ready=interactive and args.execute,
                    )
                except KeyboardInterrupt:
                    task_emit({"event": "navigation_cancelled"})
                    raise
                finally:
                    # New remote session per instruction resets KV/history only;
                    # the external inference process keeps model weights loaded.
                    policy.close()

            if interactive:
                print(
                    f"Motion {'ENABLED' if args.execute else 'disabled (dry-run)'}. "
                    f"Mode: {config.mode}. "
                    f"Task bounds: {config.session.max_chunks} chunks, {config.session.max_runtime_s:g} s "
                    "(in-flight inference cleanup may take longer).",
                    flush=True,
                )
                return interact(run_task)
            run_task(args.instruction)
        return 0
    except KeyboardInterrupt:
        print("navigation interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"navigation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
