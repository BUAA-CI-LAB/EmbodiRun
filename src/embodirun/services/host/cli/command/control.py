"""JSON-first Host commands for the versioned Control application API.

These commands only resolve an initialized service and call
``HostControlClient``.  They never import a robot adapter, camera, arbiter, or
model implementation.  A separate invocation can therefore inspect an
accepted request with the same caller/session/request IDs after a transport
timeout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from embodirun.deployment.config import config_digest
from embodirun.deployment.control import ControlClientError, HostControlClient
from embodirun.deployment.plan import RuntimeSpec
from embodirun.deployment.state import StateError, StateStore

from ..context import CommandContext

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_SERVICE = 3
EXIT_UNKNOWN = 4
DEFAULT_TIMEOUT_S = 10.0


class ControlCommandError(RuntimeError):
    """The Host cannot safely resolve a configured Control service."""


def register(commands: Any) -> None:
    """Register application and recorder operations without changing legacy ``run``."""

    describe = commands.add_parser("describe", help="query Control capabilities")
    _common(describe)
    describe.set_defaults(command_handler=describe_command)

    observe = commands.add_parser("observe", help="read one shared observation")
    _common(observe)
    observe.add_argument(
        "--observation-id",
        help="read this shared observation instead of the latest value",
    )
    observe.add_argument(
        "--include-robot",
        action="store_true",
        help="include the robot state captured with the observation",
    )
    observe.add_argument("--max-age-ns", type=_nonnegative_integer)
    observe.add_argument("--max-skew-ns", type=_nonnegative_integer)
    observe.set_defaults(command_handler=observe_command)

    media = commands.add_parser(
        "media",
        help="read media references or bounded frame data for an observation",
    )
    _common(media)
    media.add_argument("--observation-id", required=True)
    media.add_argument("--frame")
    media.add_argument("--include-data", action="store_true")
    media.set_defaults(command_handler=media_command)

    execute = commands.add_parser(
        "execute",
        help="submit one bounded JSON action chunk",
    )
    _common(execute)
    execute.add_argument("--request-id", required=True)
    execute.add_argument(
        "--action",
        required=True,
        metavar="PATH|-",
        help="one action JSON object, read from PATH or stdin when PATH is '-'",
    )
    execute.add_argument("--source", choices=("agent", "replay"), default="agent")
    execute.add_argument("--observation-id")
    execute.add_argument("--steps", type=_positive_integer, default=1)
    execute.add_argument("--control-hz", type=_positive_number)
    execute.add_argument("--job-timeout", type=_positive_number)
    execute.add_argument("--max-age-ns", type=_nonnegative_integer)
    execute.add_argument("--max-skew-ns", type=_nonnegative_integer)
    execute.add_argument(
        "--wait",
        action="store_true",
        help="wait for the bounded job result before returning",
    )
    execute.set_defaults(command_handler=execute_command)

    for name, handler, help_text in (
        ("inspect", inspect_command, "query one request by its original ID"),
        ("cancel", cancel_command, "cancel one request owned by this caller"),
        ("stop", stop_command, "request stop for one caller-owned request"),
    ):
        parser = commands.add_parser(name, help=help_text)
        _common(parser)
        parser.add_argument("--request-id", required=True)
        parser.set_defaults(command_handler=handler)

    recording_status = commands.add_parser("recording-status", help="query recorder state without changing it")
    _common(recording_status)
    recording_status.set_defaults(command_handler=recording_status_command)

    recording_start = commands.add_parser("recording-start", help="start service-owned observation recording")
    _common(recording_start)
    recording_start.set_defaults(command_handler=recording_start_command)

    recording_stop = commands.add_parser("recording-stop", help="stop service-owned observation recording")
    _common(recording_stop)
    recording_stop.add_argument("--recording-timeout", type=_nonnegative_number)
    recording_stop.set_defaults(command_handler=recording_stop_command)

    recording_get = commands.add_parser("recording-get", help="read one recorded observation by ID")
    _common(recording_get)
    recording_get.add_argument("--observation-id", required=True)
    recording_get.set_defaults(command_handler=recording_get_command)


def describe_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.describe(timeout_s=args.timeout),
    )


def observe_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, runtime: client.observe(
            runtime_id=runtime.runtime_id,
            observation_id=args.observation_id,
            include_robot=args.include_robot,
            max_age_ns=args.max_age_ns,
            max_skew_ns=args.max_skew_ns,
            timeout_s=args.timeout,
        ),
    )


def media_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.media(
            args.observation_id,
            frame=args.frame,
            runtime_id=args.runtime,
            include_data=args.include_data,
            timeout_s=args.timeout,
        ),
    )


def execute_command(args: argparse.Namespace, context: CommandContext) -> int:
    try:
        action = _read_action(args.action)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _emit_failure(
            {
                "status": "error",
                "error": {
                    "type": "invalid_action_json",
                    "message": str(error),
                },
            },
            EXIT_USAGE,
        )

    def submit(client: HostControlClient, _runtime: RuntimeSpec) -> dict[str, Any]:
        if isinstance(action, list):
            return client.execute(
                request_id=args.request_id,
                actions=action,
                source=args.source,
                observation_id=args.observation_id,
                steps=args.steps,
                control_hz=args.control_hz,
                wait=args.wait,
                job_timeout_s=args.job_timeout,
                max_age_ns=args.max_age_ns,
                max_skew_ns=args.max_skew_ns,
                timeout_s=args.timeout,
            )
        return client.execute(
            request_id=args.request_id,
            action=action,
            source=args.source,
            observation_id=args.observation_id,
            steps=args.steps,
            control_hz=args.control_hz,
            wait=args.wait,
            job_timeout_s=args.job_timeout,
            max_age_ns=args.max_age_ns,
            max_skew_ns=args.max_skew_ns,
            timeout_s=args.timeout,
        )

    return _call(args, context, submit, request_id=args.request_id)


def inspect_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.inspect(
            args.request_id,
            timeout_s=args.timeout,
        ),
        request_id=args.request_id,
    )


def cancel_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.cancel(
            args.request_id,
            timeout_s=args.timeout,
        ),
        request_id=args.request_id,
    )


def stop_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.stop(
            args.request_id,
            timeout_s=args.timeout,
        ),
        request_id=args.request_id,
    )


def recording_status_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.recording_status(timeout_s=args.timeout),
    )


def recording_start_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.recording_start(timeout_s=args.timeout),
    )


def recording_stop_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.recording_stop(
            recording_timeout_s=args.recording_timeout,
            timeout_s=args.timeout,
        ),
    )


def recording_get_command(args: argparse.Namespace, context: CommandContext) -> int:
    return _call(
        args,
        context,
        lambda client, _runtime: client.recording_get(
            args.observation_id,
            timeout_s=args.timeout,
        ),
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime", required=True, help="configured robot runtime ID")
    parser.add_argument("--caller-id", required=True, help="stable caller identity")
    parser.add_argument("--session-id", required=True, help="stable caller session")
    parser.add_argument("--token", help="optional Control token")
    parser.add_argument(
        "--timeout",
        type=_positive_number,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"HTTP transport timeout (default: {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON (the Control commands already use JSON by default)",
    )


def _call(
    args: argparse.Namespace,
    context: CommandContext,
    operation: Callable[[HostControlClient, RuntimeSpec], dict[str, Any]],
    *,
    request_id: str | None = None,
) -> int:
    try:
        runtime = _resolve_runtime(context, args.runtime)
        with context.executor(runtime.node) as executor:
            client = HostControlClient(
                executor,
                runtime.service_endpoint,
                caller_id=args.caller_id,
                session_id=args.session_id,
                token=args.token,
            )
            result = operation(client, runtime)
    except ControlClientError as error:
        return _emit_failure(_client_error_payload(error), _client_exit_code(error))
    except ControlCommandError as error:
        return _emit_failure(
            {
                "status": "error",
                "error": {
                    "type": "host_configuration",
                    "message": str(error),
                },
                **({"request_id": request_id} if request_id else {}),
            },
            EXIT_USAGE,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return _emit_failure(
            {
                "status": "error",
                "error": {
                    "type": "host_command",
                    "message": str(error),
                },
                **({"request_id": request_id} if request_id else {}),
            },
            EXIT_SERVICE,
        )
    return _emit_success(result)


def _resolve_runtime(context: CommandContext, runtime_id: str) -> RuntimeSpec:
    runtime = next(
        (item for item in context.deployment.runtimes if item.runtime_id == runtime_id),
        None,
    )
    if runtime is None:
        available = ", ".join(item.runtime_id for item in context.deployment.runtimes)
        raise ControlCommandError(f"unknown runtime {runtime_id!r}; available runtimes: {available or 'none'}")
    if runtime.target_kind != "robot":
        raise ControlCommandError(
            f"runtime {runtime.runtime_id!r} is a simulator; Control robot commands require a robot runtime"
        )
    try:
        state = StateStore(context.state_path).load()
    except (OSError, StateError, ValueError) as error:
        raise ControlCommandError(f"cannot read initialized deployment state: {error}") from error
    if state is None:
        raise ControlCommandError("deployment is not initialized; run `embodirun ... init`")
    try:
        digest = config_digest(context.config)
    except (OSError, ValueError) as error:
        raise ControlCommandError(f"cannot read deployment configuration: {error}") from error
    if state.config_digest != digest:
        raise ControlCommandError("configuration changed since init; run init again")
    if state.deploy_commit != context.deployment.deploy_commit:
        raise ControlCommandError("Deploy revision changed since init; run init again")
    environment = state.environments.get(runtime.environment_id)
    if environment is None or environment.status != "ready":
        raise ControlCommandError(f"environment {runtime.environment_id!r} is not ready; run init again")
    service = state.services.get(runtime.service_id)
    if service is None or service.status != "running":
        raise ControlCommandError(f"control service {runtime.service_id!r} is not running; run `embodirun ... up`")
    if service.node != runtime.node or service.endpoint != runtime.service_endpoint:
        raise ControlCommandError(f"runtime service {runtime.service_id!r} does not match the configured runtime")
    if runtime.node not in state.nodes:
        raise ControlCommandError(f"initialized state is missing node {runtime.node!r}")
    return runtime


def _read_action(path: str) -> Mapping[str, Any] | list[Mapping[str, Any]]:
    if not isinstance(path, str) or not path:
        raise ValueError("--action requires a JSON file path or '-' for stdin")
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(text)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        if not value:
            raise ValueError("action JSON array must not be empty")
        return [dict(item) for item in value]
    raise ValueError("action JSON must be an object or an array of objects")


def _client_error_payload(error: ControlClientError) -> dict[str, Any]:
    if error.unknown:
        return {
            "status": "unknown",
            **({"request_id": error.request_id} if error.request_id else {}),
            "error": {
                "type": "transport_unknown",
                "message": str(error),
            },
            "next": "inspect the original request_id before retrying",
        }
    payload = dict(error.payload) if isinstance(error.payload, Mapping) else {}
    payload.setdefault("status", "error")
    payload.setdefault(
        "error",
        {
            "type": "control_http_error",
            "message": str(error),
            **({"http_status": error.status} if error.status is not None else {}),
        },
    )
    if error.request_id is not None:
        payload.setdefault("request_id", error.request_id)
    return payload


def _client_exit_code(error: ControlClientError) -> int:
    if error.unknown or _payload_status(error.payload) in {"unknown", "uncertain"}:
        return EXIT_UNKNOWN
    return EXIT_SERVICE


def _payload_status(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        value = payload.get("status")
        return value if isinstance(value, str) else None
    return None


def _emit_success(payload: Mapping[str, Any]) -> int:
    _emit_json(payload)
    return EXIT_OK


def _emit_failure(payload: Mapping[str, Any], exit_code: int) -> int:
    _emit_json(payload)
    return exit_code


def _emit_json(payload: Mapping[str, Any]) -> None:
    json.dump(dict(payload), sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return result


def _positive_number(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def _nonnegative_number(value: str) -> float:
    """Parse a finite timeout where zero means no extra wait is requested."""

    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return result


__all__ = [
    "EXIT_OK",
    "EXIT_SERVICE",
    "EXIT_UNKNOWN",
    "EXIT_USAGE",
    "ControlCommandError",
    "register",
]
