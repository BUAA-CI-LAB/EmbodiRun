"""Submit one prompt task to a configured control-node service."""

from __future__ import annotations

import argparse
import math
import uuid
from typing import Any

from embodirun.bindings import binding_definition
from embodirun.deployment.config import config_digest
from embodirun.deployment.control import ControlClient
from embodirun.deployment.simulation import SimulationClient
from embodirun.deployment.state import StateStore
from embodirun.services.control.contracts import TaskRequest
from embodirun.services.simulation.contracts import EpisodeRequest

from ..context import CommandContext


class RunError(RuntimeError):
    """A configured runtime cannot accept or safely execute the requested task."""


DEFAULT_MAX_STEPS = 1
DEFAULT_CHUNK_STEPS = 10
DEFAULT_CONTROL_HZ = 5.0
DEFAULT_REQUEST_TIMEOUT_S = 60.0


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "run",
        help="run one task or episode on a configured runtime",
    )
    parser.add_argument("--runtime", required=True, help="configured runtime ID")
    parser.add_argument(
        "--prompt",
        help="instruction override; simulators may provide their own instruction",
    )
    parser.add_argument("--task", help="simulator task override")
    parser.add_argument("--seed", type=int, help="simulator episode seed")
    parser.add_argument(
        "--chunk-steps",
        type=_positive_integer,
        default=None,
        metavar="N",
        help=(
            "actions to execute from each inference chunk "
            f"(default: up to {DEFAULT_CHUNK_STEPS}, capped by the binding)"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=DEFAULT_MAX_STEPS,
        metavar="N",
        help=(f"maximum inference/action chunks to execute (default: {DEFAULT_MAX_STEPS})"),
    )
    parser.add_argument(
        "--control-hz",
        type=_positive_number,
        default=DEFAULT_CONTROL_HZ,
        metavar="HZ",
        help=f"action playback rate (default: {DEFAULT_CONTROL_HZ:g})",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_number,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        metavar="SECONDS",
        help=(f"timeout for one inference request (default: {DEFAULT_REQUEST_TIMEOUT_S:g})"),
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, context: CommandContext) -> int:
    if args.prompt is not None and not args.prompt.strip():
        raise RunError("--prompt must not be empty")
    runtime = next(
        (item for item in context.deployment.runtimes if item.runtime_id == args.runtime),
        None,
    )
    if runtime is None:
        available = ", ".join(item.runtime_id for item in context.deployment.runtimes)
        raise RunError(f"unknown runtime {args.runtime!r}; available runtimes: {available or 'none'}")
    if runtime.model is None:
        raise RunError(
            f"runtime {runtime.runtime_id!r} is device-only and has no inference "
            "model; use the control service observe/describe operations"
        )
    if runtime.target_kind == "robot" and args.prompt is None:
        raise RunError("--prompt is required for robot runtimes")
    if runtime.target_kind == "robot" and (args.task is not None or args.seed is not None):
        raise RunError("--task and --seed are only valid for simulator runtimes")
    definition = binding_definition(runtime.binding)
    chunk_steps = (
        min(DEFAULT_CHUNK_STEPS, definition.maximum_chunk_steps) if args.chunk_steps is None else args.chunk_steps
    )
    if chunk_steps > definition.maximum_chunk_steps:
        raise RunError(
            f"--chunk-steps {chunk_steps} exceeds binding {runtime.binding!r} maximum {definition.maximum_chunk_steps}"
        )

    progress = context.progress
    progress.begin("run", context.deployment.name)
    progress.add_node(runtime.node, total=2)
    try:
        progress.update(runtime.node, "Validating control service", detail=runtime.runtime_id)
        state = StateStore(context.state_path).load()
        if state is None:
            raise RunError("deployment is not initialized; run `embodirun ... init`")
        if state.config_digest != config_digest(context.config):
            raise RunError("configuration changed since init; run init again")
        if state.deploy_commit != context.deployment.deploy_commit:
            raise RunError("Deploy revision changed since init; run init again")
        environment = state.environments.get(runtime.environment_id)
        if environment is None or environment.status != "ready":
            raise RunError(f"environment {runtime.environment_id!r} is not ready; run init again")
        service = state.services.get(runtime.service_id)
        if service is None or service.status != "running":
            raise RunError(
                f"{runtime.target_kind} service {runtime.service_id!r} is not running; run `embodirun ... up`"
            )
        if service.node != runtime.node or service.endpoint != runtime.service_endpoint:
            raise RunError(
                f"runtime service {runtime.service_id!r} does not match the configured runtime; restart the deployment"
            )
        if runtime.node not in state.nodes:
            raise RunError(f"initialized state is missing node {runtime.node!r}")
        request_id = f"task-{uuid.uuid4().hex}"
        progress.advance(runtime.node)

        playback_s = (
            args.max_steps * max(0, chunk_steps - 1) / args.control_hz if runtime.target_kind == "robot" else 0.0
        )
        task_timeout_s = max(60.0, args.max_steps * args.request_timeout + playback_s + 30.0)
        progress.update(
            runtime.node,
            ("Executing control task" if runtime.target_kind == "robot" else "Executing simulation episode"),
            detail=f"{args.max_steps} chunk(s), {chunk_steps} action(s) each",
        )
        with context.executor(runtime.node) as executor:
            if runtime.target_kind == "robot":
                request = TaskRequest(
                    request_id=request_id,
                    runtime_id=runtime.runtime_id,
                    prompt=args.prompt,
                    chunk_steps=chunk_steps,
                    max_steps=args.max_steps,
                    control_hz=args.control_hz,
                    inference_timeout_s=args.request_timeout,
                )
                result = ControlClient(executor, runtime.service_endpoint).run(request, timeout_s=task_timeout_s)
                detail = f"Completed {result.completed_steps} chunk(s), {chunk_steps} action(s) each"
            else:
                request = EpisodeRequest(
                    request_id=request_id,
                    runtime_id=runtime.runtime_id,
                    prompt=args.prompt,
                    task=args.task,
                    seed=args.seed,
                    chunk_steps=chunk_steps,
                    max_policy_steps=args.max_steps,
                    inference_timeout_s=args.request_timeout,
                )
                result = SimulationClient(executor, runtime.service_endpoint).run(request, timeout_s=task_timeout_s)
                detail = (
                    f"Episode completed: {result.policy_steps} policy step(s), "
                    f"{result.environment_steps} environment step(s), "
                    f"reward={result.total_reward:g}"
                )
        progress.advance(runtime.node)
        progress.succeed(runtime.node, detail=detail)
    except BaseException as error:
        progress.fail(runtime.node, error)
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    return 0


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_number(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


__all__ = ["RunError", "register", "run"]
