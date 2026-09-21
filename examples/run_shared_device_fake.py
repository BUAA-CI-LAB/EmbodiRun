#!/usr/bin/env python3
"""Run a complete local fake-device Host walkthrough.

This is an executable software demo, not a deployment initializer.  It loads
the public fake YAML, builds the same Host plan as ``embodirun``, starts a
real :class:`ControlService` on a loopback socket, and writes an ephemeral
``StateStore`` fixture so the normal Host CLI can resolve that service.  The
fixture is deliberately temporary and never opens a serial, CAN, USB, or
network robot resource.

The production ``ControlHttpServer`` now owns the application API boundary;
this helper only supplies the temporary Host state and recorder directories.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from base64 import b64decode
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from embodirun.services.control.contracts import ControlServiceConfig
from embodirun.services.control.devices import DeviceManager
from embodirun.services.control.observations import ObservationRecorder
from embodirun.services.control.server import ControlHttpServer, ControlService
from embodirun.services.host.config import config_digest, load_config
from embodirun.services.host.plan import DeploymentPlan, ServiceSpec, build_plan
from embodirun.services.host.state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples" / "shared-device-fake.yaml"
CALLER_ID = "local-fake-walkthrough"
SESSION_ID = "local-fake-session"


class DemoError(RuntimeError):
    """The public fake walkthrough cannot safely run its software fixture."""


@contextmanager
def local_fake_service(
    config_path: Path,
    state_dir: Path,
) -> Iterator[tuple[DeploymentPlan, ServiceSpec, ControlService, Path]]:
    """Build, serve, and tear down one validated fake-only service."""

    config = load_config(config_path)
    _require_fake_config(config)
    deployment = build_plan(config)
    runtime = deployment.runtimes[0]
    service_spec = next(service for service in deployment.services if service.service_id == runtime.service_id)
    if service_spec.control_config_json is None:
        raise DemoError("fake walkthrough plan did not produce a control config")
    control_config = ControlServiceConfig.from_json(service_spec.control_config_json)
    if control_config.inference_enabled:
        raise DemoError("fake walkthrough requires inference.enabled=false")

    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{deployment.name}.json"
    _write_state(config, deployment, service_spec, state_path)
    manager = DeviceManager(
        control_config.node_id,
        owner_id=f"local-fake:{os.getpid()}",
        lock_dir=state_dir / "locks",
        state_path=state_dir / "device-state.json",
    )
    # The recorder must subscribe to the service-owned store.  Construct it
    # after the service, then attach it through the same checked API.
    service = ControlService(control_config, device_manager=manager)
    recorder = ObservationRecorder(
        service.observation_store,
        state_dir / "recordings",
        "fake-walkthrough",
        require_state=True,
    )
    service.attach_recorder(recorder)
    # This is the production socket/server and its now-integrated application
    # API.  The only special setup is the temporary Host state fixture above.
    server = ControlHttpServer(service, state_dir=state_dir)
    thread = threading.Thread(
        target=server.serve_forever,
        name="local-fake-control-http",
        daemon=True,
    )
    thread.start()
    try:
        yield deployment, service_spec, service, state_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        try:
            service.close()
        except Exception as error:
            raise DemoError(f"fake service cleanup failed: {error}") from error


def _require_fake_config(config: Any) -> None:
    if config.models:
        raise DemoError("the local walkthrough refuses model runtimes")
    if len(config.nodes) != 1 or any(node.connection.kind != "local" for node in config.nodes.values()):
        raise DemoError("the local walkthrough requires one local-only node")
    if len(config.runtimes) != 1:
        raise DemoError("the local walkthrough requires exactly one runtime")
    runtime = next(iter(config.runtimes.values()))
    if runtime.robot is None or runtime.model is not None or runtime.binding is not None:
        raise DemoError("the local walkthrough requires one no-model robot runtime")
    robot = config.robots[runtime.robot]
    if robot.kind != "simulated.joints":
        raise DemoError("the local walkthrough accepts only simulated.joints")
    if any(config.sensors[sensor_id].kind != "fake" for sensor_id in runtime.inputs.values()):
        raise DemoError("the local walkthrough accepts only fake camera inputs")
    if runtime.server.bind not in {"127.0.0.1", "localhost", "::1"}:
        raise DemoError("the local walkthrough requires a loopback control bind")


def _write_state(config: Any, deployment: DeploymentPlan, service: ServiceSpec, path: Path) -> None:
    nodes = {
        node_id: NodeState(
            node_id=node_id,
            home=str(path.parent),
            root=str(path.parent),
            deploy_project=str(ROOT),
            inference_project=str(ROOT),
            platform=sys.platform,
            machine=platform.machine() or "local-fake",
            python=sys.executable,
            python_version=platform.python_version(),
        )
        for node_id in config.nodes
    }
    environments = {
        profile.environment_id: EnvironmentState(
            environment_id=profile.environment_id,
            node=profile.node,
            project=profile.project,
            group=profile.group,
            path=profile.path,
            status="ready",
        )
        for profile in deployment.environments
    }
    state = DeploymentState(
        name=deployment.name,
        config_digest=config_digest(config),
        deploy_commit=deployment.deploy_commit,
        inference_commit=None,
        nodes=nodes,
        environments=environments,
        services={
            service.service_id: ServiceState(
                service_id=service.service_id,
                node=service.node,
                status="running",
                pid=os.getpid(),
                endpoint=service.endpoint,
            )
        },
    )
    StateStore(path).save(state)


def _run_cli(config_path: Path, state_dir: Path, args: list[str]) -> dict[str, Any]:
    """Run the repository's actual ``main(argv)`` in a clean subprocess."""

    command = [
        sys.executable,
        "-c",
        "from embodirun.services.host.cli.cli import main; raise SystemExit(main())",
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
        *args,
    ]
    env = os.environ.copy()
    source_root = str(ROOT / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    # Include the temporary fixture arguments in the echoed command.  The
    # output is therefore copyable while this process is running instead of
    # depending on hidden module globals or an implicit state directory.
    print(
        "$ "
        + shlex.join(
            [
                "embodirun",
                "--config",
                str(config_path),
                "--state-dir",
                str(state_dir),
                *args,
            ]
        ),
        file=sys.stderr,
    )
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise DemoError(f"CLI failed with exit {result.returncode}: {result.stderr.strip() or result.stdout.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise DemoError(f"CLI did not emit one JSON result: {result.stdout!r}") from error
    if not isinstance(payload, dict):
        raise DemoError("CLI result must be one JSON object")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _action_file(state_dir: Path, name: str) -> Path:
    action = {
        "timestamp_s": 0,
        "values": {
            "type": "joint_position",
            "joint_positions_deg": [6, 0, 0, 0, 0],
            "gripper_position": 25,
        },
        "metadata": {"action_space": "simulated.so101.position.v1"},
    }
    path = state_dir / name
    path.write_text(json.dumps(action), encoding="utf-8")
    return path


def run_smoke(config_path: Path, state_dir: Path) -> None:
    with local_fake_service(config_path, state_dir) as (
        deployment,
        _service_spec,
        _service,
        _state_path,
    ):
        runtime_id = deployment.runtimes[0].runtime_id
        common = [
            "--runtime",
            runtime_id,
            "--caller-id",
            CALLER_ID,
            "--session-id",
            SESSION_ID,
            "--json",
        ]
        _run_cli(config_path, state_dir, ["describe", *common])
        observed = _run_cli(config_path, state_dir, ["observe", *common, "--include-robot"])
        # Older service observation payloads call this field ``snapshot_id``;
        # both names identify the same immutable shared observation.
        observation_id = observed.get("observation_id", observed.get("snapshot_id"))
        if not isinstance(observation_id, str) or not observation_id:
            raise DemoError("observe did not return an observation_id")
        media = _run_cli(
            config_path,
            state_dir,
            [
                "media",
                *common,
                "--observation-id",
                observation_id,
                "--include-data",
            ],
        )
        encoded = media.get("media", [{}])[0].get("data_base64")
        if not isinstance(encoded, str) or b64decode(encoded)[:8] != b"\x89PNG\r\n\x1a\n":
            raise DemoError("media did not return the fake PNG bytes")

        recorder_status = _run_cli(config_path, state_dir, ["recording-start", *common])
        if recorder_status.get("state") != "running":
            raise DemoError("recording did not start")
        _run_cli(config_path, state_dir, ["recording-status", *common])

        # Keep recording live through both direct actions.  Each action uses
        # a freshly captured observation; reusing an old ID would make a
        # successful software demo look like a stale-input path.
        recorded = _run_cli(
            config_path,
            state_dir,
            ["observe", *common, "--include-robot"],
        )
        recorded_observation_id = recorded.get("observation_id", recorded.get("snapshot_id"))
        if not isinstance(recorded_observation_id, str) or not recorded_observation_id:
            raise DemoError("recording observe did not return an observation ID")
        action_path = _action_file(state_dir, "short-action.json")
        short_observation = _run_cli(
            config_path,
            state_dir,
            ["observe", *common, "--include-robot"],
        )
        short_observation_id = short_observation.get("observation_id", short_observation.get("snapshot_id"))
        if not isinstance(short_observation_id, str) or not short_observation_id:
            raise DemoError("short action observe did not return an observation ID")
        completed = _run_cli(
            config_path,
            state_dir,
            [
                "execute",
                *common,
                "--request-id",
                "fake-short-action",
                "--action",
                str(action_path),
                "--observation-id",
                short_observation_id,
                "--max-age-ns",
                "1000000000",
                "--wait",
            ],
        )
        if completed.get("status") != "completed":
            raise DemoError("short fake action did not complete")
        _run_cli(
            config_path,
            state_dir,
            ["inspect", *common, "--request-id", "fake-short-action"],
        )

        long_action = _action_file(state_dir, "long-action.json")
        long_observation = _run_cli(
            config_path,
            state_dir,
            ["observe", *common, "--include-robot"],
        )
        long_observation_id = long_observation.get("observation_id", long_observation.get("snapshot_id"))
        if not isinstance(long_observation_id, str) or not long_observation_id:
            raise DemoError("long action observe did not return an observation ID")
        accepted = _run_cli(
            config_path,
            state_dir,
            [
                "execute",
                *common,
                "--request-id",
                "fake-cancelled-action",
                "--action",
                str(long_action),
                "--observation-id",
                long_observation_id,
                "--max-age-ns",
                "1000000000",
                "--steps",
                "50",
                "--control-hz",
                "2",
            ],
        )
        if accepted.get("status") not in {"accepted", "running"}:
            raise DemoError(f"long fake action did not start: {accepted}")
        deadline = time.monotonic() + 2.0
        running = accepted
        while running.get("status") != "running" and time.monotonic() < deadline:
            time.sleep(0.02)
            running = _run_cli(
                config_path,
                state_dir,
                ["inspect", *common, "--request-id", "fake-cancelled-action"],
            )
        if running.get("status") != "running":
            raise DemoError(f"long fake action was not live before cancel: {running}")
        _run_cli(
            config_path,
            state_dir,
            ["cancel", *common, "--request-id", "fake-cancelled-action"],
        )
        final = running
        deadline = time.monotonic() + 3.0
        while final.get("status") not in {"cancelled", "completed", "failed"}:
            if time.monotonic() >= deadline:
                raise DemoError(f"cancelled action did not finish: {final}")
            time.sleep(0.02)
            final = _run_cli(
                config_path,
                state_dir,
                ["inspect", *common, "--request-id", "fake-cancelled-action"],
            )
        if final.get("status") != "cancelled":
            raise DemoError(f"cancelled action unexpectedly finished: {final}")
        # The cancelled direct segment intentionally has no completed result
        # object.  Its monotonic lifecycle timestamps still prove that the
        # request was stopped well before the 50 x 0.5 second schedule could
        # finish; this is the public API's bounded cancellation evidence.
        started_at = final.get("started_at_s")
        finished_at = final.get("finished_at_s")
        if (
            not isinstance(started_at, (int, float))
            or not isinstance(finished_at, (int, float))
            or finished_at - started_at >= 20.0
        ):
            raise DemoError(f"cancelled action ran too long: {final}")

        stopped = _run_cli(
            config_path,
            state_dir,
            ["recording-stop", *common, "--recording-timeout", "2"],
        )
        if stopped.get("state") != "stopped":
            raise DemoError("recording did not stop")
        recording_status = _run_cli(config_path, state_dir, ["recording-status", *common])
        if recording_status.get("state") != "stopped":
            raise DemoError("recording status did not become stopped")
        if recording_status.get("action_count", 0) < 1:
            raise DemoError("recording did not capture direct action events")
        _run_cli(
            config_path,
            state_dir,
            ["recording-get", *common, "--observation-id", recorded_observation_id],
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="ephemeral fixture directory (default: a temporary directory)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start the real local fake service and print commands without running smoke",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.config.exists():
        raise SystemExit(f"fake config does not exist: {args.config}")
    if args.state_dir is not None:
        if args.serve:
            with local_fake_service(args.config, args.state_dir) as (
                deployment,
                service,
                _control,
                _state,
            ):
                print_commands(args.config, args.state_dir, deployment, service)
                _wait_for_interrupt()
        else:
            run_smoke(args.config, args.state_dir)
        return 0
    with tempfile.TemporaryDirectory(prefix="embodirun-shared-device-fake-") as directory:
        state_dir = Path(directory)
        if args.serve:
            with local_fake_service(args.config, state_dir) as (deployment, service, _control, _state):
                print_commands(args.config, state_dir, deployment, service)
                _wait_for_interrupt()
        else:
            run_smoke(args.config, state_dir)
    return 0


def print_commands(
    config_path: Path,
    state_dir: Path,
    deployment: DeploymentPlan,
    service: ServiceSpec,
) -> None:
    """Print copyable commands for the same temporary Host fixture."""

    runtime = deployment.runtimes[0].runtime_id
    prefix = f"embodirun --config {shlex.quote(str(config_path))} --state-dir {shlex.quote(str(state_dir))}"
    print(f"service endpoint: {service.endpoint}", file=sys.stderr)
    print(
        f"{prefix} describe --runtime {runtime} --caller-id {CALLER_ID} --session-id {SESSION_ID} --json",
        file=sys.stderr,
    )
    print(
        f"{prefix} observe --runtime {runtime} --caller-id {CALLER_ID} --session-id {SESSION_ID} --json",
        file=sys.stderr,
    )


def _wait_for_interrupt() -> None:
    print("fake service is running; press Ctrl-C to stop", file=sys.stderr)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoError as error:
        print(f"run_shared_device_fake.py: error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
