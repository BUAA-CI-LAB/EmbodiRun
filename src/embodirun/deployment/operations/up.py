"""Start configured services and wait for their health checks.

The operation owns managed service processes and persists each observed state.
Readiness timeouts and node failures can leave partial state that callers must
inspect before retrying; external endpoints outside the deployment plan are not
started or modified.
"""

from __future__ import annotations

import posixpath
import time
from dataclasses import dataclass, replace

from embodirun.application.contracts import ControlServiceConfig
from embodirun.application.simulation.contracts import SimulationServiceConfig
from embodirun.deployment.config import config_digest
from embodirun.deployment.executor import Command, Executor
from embodirun.deployment.plan import ServiceSpec
from embodirun.deployment.source import active_deploy_project
from embodirun.deployment.state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)
from embodirun.deployment.supervisor import ServiceSupervisor

from ._context import DeploymentContext
from ._parallel import run_on_nodes


class UpError(RuntimeError):
    """Configured services could not be started safely."""


@dataclass(frozen=True, slots=True)
class NodeUpResult:
    """Service states produced by one node worker."""

    services: dict[str, ServiceState]
    error: Exception | None = None


DEFAULT_WAIT_TIMEOUT_S = 600.0

_HEALTH_REQUEST_TIMEOUT_S = 2.0

_READY_POLL_INTERVAL_S = 1.0


def start_services(context: DeploymentContext, wait_timeout_s: float) -> int:
    """Start managed services and persist running/failed states.

    ``wait_timeout_s`` is a finite positive number of seconds for each service's
    health wait. A failed node may leave a partial state file with successful and
    failed service records; the operation does not start external endpoints or
    change services outside the configured managed plan.
    """
    progress = context.progress
    progress.begin("up", context.deployment.name)
    node_services = services_by_node(context)
    for node_id in sorted(context.config.nodes):
        progress.add_node(
            node_id,
            total=max(1, 3 * len(node_services[node_id])),
        )
    try:
        state = _up(
            context,
            services_by_node=node_services,
            wait_timeout_s=wait_timeout_s,
        )
    except BaseException:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    running = sum(service.status == "running" for service in state.services.values())
    progress.message(f"{running}/{len(state.services)} services running")
    for service in sorted(state.services.values(), key=lambda item: item.service_id):
        if service.status == "running":
            progress.message(f"{service.service_id}: {service.endpoint} on {service.node} (PID {service.pid})")
    return 0


def _up(
    context: DeploymentContext,
    *,
    services_by_node: dict[str, tuple[ServiceSpec, ...]],
    wait_timeout_s: float,
) -> DeploymentState:
    store = StateStore(context.state_path)
    state = store.load()
    if state is None:
        raise UpError("deployment is not initialized; run `embodirun ... init`")
    if state.config_digest != config_digest(context.config):
        raise UpError("configuration changed since init; run init again")
    missing_nodes = set(context.config.nodes) - state.nodes.keys()
    if missing_nodes:
        raise UpError("initialized state is missing nodes: " + ", ".join(sorted(missing_nodes)))

    for service in context.deployment.services:
        environment = state.environments.get(service.environment_id)
        if environment is None or environment.status != "ready":
            raise UpError(f"environment {service.environment_id!r} is not ready; run init again")

    def start_node(node_id: str) -> NodeUpResult:
        try:
            return _up_node(
                context,
                state,
                node_id,
                services_by_node[node_id],
                wait_timeout_s=wait_timeout_s,
            )
        except Exception as error:
            context.progress.fail(node_id, error)
            raise

    results = run_on_nodes(context.config.nodes, start_node)
    services = dict(state.services)
    failures: dict[str, Exception] = dict(results.errors)
    for node_id, result in results.values.items():
        services.update(result.services)
        if result.error is not None:
            failures[node_id] = result.error
    state = replace(state, services=services)
    store.save(state)
    if failures:
        raise UpError(_node_error_summary("service startup", failures))
    return state


def _up_node(
    context: DeploymentContext,
    state: DeploymentState,
    node_id: str,
    services: tuple[ServiceSpec, ...],
    *,
    wait_timeout_s: float,
) -> NodeUpResult:
    progress = context.progress
    if not services:
        progress.update(node_id, "No services configured")
        progress.advance(node_id)
        progress.succeed(node_id, detail="No services")
        return NodeUpResult({})

    service_states: dict[str, ServiceState] = {}
    with context.executor(node_id) as executor:
        node = state.nodes[node_id]
        deploy_project = active_deploy_project(node.root)
        for service in services:
            process = None
            try:
                environment = state.environments[service.environment_id]
                materialized = _materialize_service(service, node, environment)
                executable = materialized.command.argv[0]
                progress.update(
                    node_id,
                    "Checking service executable",
                    detail=service.service_id,
                )
                available = executor.run(
                    Command(("test", "-x", executable)),
                    check=False,
                )
                if available.exit_code != 0:
                    raise UpError(f"service executable is missing or not executable: {executable}")
                progress.advance(node_id)

                _write_generated_configs(
                    materialized,
                    executor,
                    node=node,
                )
                supervisor = ServiceSupervisor(
                    executor,
                    python=node.python,
                    agent_path=posixpath.join(
                        deploy_project,
                        "src/embodirun/deployment/supervisor.py",
                    ),
                    run_root=posixpath.join(node.root, "run"),
                    log_root=posixpath.join(node.root, "logs"),
                )
                progress.update(
                    node_id,
                    "Starting service",
                    detail=service.service_id,
                )
                process = supervisor.start(materialized)
                progress.advance(node_id)

                progress.update(
                    node_id,
                    "Waiting for service health",
                    detail=service.service_id,
                )
                _wait_until_ready(
                    materialized,
                    supervisor,
                    executor,
                    pid=process.pid,
                    log=process.log,
                    timeout_s=wait_timeout_s,
                )
                progress.advance(node_id)
                service_states[service.service_id] = ServiceState(
                    service_id=service.service_id,
                    node=service.node,
                    status="running",
                    pid=process.pid,
                    endpoint=service.endpoint,
                )
            except (OSError, RuntimeError, ValueError) as error:
                service_states[service.service_id] = ServiceState(
                    service_id=service.service_id,
                    node=service.node,
                    status="failed",
                    pid=process.pid if process is not None else None,
                    endpoint=service.endpoint,
                )
                progress.fail(node_id, error)
                return NodeUpResult(service_states, error)

    progress.succeed(node_id, detail=f"{len(services)} service(s) ready")
    return NodeUpResult(service_states)


def services_by_node(
    context: DeploymentContext,
) -> dict[str, tuple[ServiceSpec, ...]]:
    return {
        node_id: tuple(service for service in context.deployment.services if service.node == node_id)
        for node_id in context.config.nodes
    }


def _wait_until_ready(
    service: ServiceSpec,
    supervisor: ServiceSupervisor,
    executor: Executor,
    *,
    pid: int | None,
    log: str | None,
    timeout_s: float,
) -> None:
    if pid is None:
        raise UpError(f"service {service.service_id!r} started without a PID")
    deadline = time.monotonic() + timeout_s
    while True:
        process = supervisor.status(service.service_id)
        if process.state != "running" or process.pid != pid:
            raise UpError(f"service {service.service_id!r} exited before becoming ready; log: {log}")
        remaining_s = deadline - time.monotonic()
        request_timeout_s = min(
            _HEALTH_REQUEST_TIMEOUT_S,
            max(0.01, remaining_s),
        )
        try:
            if service.health_endpoint is None:
                return
            health = executor.get_json(
                service.health_endpoint,
                timeout_s=request_timeout_s,
            )
            ready = health.status == 200 and isinstance(health.payload, dict) and health.payload.get("status") == "ok"
        except (OSError, RuntimeError, ValueError):
            ready = False
        if ready:
            return
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise UpError(
                f"service {service.service_id!r} did not become healthy at "
                f"{service.health_endpoint} within {timeout_s:g} seconds; "
                f"log: {log}"
            )
        time.sleep(min(_READY_POLL_INTERVAL_S, remaining_s))


def _materialize_service(
    service: ServiceSpec,
    node: NodeState,
    environment: EnvironmentState,
) -> ServiceSpec:
    deploy_project = active_deploy_project(node.root)
    argv = list(service.command.argv)
    argv[0] = posixpath.join(environment.path, "bin", posixpath.basename(argv[0]))
    if service.adapter_config_json is not None:
        argv.extend(
            (
                "--adapter-config",
                posixpath.join(
                    node.root,
                    "generated",
                    f"{service.service_id}.adapter.json",
                ),
            )
        )
    if service.control_config_json is not None:
        argv.extend(
            (
                "--config",
                posixpath.join(
                    node.root,
                    "generated",
                    f"{service.service_id}.control.json",
                ),
            )
        )
    if service.simulation_config_json is not None:
        argv.extend(
            (
                "--config",
                posixpath.join(
                    node.root,
                    "generated",
                    f"{service.service_id}.simulation.json",
                ),
            )
        )
    try:
        adapter_index = argv.index("--adapter-config") + 1
    except ValueError:
        pass
    else:
        argv[adapter_index] = _configured_path(argv[adapter_index], deploy_project)
    # The provider descriptor chooses the owning source project.  A managed
    # model may therefore run from Deploy (for a local optional integration)
    # or from the separate Inference checkout (for the EmbodiInfer service).
    project = deploy_project if environment.project == "deploy" else node.inference_project
    if service.wireless_config_json is not None:
        wireless_path = posixpath.join(node.root, "generated", f"{service.service_id}.wireless.json")
        if service.kind == "model":
            argv[argv.index("--comm-config") + 1] = wireless_path
        for field_name, config_type in (
            ("control_config_json", ControlServiceConfig),
            ("simulation_config_json", SimulationServiceConfig),
        ):
            payload = getattr(service, field_name)
            if payload is not None:
                config = config_type.from_json(payload)
                config = replace(
                    config,
                    inference_options={
                        **config.inference_options,
                        "comm_config": wireless_path,
                    },
                )
                service = replace(service, **{field_name: config.to_json()})
    if "--comm-config" in argv:
        config_index = argv.index("--comm-config") + 1
        argv[config_index] = _configured_path(argv[config_index], project)
    if "--pipeline-config-path" in argv:
        config_index = argv.index("--pipeline-config-path") + 1
        argv[config_index] = _configured_path(argv[config_index], project)
    environment_variables = dict(service.command.environment)
    environment_variables["VIRTUAL_ENV"] = environment.path
    if service.kind in {"control", "simulation"}:
        environment_variables["PYTHONPATH"] = posixpath.join(deploy_project, "src")
    return replace(
        service,
        command=Command(
            tuple(argv),
            cwd=project,
            environment=environment_variables,
        ),
    )


def _write_generated_configs(
    service: ServiceSpec,
    executor: Executor,
    *,
    node: NodeState,
) -> None:
    if service.wireless_config_json is not None:
        executor.write_text(
            posixpath.join(node.root, "generated", f"{service.service_id}.wireless.json"),
            f"{service.wireless_config_json}\n",
            mode=0o600,
        )
    if service.adapter_config_json is not None:
        adapter_index = service.command.argv.index("--adapter-config") + 1
        executor.write_text(
            service.command.argv[adapter_index],
            f"{service.adapter_config_json}\n",
            mode=0o600,
        )
    if service.control_config_json is not None:
        config_index = service.command.argv.index("--config") + 1
        executor.write_text(
            service.command.argv[config_index],
            f"{service.control_config_json}\n",
            mode=0o600,
        )
    if service.simulation_config_json is not None:
        config_index = service.command.argv.index("--config") + 1
        executor.write_text(
            service.command.argv[config_index],
            f"{service.simulation_config_json}\n",
            mode=0o600,
        )


def _configured_path(path: str, project_dir: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(project_dir, path))


def _node_error_summary(command: str, errors: dict[str, Exception]) -> str:
    details = "; ".join(f"{node_id}: {error}" for node_id, error in sorted(errors.items()))
    return f"{command} failed on {len(errors)} node(s): {details}"


__all__ = ["start_services", "services_by_node"]
