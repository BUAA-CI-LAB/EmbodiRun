"""Initialize managed nodes, source checkouts, and locked environments.

The operation persists ready resources in the deployment state file. A node or
resource failure is retained as a failed/partial initialization result; this
operation only manages configured Deploy/Inference checkouts and does not call
external model endpoints.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from embodirun.deployment.config import config_digest
from embodirun.deployment.config.robot import robot_calibration_ids, robot_ports
from embodirun.deployment.environment import EnvironmentProfile, UvEnvironmentManager
from embodirun.deployment.executor import Command, Executor
from embodirun.deployment.plan import DeploymentPlan
from embodirun.deployment.probe import probe_node
from embodirun.deployment.source import (
    ProjectManager,
    active_deploy_project,
    managed_root,
)
from embodirun.deployment.state import (
    DeploymentState,
    EnvironmentState,
    NodeState,
    ServiceState,
    StateStore,
)
from embodirun.model_services.providers import provider

from ._context import DeploymentContext
from ._parallel import run_on_nodes

DEPLOY_REPOSITORY = "https://github.com/BUAA-CI-LAB/EmbodiRun.git"

INFERENCE_REPOSITORY = "https://github.com/BUAA-CI-LAB/EmbodiInfer.git"

DEFAULT_MANAGED_ROOT = ".local/share/embodirun"


class InitError(RuntimeError):
    """A deployment node could not be initialized safely."""


@dataclass(frozen=True, slots=True)
class NodeInitialization:
    """Initialized state produced independently by one node worker."""

    node: NodeState
    environments: tuple[EnvironmentState, ...]
    inference_commit: str | None


def initialize(context: DeploymentContext, managed_root_base: str) -> int:
    """Initialize configured nodes and persist their ready environments.

    ``managed_root_base`` is a POSIX path relative to each node's home unless the
    existing source/project helpers resolve it otherwise. The operation probes and
    prepares only configured managed resources, writes the resulting state file,
    and raises after preserving successful node results when another node fails.
    """
    progress = context.progress
    progress.begin("init", context.deployment.name)
    for node_id in sorted(context.config.nodes):
        progress.add_node(
            node_id,
            total=3 + len(_profiles_on(context.deployment, node_id)),
        )
    try:
        state = _initialize(context, managed_root_base=managed_root_base)
    except BaseException:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    progress.message(f"State saved to {context.state_path} ({len(state.environments)} environments ready)")
    return 0


def _initialize(
    context: DeploymentContext,
    *,
    managed_root_base: str,
) -> DeploymentState:
    store = StateStore(context.state_path)
    previous = store.load()
    digest = config_digest(context.config)
    if previous is not None and previous.config_digest != digest:
        running = [service.service_id for service in previous.services.values() if service.status == "running"]
        if running:
            raise InitError(
                "configuration changed while services are recorded as running: " + ", ".join(sorted(running))
            )

    def initialize(node_id: str) -> NodeInitialization:
        try:
            return _initialize_node(
                context,
                node_id,
                managed_root_base=managed_root_base,
            )
        except Exception as error:
            context.progress.fail(node_id, error)
            raise

    results = run_on_nodes(context.config.nodes, initialize)
    inference_commits = {
        result.inference_commit for result in results.values.values() if result.inference_commit is not None
    }
    if len(inference_commits) > 1:
        raise InitError(
            "nodes resolved different Inference revisions from Deploy; "
            "use an immutable deploy-commit and run init again"
        )
    nodes = {node_id: result.node for node_id, result in results.values.items()}
    environments = {
        environment.environment_id: environment
        for result in results.values.values()
        for environment in result.environments
    }
    state = DeploymentState(
        name=context.deployment.name,
        config_digest=digest,
        deploy_commit=context.deployment.deploy_commit,
        inference_commit=next(iter(inference_commits), None),
        nodes=nodes,
        environments=environments,
        services=_initial_service_state(context.deployment, previous, digest),
    )
    store.save(state)
    if results.errors:
        raise InitError(_node_error_summary("initialization", results.errors))
    return state


def _initialize_node(
    context: DeploymentContext,
    node_id: str,
    *,
    managed_root_base: str,
) -> NodeInitialization:
    progress = context.progress
    with context.executor(node_id) as executor:
        progress.update(node_id, "Probing node and required tools")
        probe = probe_node(executor, require_init_tools=True)
        assert probe.python is not None
        assert probe.python_version is not None
        assert probe.git is not None
        assert probe.uv is not None
        progress.advance(node_id)

        root = managed_root(probe.home, managed_root_base, context.deployment.name)
        node_state = NodeState(
            node_id=node_id,
            home=probe.home,
            root=root,
            deploy_project=posixpath.join(root, "sources", "deploy"),
            inference_project=posixpath.join(root, "sources", "inference"),
            platform=probe.platform,
            machine=probe.machine,
            python=probe.python,
            python_version=probe.python_version,
        )
        progress.update(node_id, "Preparing locked sources")
        inference_commit = _prepare_projects(context, executor, node_state, git=probe.git)
        executor.replace_symlink(
            active_deploy_project(node_state.root),
            node_state.deploy_project,
        )
        progress.advance(node_id)

        environments: list[EnvironmentState] = []
        for profile in _profiles_on(context.deployment, node_id):
            progress.update(
                node_id,
                "Synchronizing environment",
                detail=f"{profile.project}/{profile.group}",
            )
            project_dir = _project_dir(node_state, profile.project)
            UvEnvironmentManager(executor, uv_executable=probe.uv).prepare(
                profile,
                project_dir=project_dir,
            )
            environments.append(
                EnvironmentState(
                    environment_id=profile.environment_id,
                    node=node_id,
                    project=profile.project,
                    group=profile.group,
                    path=_environment_path(project_dir, profile.path),
                    status="ready",
                )
            )
            progress.advance(node_id)

        progress.update(node_id, "Verifying configured resources")
        _probe_resources(context, executor, node_state)
        progress.advance(node_id)
        progress.succeed(node_id)
        return NodeInitialization(node_state, tuple(environments), inference_commit)


def _profiles_on(
    deployment: DeploymentPlan,
    node_id: str,
) -> tuple[EnvironmentProfile, ...]:
    return tuple(profile for profile in deployment.environments if profile.node == node_id)


def _prepare_projects(
    context: DeploymentContext,
    executor: Executor,
    node: NodeState,
    *,
    git: str,
) -> str | None:
    projects = {profile.project for profile in _profiles_on(context.deployment, node.node_id)}
    projects.add("deploy")
    manager = ProjectManager(executor, git_executable=git)
    if "deploy" in projects:
        manager.prepare(
            repository=DEPLOY_REPOSITORY,
            revision=context.deployment.deploy_commit,
            project_dir=node.deploy_project,
        )
    inference_commit: str | None = None
    source_descriptors = [
        provider(model.backend)
        for model in context.config.models.values()
        if model.lifecycle == "managed"
        and model.node == node.node_id
        and provider(model.backend).requires_source_checkout
    ]
    needs_inference_source = bool(source_descriptors)
    if "inference" in projects and not needs_inference_source:
        executor.run(Command(("mkdir", "-p", node.inference_project)))
    if "inference" in projects and needs_inference_source:
        repositories = {descriptor.source_repository for descriptor in source_descriptors}
        submodule_paths = {descriptor.source_submodule_path for descriptor in source_descriptors}
        if None in repositories or None in submodule_paths or len(repositories) != 1 or len(submodule_paths) != 1:
            raise InitError("managed inference providers declare incompatible source checkouts")
        repository = next(iter(repositories))
        submodule_path = next(iter(submodule_paths))
        assert repository is not None and submodule_path is not None
        inference_commit = manager.submodule_revision(
            project_dir=node.deploy_project,
            revision=context.deployment.deploy_commit,
            path=submodule_path,
        )
        if inference_commit is not None:
            manager.prepare(
                repository=repository,
                revision=inference_commit,
                project_dir=node.inference_project,
            )
    return inference_commit


def _probe_resources(
    context: DeploymentContext,
    executor: Executor,
    node: NodeState,
) -> None:
    for robot in context.config.robots.values():
        if robot.node != node.node_id:
            continue
        for port in robot_ports(robot):
            _require_path(
                executor,
                port,
                kind="e",
                description=f"robot {robot.robot_id!r} port {port!r}",
            )
        calibration = robot.options.get("calibration_dir")
        if isinstance(calibration, str):
            calibration_path = _configured_path(calibration, node.deploy_project)
            _require_path(
                executor,
                calibration_path,
                kind="d",
                description=f"robot {robot.robot_id!r} calibration directory",
            )
            if robot.kind == "lerobot.bi_so101":
                for calibration_id in robot_calibration_ids(robot):
                    _require_path(
                        executor,
                        posixpath.join(calibration_path, f"{calibration_id}.json"),
                        kind="f",
                        description=(f"robot {robot.robot_id!r} calibration {calibration_id!r}"),
                    )
    for model in context.config.models.values():
        if model.lifecycle == "external" or model.node != node.node_id:
            continue
        descriptor = provider(model.backend)
        model_project = _project_dir(node, descriptor.source_project)
        source = model.options.get("source")
        if isinstance(source, str) and source.startswith(("/", "./", "../")):
            _require_path(
                executor,
                _configured_path(source, model_project),
                kind="e",
                description=f"model {model.model_id!r} source",
            )
        adapter = model.options.get("adapter_config")
        if isinstance(adapter, str):
            _require_path(
                executor,
                _configured_path(adapter, node.deploy_project),
                kind="f",
                description=f"model {model.model_id!r} adapter config",
            )
        pipeline_config = model.options.get("pipeline_config")
        if isinstance(pipeline_config, str):
            _require_path(
                executor,
                _configured_path(pipeline_config, model_project),
                kind="f",
                description=f"model {model.model_id!r} pipeline config",
            )


def _initial_service_state(
    deployment: DeploymentPlan,
    previous: DeploymentState | None,
    digest: str,
) -> dict[str, ServiceState]:
    result: dict[str, ServiceState] = {}
    preserve = previous is not None and previous.config_digest == digest
    for service in deployment.services:
        old = previous.services.get(service.service_id) if preserve else None
        if old is not None and old.node == service.node and old.endpoint == service.endpoint:
            result[service.service_id] = old
        else:
            result[service.service_id] = ServiceState(
                service_id=service.service_id,
                node=service.node,
                endpoint=service.endpoint,
            )
    return result


def _project_dir(node: NodeState, project: str) -> str:
    if project == "deploy":
        return node.deploy_project
    if project == "inference":
        return node.inference_project
    raise InitError(f"unsupported project role {project!r}")


def _environment_path(project_dir: str, path: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(project_dir, path))


def _configured_path(path: str, project_dir: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(project_dir, path))


def _require_path(
    executor: Executor,
    path: str,
    *,
    kind: str,
    description: str,
) -> None:
    result = executor.run(Command(("test", f"-{kind}", path)), check=False)
    if result.exit_code != 0:
        raise InitError(f"{description} was not found at {path!r}")


def _node_error_summary(command: str, errors: dict[str, Exception]) -> str:
    details = "; ".join(f"{node_id}: {error}" for node_id, error in sorted(errors.items()))
    return f"{command} failed on {len(errors)} node(s): {details}"


__all__ = ["initialize"]
