"""Synchronize local Deploy source into managed node overlays.

The operation archives the supplied local path, uploads it to configured nodes,
and records per-node failures. It does not restart inference or contact an
external service endpoint; callers must perform a separate managed restart.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, replace
from pathlib import Path

from embodirun.deployment.environment import EnvironmentProfile, UvEnvironmentManager
from embodirun.deployment.executor import Command, Executor
from embodirun.deployment.source import (
    SourceArchive,
    active_deploy_project,
    build_source_archive,
    install_source_release,
)
from embodirun.deployment.state import DeploymentState, NodeState, StateStore

from ._context import DeploymentContext
from ._parallel import run_on_nodes


class SyncError(RuntimeError):
    """Local Deploy code cannot be synchronized safely."""


@dataclass(frozen=True, slots=True)
class NodeSync:
    uploaded: bool
    dependencies_updated: bool


def synchronize_deploy(source: Path, context: DeploymentContext) -> int:
    """Upload and activate a Deploy source archive without restarting inference.

    ``source`` is the local repository path used to build the archive. Dependency
    and overlay state is persisted remotely through the deployment executor; a
    node failure is reported after other node results are collected.
    """
    archive = build_source_archive(source)
    state = _initialized_state(context)
    _require_control_services_stopped(context, state)
    profiles = _deploy_profiles_by_node(context, state)

    progress = context.progress
    progress.begin("sync", context.deployment.name)
    for node_id in profiles:
        progress.add_node(node_id, total=3)

    def synchronize(node_id: str) -> NodeSync:
        try:
            result = _sync_node(
                context,
                state,
                node_id,
                profiles[node_id],
                archive,
            )
        except Exception as error:
            progress.fail(node_id, error)
            raise
        progress.succeed(
            node_id,
            detail=("Deploy overlay active" + ("; dependencies updated" if result.dependencies_updated else "")),
        )
        return result

    results = run_on_nodes(profiles, synchronize)
    success = not results.errors
    progress.finish(success=success)
    if results.errors:
        raise SyncError(_error_summary(results.errors))

    uploads = sum(result.uploaded for result in results.values.values())
    progress.message(
        f"Deploy overlay {archive.digest[:12]} active on {len(results.values)} "
        f"node(s) ({uploads} upload(s)); inference services were not restarted; "
        "restart control services to load the new source"
    )
    return 0


def _initialized_state(context: DeploymentContext) -> DeploymentState:
    state = StateStore(context.state_path).load()
    if state is None:
        raise SyncError("deployment is not initialized; run `embodirun ... init`")
    return state


def _require_control_services_stopped(
    context: DeploymentContext,
    state: DeploymentState,
) -> None:
    control_ids = {service.service_id for service in context.deployment.services if service.kind == "control"}
    running = sorted(
        service_id
        for service_id in control_ids
        if service_id in state.services and state.services[service_id].status == "running"
    )
    if running:
        raise SyncError(
            f"control services are running: {', '.join(running)}; run `embodirun ... down --target control` before sync"
        )


def _deploy_profiles_by_node(
    context: DeploymentContext,
    state: DeploymentState,
) -> dict[str, tuple[EnvironmentProfile, ...]]:
    profiles: dict[str, list[EnvironmentProfile]] = {}
    for profile in context.deployment.environments:
        if profile.project != "deploy":
            continue
        environment = state.environments.get(profile.environment_id)
        if environment is None or environment.status != "ready":
            raise SyncError(f"environment {profile.environment_id!r} is not ready; run init again")
        if profile.node not in state.nodes:
            raise SyncError(f"initialized state is missing node {profile.node!r}")
        profiles.setdefault(profile.node, []).append(replace(profile, path=environment.path))
    if not profiles:
        raise SyncError("deployment has no Deploy environments to synchronize")
    return {node_id: tuple(node_profiles) for node_id, node_profiles in sorted(profiles.items())}


def _sync_node(
    context: DeploymentContext,
    state: DeploymentState,
    node_id: str,
    profiles: tuple[EnvironmentProfile, ...],
    archive: SourceArchive,
) -> NodeSync:
    progress = context.progress
    node = state.nodes[node_id]
    overlay_root = posixpath.join(node.root, "overlays", "deploy")
    dependency_marker = posixpath.join(overlay_root, "dependency-digest")
    current = active_deploy_project(node.root)

    with context.executor(node_id) as executor:
        progress.update(node_id, "Uploading Deploy source", detail=archive.digest[:12])
        installed = install_source_release(
            executor,
            overlay_root=overlay_root,
            archive=archive,
        )
        progress.advance(node_id)

        progress.update(node_id, "Checking Deploy dependencies")
        dependencies_updated = _synchronize_dependencies(
            executor,
            node=node,
            release=installed.path,
            active_project=current,
            dependency_marker=dependency_marker,
            profiles=profiles,
            archive=archive,
        )
        progress.advance(node_id)

        progress.update(node_id, "Activating Deploy overlay")
        executor.replace_symlink(current, installed.path)
        progress.advance(node_id)
    return NodeSync(installed.uploaded, dependencies_updated)


def _synchronize_dependencies(
    executor: Executor,
    *,
    node: NodeState,
    release: str,
    active_project: str,
    dependency_marker: str,
    profiles: tuple[EnvironmentProfile, ...],
    archive: SourceArchive,
) -> bool:
    recorded = _read_optional_file(executor, dependency_marker)
    expected = f"{archive.dependency_digest}\n".encode()
    if recorded == expected and _symlink_target(executor, active_project) == release:
        return False

    changed = not _managed_dependencies_match(executor, node, archive)
    if changed:
        uv = _uv_executable(executor, node)
        manager = UvEnvironmentManager(executor, uv_executable=uv)
        for profile in profiles:
            manager.prepare(profile, project_dir=release)

    executor.write_text(
        dependency_marker,
        expected.decode(),
        mode=0o600,
    )
    return changed


def _symlink_target(executor: Executor, path: str) -> str | None:
    result = executor.run(Command(("readlink", path)), check=False)
    if result.exit_code == 1:
        return None
    if result.exit_code != 0:
        detail = result.stderr.strip() or f"exit code {result.exit_code}"
        raise SyncError(f"could not inspect active Deploy source: {detail}")
    return result.stdout.strip()


def _managed_dependencies_match(
    executor: Executor,
    node: NodeState,
    archive: SourceArchive,
) -> bool:
    for name, expected in archive.dependency_files.items():
        remote_path = posixpath.join(node.deploy_project, name)
        available = executor.run(Command(("test", "-f", remote_path)), check=False)
        if available.exit_code == 1:
            return False
        if available.exit_code != 0:
            detail = available.stderr.strip() or f"exit code {available.exit_code}"
            raise SyncError(f"could not inspect {remote_path}: {detail}")
        if executor.read_bytes(remote_path) != expected:
            return False
    return True


def _read_optional_file(executor: Executor, path: str) -> bytes | None:
    available = executor.run(Command(("test", "-f", path)), check=False)
    if available.exit_code == 1:
        return None
    if available.exit_code != 0:
        detail = available.stderr.strip() or f"exit code {available.exit_code}"
        raise SyncError(f"could not inspect {path}: {detail}")
    return executor.read_bytes(path)


def _uv_executable(executor: Executor, node: NodeState) -> str:
    candidate = posixpath.join(node.home, ".local", "bin", "uv")
    available = executor.run(Command(("test", "-x", candidate)), check=False)
    if available.exit_code == 0:
        return candidate
    if available.exit_code == 1:
        return "uv"
    detail = available.stderr.strip() or f"exit code {available.exit_code}"
    raise SyncError(f"could not inspect uv executable: {detail}")


def _error_summary(errors: dict[str, Exception]) -> str:
    return "; ".join(f"{node}: {error}" for node, error in sorted(errors.items()))


__all__ = ["synchronize_deploy"]
