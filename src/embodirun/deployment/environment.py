"""Describe and synchronize the uv environments required by a deployment."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, replace
from typing import Literal

from embodirun.model_services.providers import provider
from embodirun.robots import robot_definition
from embodirun.simulators import simulator_definition

from .config import DeploymentConfig
from .executor import Command, CommandResult, Executor

_MODEL_PYTHON: dict[str, str] = {
    "pi05": "3.12",
    "streamvln": "3.12",
}


class EnvironmentError(ValueError):
    """A deployment environment cannot be resolved or prepared safely."""


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """One reusable environment on one deployment node."""

    environment_id: str
    node: str
    project: Literal["deploy", "inference"]
    group: str
    path: str
    python: str | None = None
    package_index: str | None = None
    packages: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    install: Literal["project-group", "packages"] = "project-group"


def environment_profiles(config: DeploymentConfig) -> tuple[EnvironmentProfile, ...]:
    """Return the distinct uv environments needed by the deployment."""

    profiles: list[EnvironmentProfile] = []
    wireless_robots = {
        runtime.robot
        for runtime in config.runtimes.values()
        if runtime.robot is not None
        and runtime.model is not None
        and config.models[runtime.model].transport == "wireless"
    }
    wireless_simulators = {
        runtime.simulator
        for runtime in config.runtimes.values()
        if runtime.simulator is not None
        and runtime.model is not None
        and config.models[runtime.model].transport == "wireless"
    }
    for robot in sorted(config.robots.values(), key=lambda item: item.robot_id):
        try:
            group, python = robot_environment_profile(robot.kind)
        except EnvironmentError:
            raise EnvironmentError(
                f"robot {robot.robot_id!r} has no environment profile for type {robot.kind!r}"
            ) from None
        profiles.append(
            EnvironmentProfile(
                environment_id=f"{robot.node}:deploy:{group}",
                node=robot.node,
                project="deploy",
                group=group,
                path=f".venv-{group}",
                python=python,
                extras=("wireless",) if robot.robot_id in wireless_robots else (),
            )
        )

    for simulator in sorted(config.simulators.values(), key=lambda item: item.simulator_id):
        try:
            definition = simulator_definition(simulator.kind)
        except (KeyError, TypeError):
            raise EnvironmentError(
                f"simulator {simulator.simulator_id!r} has no environment profile for type {simulator.kind!r}"
            ) from None
        profiles.append(
            EnvironmentProfile(
                environment_id=(f"{simulator.node}:deploy:{definition.environment_group}"),
                node=simulator.node,
                project="deploy",
                group=definition.environment_group,
                path=f".venv-{definition.environment_group}",
                python=definition.python,
                extras=(("wireless",) if simulator.simulator_id in wireless_simulators else ()),
            )
        )

    for model in sorted(config.models.values(), key=lambda item: item.model_id):
        if model.lifecycle == "external":
            continue
        try:
            descriptor = provider(model.backend)
        except ValueError as error:
            raise EnvironmentError(str(error)) from error
        if descriptor.managed_command is None:
            raise EnvironmentError(f"provider {model.backend!r} has no managed service descriptor")
        if descriptor.requires_environment_packages and not model.environment_packages:
            raise EnvironmentError(
                f"model {model.model_id!r} must set environment_packages for provider {model.backend!r}"
            )
        assert model.node is not None
        group = descriptor.environment_group or model.kind
        path = model.environment or f".venv-{model.backend}-{model.kind}"
        package_requirements = (
            *model.environment_packages,
            *descriptor.environment_packages(model.options),
        )
        profiles.append(
            EnvironmentProfile(
                environment_id=f"{model.node}:inference:{group}:{path}",
                node=model.node,
                project=descriptor.source_project,
                group=group,
                path=path,
                python=model.python or descriptor.default_python or _MODEL_PYTHON.get(group),
                package_index=model.environment_index,
                packages=package_requirements,
                extras=("wireless",) if model.transport == "wireless" else (),
                install=("project-group" if not descriptor.requires_environment_packages else "packages"),
            )
        )

    return _deduplicate(profiles)


def robot_environment_profile(kind: str) -> tuple[str, str | None]:
    """Resolve a robot adapter type to its Deploy dependency group."""

    try:
        definition = robot_definition(kind)
    except (KeyError, TypeError):
        raise EnvironmentError(f"unsupported robot type {kind!r}") from None
    return definition.environment_group, definition.python


def _deduplicate(profiles: list[EnvironmentProfile]) -> tuple[EnvironmentProfile, ...]:
    by_identity: dict[tuple[str, str], EnvironmentProfile] = {}
    for profile in profiles:
        identity = (profile.node, profile.path)
        previous = by_identity.get(identity)
        if previous is not None and replace(previous, extras=()) != replace(profile, extras=()):
            raise EnvironmentError(
                f"environment path {profile.path!r} on node {profile.node!r} is assigned incompatible profiles"
            )
        if previous is None:
            by_identity[identity] = profile
        else:
            by_identity[identity] = replace(
                previous,
                extras=tuple(sorted(set(previous.extras) | set(profile.extras))),
            )
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (item.node, item.project, item.group, item.path),
        )
    )


class UvEnvironmentManager:
    """Prepare an environment through an injected local or SSH executor."""

    def __init__(self, executor: Executor, *, uv_executable: str = "uv") -> None:
        self.executor = executor
        self.uv_executable = uv_executable

    def sync_command(self, profile: EnvironmentProfile, *, project_dir: str) -> Command:
        if profile.install == "packages":
            argv = [self.uv_executable, "venv"]
            if profile.python is not None:
                argv.extend(("--python", profile.python))
            argv.append(profile.path)
            return Command(
                argv=tuple(argv),
                cwd=project_dir,
                timeout_s=1800.0,
            )
        argv = [
            self.uv_executable,
            "sync",
            "--frozen",
            "--no-dev",
            "--group",
            profile.group,
        ]
        if profile.python is not None:
            argv[2:2] = ["--python", profile.python]
        for extra in profile.extras:
            argv.extend(("--extra", extra))
        return Command(
            argv=tuple(argv),
            cwd=project_dir,
            environment={"UV_PROJECT_ENVIRONMENT": profile.path},
            timeout_s=1800.0,
        )

    def prepare(
        self,
        profile: EnvironmentProfile,
        *,
        project_dir: str,
    ) -> CommandResult:
        """Synchronize a project group or a pinned standalone package set."""

        result = self._existing_environment(profile, project_dir=project_dir) if profile.install == "packages" else None
        if result is None:
            result = self.executor.run(self.sync_command(profile, project_dir=project_dir))
        if not profile.packages:
            return result
        return self.executor.run(self.package_command(profile, project_dir=project_dir))

    def _existing_environment(
        self,
        profile: EnvironmentProfile,
        *,
        project_dir: str,
    ) -> CommandResult | None:
        """Reuse a valid standalone environment without changing its interpreter."""

        path = posixpath.join(project_dir, profile.path)
        exists = self.executor.run(
            Command(("test", "-f", posixpath.join(path, "pyvenv.cfg")), timeout_s=20.0),
            check=False,
        )
        if exists.exit_code != 0:
            return None

        find_python = (
            self.uv_executable,
            "python",
            "find",
            "--no-project",
            "--no-python-downloads",
            "--resolve-links",
        )
        actual = self.executor.run(
            Command((*find_python, path), cwd=project_dir, timeout_s=20.0),
            check=False,
        )
        if actual.exit_code != 0 or not actual.stdout.strip():
            raise EnvironmentError(
                f"environment {path!r} has an unusable Python interpreter; "
                "choose a different environment path or repair it before retrying init"
            )
        if profile.python is not None:
            # uv prefers the active environment if it satisfies the request;
            # resolving links also supports requests naming an interpreter path.
            requested = self.executor.run(
                Command(
                    (*find_python, profile.python),
                    cwd=project_dir,
                    environment={"VIRTUAL_ENV": path},
                    timeout_s=20.0,
                ),
                check=False,
            )
            if requested.exit_code != 0 or requested.stdout.strip() != actual.stdout.strip():
                raise EnvironmentError(
                    f"environment {path!r} does not match configured Python "
                    f"{profile.python!r}; choose a different environment path or "
                    "explicitly rebuild it before retrying init"
                )
        return actual

    def package_command(
        self,
        profile: EnvironmentProfile,
        *,
        project_dir: str,
    ) -> Command:
        if not profile.packages:
            raise ValueError("environment profile has no package overlay")
        environment_path = profile.path
        if not environment_path.startswith("/"):
            environment_path = posixpath.join(project_dir, environment_path)
        argv = [
            self.uv_executable,
            "pip",
            "install",
            "--python",
            posixpath.join(environment_path, "bin", "python"),
        ]
        if profile.package_index is not None:
            argv.extend(("--index-url", profile.package_index))
        argv.extend(profile.packages)
        return Command(
            argv=tuple(argv),
            cwd=project_dir,
            timeout_s=1800.0,
        )


__all__ = [
    "EnvironmentError",
    "EnvironmentProfile",
    "UvEnvironmentManager",
    "environment_profiles",
    "robot_environment_profile",
]
