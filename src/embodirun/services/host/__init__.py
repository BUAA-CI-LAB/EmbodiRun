"""Compatibility facade for the canonical :mod:`embodirun.deployment` API."""

from embodirun.deployment import (
    ConfigError,
    DeploymentConfig,
    DeploymentPlan,
    EnvironmentError,
    EnvironmentProfile,
    ProcessStatus,
    RuntimeSpec,
    ServiceError,
    ServiceSpec,
    ServiceSupervisor,
    SupervisorError,
    UvEnvironmentManager,
    build_plan,
    environment_profiles,
    load_config,
)

__all__ = [
    "ConfigError",
    "DeploymentConfig",
    "DeploymentPlan",
    "EnvironmentError",
    "EnvironmentProfile",
    "ProcessStatus",
    "RuntimeSpec",
    "ServiceError",
    "ServiceSpec",
    "ServiceSupervisor",
    "SupervisorError",
    "UvEnvironmentManager",
    "build_plan",
    "environment_profiles",
    "load_config",
]
