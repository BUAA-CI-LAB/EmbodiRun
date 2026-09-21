"""Canonical deployment domain: config, planning, execution, and lifecycle.

The CLI remains under ``services.host.cli``; this package owns the reusable
Host implementation so non-CLI callers do not depend on process entrypoints.
"""

from .config import ConfigError, DeploymentConfig, load_config
from .environment import (
    EnvironmentError,
    EnvironmentProfile,
    UvEnvironmentManager,
    environment_profiles,
)
from .plan import (
    DeploymentPlan,
    RuntimeSpec,
    ServiceError,
    ServiceSpec,
    build_plan,
)
from .supervisor import ProcessStatus, ServiceSupervisor, SupervisorError

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
