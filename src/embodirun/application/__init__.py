"""Canonical application domain for control and simulation services.

HTTP listeners remain in ``services.control.server`` and
``services.simulation.server``. This package owns request contracts,
authorization, jobs, model loops, and domain coordinators without opening a
transport listener at import time. The heavier coordinator is lazy-imported to
keep contract imports cycle-free.
"""

from .auth import AuthenticationError, AuthorizationError, AuthPolicy, Role
from .contracts import (
    ControlContractError,
    ControlRuntimeProfile,
    ControlServiceConfig,
    TaskRequest,
    TaskResult,
)

_LAZY_CONTROL = {"ControlService", "ControlServiceError", "ControlTaskRejected"}


def __getattr__(name: str):
    if name in _LAZY_CONTROL:
        from .control_service import (
            ControlService,
            ControlServiceError,
            ControlTaskRejected,
        )

        return {
            "ControlService": ControlService,
            "ControlServiceError": ControlServiceError,
            "ControlTaskRejected": ControlTaskRejected,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AuthPolicy",
    "AuthenticationError",
    "AuthorizationError",
    "ControlContractError",
    "ControlRuntimeProfile",
    "ControlService",
    "ControlServiceConfig",
    "ControlServiceError",
    "ControlTaskRejected",
    "Role",
    "TaskRequest",
    "TaskResult",
]
