"""Compatibility import for the optional SGLang LeRobot Pi05 package.

Model math and checkpoint processing live in ``integrations/sglang_pi05``.
Install that package in the managed SGLang environment before importing this
historical path.
"""

from __future__ import annotations

try:
    from embodirun_sglang_pi05.pi05 import (  # type: ignore[import-not-found]
        LeRobotPi05Pipeline,
        _LeRobotPolicyModel,
        _Statistics,
        main,
    )
except ImportError as error:  # pragma: no cover - optional environment
    raise ImportError(
        "SGLang Pi05 integration is optional; install integrations/sglang_pi05 in the managed inference environment"
    ) from error

__all__ = ["LeRobotPi05Pipeline", "_LeRobotPolicyModel", "_Statistics", "main"]
