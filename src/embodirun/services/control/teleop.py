"""Compatibility facade; canonical implementation lives in ``embodirun.devices.execution.teleop``."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.devices.execution.teleop")
_sys.modules[__name__] = _canonical


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI test
    raise SystemExit(_canonical.main())
