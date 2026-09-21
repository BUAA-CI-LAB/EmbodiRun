"""Compatibility alias for :mod:`embodirun.deployment.executor.ssh`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.executor.ssh")
_sys.modules[__name__] = _canonical
