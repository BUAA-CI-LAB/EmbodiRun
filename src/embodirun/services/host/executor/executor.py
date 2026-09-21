"""Compatibility alias for :mod:`embodirun.deployment.executor.executor`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.executor.executor")
_sys.modules[__name__] = _canonical
