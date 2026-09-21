"""Compatibility alias for :mod:`embodirun.deployment.config.runtime`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.config.runtime")
_sys.modules[__name__] = _canonical
