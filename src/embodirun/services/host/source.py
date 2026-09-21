"""Compatibility alias for :mod:`embodirun.deployment.source`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.source")
_sys.modules[__name__] = _canonical
