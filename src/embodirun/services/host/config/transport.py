"""Compatibility alias for :mod:`embodirun.deployment.config.transport`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.config.transport")
_sys.modules[__name__] = _canonical
