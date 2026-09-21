"""Compatibility alias for :mod:`embodirun.application.direct_execution`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.application.direct_execution")
_sys.modules[__name__] = _canonical
