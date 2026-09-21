"""Compatibility alias for :mod:`embodirun.application.auth`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.application.auth")
_sys.modules[__name__] = _canonical
