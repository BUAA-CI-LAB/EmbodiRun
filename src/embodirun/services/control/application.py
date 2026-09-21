"""Compatibility alias for the canonical application API facade."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.application.api")
_sys.modules[__name__] = _canonical
