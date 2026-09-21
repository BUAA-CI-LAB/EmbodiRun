"""Compatibility alias for the canonical control model loop."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.application.model_loop")
_sys.modules[__name__] = _canonical
