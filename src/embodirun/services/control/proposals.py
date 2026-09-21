"""Compatibility alias for :mod:`embodirun.application.proposals`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.application.proposals")
_sys.modules[__name__] = _canonical
