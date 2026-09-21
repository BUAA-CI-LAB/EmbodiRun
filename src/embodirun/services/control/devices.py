"""Compatibility facade; canonical implementation lives in ``embodirun.devices.lifecycle``."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.devices.lifecycle")
_sys.modules[__name__] = _canonical
