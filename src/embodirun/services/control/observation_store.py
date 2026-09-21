"""Compatibility facade; canonical implementation lives in ``embodirun.devices.observations.store``."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.devices.observations.store")
_sys.modules[__name__] = _canonical
