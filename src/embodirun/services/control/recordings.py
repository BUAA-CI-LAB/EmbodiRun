"""Compatibility facade; canonical implementation lives in ``embodirun.devices.recording``."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.devices.recording")
_sys.modules[__name__] = _canonical
