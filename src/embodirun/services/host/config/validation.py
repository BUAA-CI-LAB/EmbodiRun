"""Compatibility alias for :mod:`embodirun.deployment.config.validation`."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.config.validation")
_sys.modules[__name__] = _canonical
