"""Legacy module alias for the canonical EmbodiInfer HTTP backend."""

import sys as _sys
from importlib import import_module as _import_module

_canonical = _import_module("embodirun.model_services.backends.embodiinfer.http")
_sys.modules[__name__] = _canonical
