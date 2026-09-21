"""Compatibility imports for the package now named :mod:`embodirun`."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, spec):
        self.spec = spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Reuse the canonical module so registries, exceptions and dataclasses
        # have the same identity even when both import names are used together.
        sys.modules[module.__name__] = importlib.import_module(self.spec.name)

    def get_code(self, fullname):
        # Keep legacy `python -m rlinf_deploy.<module>` invocations working.
        return self.spec.loader.get_code(self.spec.name)


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("rlinf_deploy."):
            return None
        canonical = "embodirun" + fullname[len("rlinf_deploy") :]
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _LegacyLoader(spec),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None,
        )


sys.meta_path.insert(0, _LegacyFinder())
sys.modules[__name__] = importlib.import_module("embodirun")
