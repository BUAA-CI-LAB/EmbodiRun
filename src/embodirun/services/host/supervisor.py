"""Compatibility alias and bare-script shim for the canonical supervisor."""

import runpy
import sys
from pathlib import Path

if __package__ in {None, ""}:  # direct ``python /abs/path/supervisor.py``
    _SOURCE_ROOT = Path(__file__).resolve().parents[3]
    if str(_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_ROOT))
    if __name__ == "__main__":
        canonical_path = Path(__file__).resolve().parents[3] / "embodirun" / "deployment" / "supervisor.py"
        runpy.run_path(str(canonical_path), run_name="__main__")
        raise SystemExit

from importlib import import_module as _import_module

_canonical = _import_module("embodirun.deployment.supervisor")
sys.modules[__name__] = _canonical
