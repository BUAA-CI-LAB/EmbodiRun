"""Legacy facade for the canonical SGLang backend.

The standard-library module attributes remain available for old tests and
callers that monkeypatch toolkit discovery.  They are the same module objects
used by the canonical implementation, so there is only one implementation.
"""

import os  # noqa: F401 - retained for legacy monkeypatch callers
import shutil  # noqa: F401 - retained for legacy monkeypatch callers
import sysconfig  # noqa: F401 - retained for legacy monkeypatch callers
from pathlib import Path  # noqa: F401 - retained for legacy monkeypatch callers

from embodirun.model_services.backends.sglang import (
    SglangHttpClient,
    SglangHttpError,
    prepare_sglang_environment,
    sglang_server_command,
)

__all__ = [
    "SglangHttpClient",
    "SglangHttpError",
    "prepare_sglang_environment",
    "sglang_server_command",
]
