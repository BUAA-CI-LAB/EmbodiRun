"""SGLang inference service integrations."""

import os
import shutil
import sysconfig
from pathlib import Path

from .http import SglangHttpClient, SglangHttpError, sglang_server_command


def prepare_sglang_environment() -> None:
    """Find the pip CUDA toolkit unless the service already selected a toolkit."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or shutil.which("nvcc"):
        return
    toolkit = Path(sysconfig.get_path("purelib")) / "nvidia" / "cu13"
    if (toolkit / "bin" / "nvcc").is_file():
        os.environ["CUDA_HOME"] = str(toolkit)


__all__ = [
    "SglangHttpClient",
    "SglangHttpError",
    "prepare_sglang_environment",
    "sglang_server_command",
]
