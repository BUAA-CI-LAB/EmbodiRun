"""Load Unitree SDK components under a verified CycloneDDS runtime."""

from __future__ import annotations

import os
import sys
from typing import Any

UNSAFE_ICEORYX_PUBLISH_SYMBOL = b"iox_pub_publish_chunk"
CONTROL_MODULE = "embodirun.robots.unitree.go2.agent.control"


def loaded_dds_library_path() -> str | None:
    """Return the loaded CycloneDDS C library path on Linux, if available."""

    try:
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            for line in maps_file:
                path = line.split()[-1]
                if "/libddsc.so" in path:
                    return os.path.realpath(path)
    except (OSError, IndexError):
        return None
    return None


def dds_library_has_unsafe_iceoryx(library_path: str) -> bool:
    """Detect the Iceoryx publisher ABI that is unsafe for this binding."""

    with open(library_path, "rb") as library_file:
        return UNSAFE_ICEORYX_PUBLISH_SYMBOL in library_file.read()


def ensure_cyclonedds_library_dir(
    library_dir: str,
    *,
    reexec_args: list[str] | None = None,
) -> str:
    """Validate libddsc and re-exec the package with it first on the path."""

    resolved = os.path.realpath(os.path.abspath(library_dir))
    library = os.path.join(resolved, "libddsc.so.0")
    if not os.path.isfile(library):
        raise RuntimeError(f"CycloneDDS library does not exist: {library}")
    if dds_library_has_unsafe_iceoryx(library):
        raise RuntimeError(f"CycloneDDS library contains the incompatible Iceoryx publisher ABI: {library}")

    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if entries and os.path.realpath(entries[0]) == resolved:
        return resolved

    remaining = [entry for entry in entries if os.path.realpath(entry) != resolved]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([resolved, *remaining])
    arguments = sys.argv[1:] if reexec_args is None else reexec_args
    argv = [sys.executable, "-m", CONTROL_MODULE, *arguments]
    os.execve(sys.executable, argv, environment)
    raise RuntimeError("failed to re-exec with the requested CycloneDDS library")


def load_sdk_components(
    required_dds_lib_dir: str | None,
) -> tuple[Any, Any, Any, Any]:
    """Import SDK factories lazily and verify their already-loaded libddsc."""

    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    except ImportError as exc:
        raise RuntimeError("Unitree SDK2 is unavailable; install unitree_sdk2_python in the robot environment") from exc

    if required_dds_lib_dir is not None:
        expected_dir = os.path.realpath(required_dds_lib_dir)
        loaded_library = loaded_dds_library_path()
        if loaded_library is None or os.path.dirname(loaded_library) != expected_dir:
            raise RuntimeError(
                "loaded CycloneDDS library does not match the required "
                f"directory: loaded={loaded_library!r}, required={expected_dir!r}"
            )
        if dds_library_has_unsafe_iceoryx(loaded_library):
            raise RuntimeError(
                "refusing an Iceoryx-enabled CycloneDDS library that is ABI-incompatible with this Python binding"
            )
    return (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        SportClient,
        SportModeState_,
    )


__all__ = [
    "CONTROL_MODULE",
    "UNSAFE_ICEORYX_PUBLISH_SYMBOL",
    "dds_library_has_unsafe_iceoryx",
    "ensure_cyclonedds_library_dir",
    "load_sdk_components",
    "loaded_dds_library_path",
]
