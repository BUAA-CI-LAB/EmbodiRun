"""Camera contracts and hardware implementations."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import cache
from importlib import import_module

from .. import SensorInput
from .camera import CameraFrame, CameraSource, CameraSources, RawCameraFrame
from .fake import FakeCameraConfig, FakeCameraError, FakeCameraSource
from .realsense import (
    RealSenseCameraConfig,
    RealSenseCameraError,
    RealSenseCameraSource,
)
from .v4l2 import CameraError, V4L2CameraConfig, V4L2CameraSource

_CAMERA_KIND = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


def create_camera_source(inputs: Sequence[SensorInput]) -> CameraSource:
    """Create camera sources by each YAML-declared sensor kind."""

    configured = tuple(inputs)
    if not configured:
        raise ValueError("at least one camera input is required")
    names = [item.name for item in configured]
    if len(names) != len(set(names)):
        raise ValueError("camera input names must be unique")

    grouped: dict[str, list[SensorInput]] = {}
    for item in configured:
        grouped.setdefault(item.kind, []).append(item)

    sources: list[CameraSource] = []
    try:
        for kind, items in grouped.items():
            sources.append(_source_builder(kind)(items))
    except BaseException:
        for source in reversed(sources):
            source.close()
        raise
    if len(sources) == 1:
        return sources[0]
    return CameraSources(tuple(sources))


@cache
def _source_builder(
    kind: str,
) -> Callable[[Sequence[SensorInput]], CameraSource]:
    if not isinstance(kind, str) or _CAMERA_KIND.fullmatch(kind) is None:
        raise ValueError(f"invalid camera kind {kind!r}")
    module_name = f"{__name__}.{kind}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is not None and (error.name == module_name or module_name.startswith(f"{error.name}.")):
            raise ValueError(f"camera kind {kind!r} is not available") from None
        raise
    builder = getattr(module, "create_source", None)
    if not callable(builder):
        raise TypeError(f"{module_name} does not define create_source")
    return builder


__all__ = [
    "CameraError",
    "CameraFrame",
    "CameraSource",
    "CameraSources",
    "RawCameraFrame",
    "FakeCameraConfig",
    "FakeCameraError",
    "FakeCameraSource",
    "RealSenseCameraConfig",
    "RealSenseCameraError",
    "RealSenseCameraSource",
    "V4L2CameraConfig",
    "V4L2CameraSource",
    "create_camera_source",
]
