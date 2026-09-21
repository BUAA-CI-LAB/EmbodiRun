"""RealSense camera implementation."""

from collections.abc import Sequence

from ... import SensorInput
from .camera import (
    RealSenseCameraConfig,
    RealSenseCameraError,
    RealSenseCameraSource,
)


def create_source(inputs: Sequence[SensorInput]) -> RealSenseCameraSource:
    """Create persistent RealSense pipelines from generic sensor inputs."""

    cameras: list[RealSenseCameraConfig] = []
    allowed = {
        "serial",
        "width",
        "height",
        "fps",
        "jpeg_quality",
        "frame_timeout_s",
    }
    for item in inputs:
        options = dict(item.options)
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise RealSenseCameraError(
                f"sensor {item.sensor_id!r} contains unknown RealSense fields: {', '.join(unknown)}"
            )
        try:
            cameras.append(RealSenseCameraConfig(name=item.name, **options))
        except (TypeError, ValueError) as error:
            raise RealSenseCameraError(
                f"sensor {item.sensor_id!r} has invalid RealSense configuration: {error}"
            ) from error
    return RealSenseCameraSource(cameras)


__all__ = [
    "RealSenseCameraConfig",
    "RealSenseCameraError",
    "RealSenseCameraSource",
    "create_source",
]
