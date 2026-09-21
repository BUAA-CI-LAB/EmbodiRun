"""V4L2 camera implementation."""

from collections.abc import Sequence

from ... import SensorInput
from .camera import CameraError, V4L2CameraConfig, V4L2CameraSource


def create_source(inputs: Sequence[SensorInput]) -> V4L2CameraSource:
    """Create one batched V4L2 source from generic sensor inputs."""

    cameras: list[V4L2CameraConfig] = []
    for item in inputs:
        options = dict(item.options)
        unknown = sorted(set(options) - {"device", "width", "height", "fps"})
        if unknown:
            raise CameraError(f"sensor {item.sensor_id!r} contains unknown V4L2 fields: {', '.join(unknown)}")
        missing = [name for name in ("device", "width", "height", "fps") if name not in options]
        if missing:
            raise CameraError(f"sensor {item.sensor_id!r} is missing V4L2 fields: {', '.join(missing)}")
        try:
            cameras.append(
                V4L2CameraConfig(
                    name=item.name,
                    device=options["device"],
                    width=options["width"],
                    height=options["height"],
                    fps=options["fps"],
                )
            )
        except (TypeError, ValueError) as error:
            raise CameraError(f"sensor {item.sensor_id!r} has invalid V4L2 configuration: {error}") from error
    return V4L2CameraSource(cameras)


__all__ = [
    "CameraError",
    "V4L2CameraConfig",
    "V4L2CameraSource",
    "create_source",
]
