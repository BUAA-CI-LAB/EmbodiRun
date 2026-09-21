"""Dependency-free static PNG camera implementation."""

from collections.abc import Sequence

from ... import SensorInput
from .camera import FakeCameraConfig, FakeCameraError, FakeCameraSource


def create_source(inputs: Sequence[SensorInput]) -> FakeCameraSource:
    """Build fake camera sources from ordinary Host sensor inputs."""

    cameras: list[FakeCameraConfig] = []
    allowed = {"width", "height", "color"}
    for item in inputs:
        options = dict(item.options)
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise FakeCameraError(
                f"sensor {item.sensor_id!r} contains unknown fake camera fields: {', '.join(unknown)}"
            )
        color = options.get("color", (32, 96, 160))
        if isinstance(color, list):
            color = tuple(color)
        try:
            cameras.append(
                FakeCameraConfig(
                    name=item.name, color=color, **{key: value for key, value in options.items() if key != "color"}
                )
            )
        except (TypeError, ValueError) as error:
            raise FakeCameraError(
                f"sensor {item.sensor_id!r} has invalid fake camera configuration: {error}"
            ) from error
    return FakeCameraSource(cameras)


__all__ = [
    "FakeCameraConfig",
    "FakeCameraError",
    "FakeCameraSource",
    "create_source",
]
