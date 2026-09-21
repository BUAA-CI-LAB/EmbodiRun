"""Device-domain primitives for lifecycle, execution, observation, and recording.

The package owns physical resource lifecycle, serialized robot I/O and control
arbitration, immutable observation snapshots, and bounded recordings. It does
not own application routing, model inference, agent policy, or HTTP servers.
"""

from .lifecycle import (
    DeviceBusyError,
    DeviceCloseError,
    DeviceError,
    DeviceLease,
    DeviceManager,
    DeviceOpenError,
    DeviceResource,
    DeviceStateError,
    DeviceStateStore,
    DeviceStatus,
    DeviceUncertainError,
    ResourceIdentity,
    canonical_resource_identity,
)

__all__ = [
    "DeviceBusyError",
    "DeviceCloseError",
    "DeviceError",
    "DeviceLease",
    "DeviceManager",
    "DeviceOpenError",
    "DeviceResource",
    "DeviceStateError",
    "DeviceStateStore",
    "DeviceStatus",
    "DeviceUncertainError",
    "ResourceIdentity",
    "canonical_resource_identity",
]
