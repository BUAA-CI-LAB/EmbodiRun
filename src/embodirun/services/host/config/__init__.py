"""Compatibility exports for :mod:`embodirun.deployment.config`.

This package keeps its historical path so imports of ``services.host.config.*``
remain valid while each leaf module resolves to the canonical implementation.
"""

from embodirun.deployment.config import (
    ConfigError,
    ConnectionConfig,
    DeploymentConfig,
    InferenceClientConfig,
    MetadataConfig,
    ModelConfig,
    NodeConfig,
    RobotConfig,
    RuntimeConfig,
    SensorConfig,
    ServerConfig,
    SimulatorConfig,
    config_digest,
    load_config,
)

__all__ = [
    "ConfigError",
    "ConnectionConfig",
    "DeploymentConfig",
    "InferenceClientConfig",
    "MetadataConfig",
    "ModelConfig",
    "NodeConfig",
    "RobotConfig",
    "RuntimeConfig",
    "SensorConfig",
    "SimulatorConfig",
    "ServerConfig",
    "config_digest",
    "load_config",
]
