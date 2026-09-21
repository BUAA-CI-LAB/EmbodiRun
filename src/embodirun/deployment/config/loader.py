"""Load a deployment YAML and validate relationships between its resources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .meta import MetadataConfig, parse_metadata
from .model import ModelConfig, parse_model
from .node import NodeConfig, parse_node
from .robot import RobotConfig, parse_robot, robot_ports
from .runtime import RuntimeConfig, parse_runtime
from .sensor import SensorConfig, parse_sensor
from .simulator import SimulatorConfig, parse_simulator
from .validation import ConfigError, mapping, named_section

_TOP_LEVEL_KEYS = {
    "metadata",
    "nodes",
    "robots",
    "simulators",
    "sensors",
    "models",
    "runtimes",
}


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    path: Path
    metadata: MetadataConfig
    nodes: dict[str, NodeConfig]
    robots: dict[str, RobotConfig]
    simulators: dict[str, SimulatorConfig]
    sensors: dict[str, SensorConfig]
    models: dict[str, ModelConfig]
    runtimes: dict[str, RuntimeConfig]


def load_config(path: str | Path) -> DeploymentConfig:
    """Load one deployment file without resolving secrets or touching nodes."""

    source = Path(path).expanduser().resolve()
    root = mapping(_load_yaml(source), "config")
    unknown = sorted(set(root) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"config contains unknown fields: {', '.join(unknown)}")

    metadata = parse_metadata(mapping(root.get("metadata"), "metadata"))
    nodes = named_section(root.get("nodes"), "nodes", parse_node)
    robots = named_section(root.get("robots", {}), "robots", parse_robot)
    simulators = named_section(root.get("simulators", {}), "simulators", parse_simulator)
    sensors = named_section(root.get("sensors", {}), "sensors", parse_sensor)
    models = named_section(root.get("models", {}), "models", parse_model)
    runtimes = named_section(root.get("runtimes", {}), "runtimes", parse_runtime)
    if not nodes:
        raise ConfigError("nodes must contain at least one node")

    _validate_references(nodes, robots, simulators, sensors, models, runtimes)
    _validate_unique_robot_ports(robots)
    _validate_unique_service_ports(models, runtimes, robots, simulators)
    return DeploymentConfig(
        path=source,
        metadata=metadata,
        nodes=nodes,
        robots=robots,
        simulators=simulators,
        sensors=sensors,
        models=models,
        runtimes=runtimes,
    )


def config_digest(config: DeploymentConfig) -> str:
    """Hash the exact configuration bytes used to initialize a deployment."""

    try:
        payload = config.path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {config.path}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on installed group
        raise RuntimeError(
            "YAML configuration requires the host environment; run `uv sync --frozen --no-dev --group host`"
        ) from error

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise ConfigError("YAML mapping keys must be scalar values") from error
            if duplicate:
                raise ConfigError(f"duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error


def _validate_references(
    nodes: dict[str, NodeConfig],
    robots: dict[str, RobotConfig],
    simulators: dict[str, SimulatorConfig],
    sensors: dict[str, SensorConfig],
    models: dict[str, ModelConfig],
    runtimes: dict[str, RuntimeConfig],
) -> None:
    for robot in robots.values():
        if robot.node not in nodes:
            raise ConfigError(f"robot {robot.robot_id!r} references unknown node {robot.node!r}")
    for simulator in simulators.values():
        if simulator.node not in nodes:
            raise ConfigError(f"simulator {simulator.simulator_id!r} references unknown node {simulator.node!r}")
    for model in models.values():
        if model.lifecycle == "external":
            continue
        assert model.node is not None
        if model.node not in nodes:
            raise ConfigError(f"model {model.model_id!r} references unknown node {model.node!r}")
    for sensor in sensors.values():
        if sensor.node not in nodes:
            raise ConfigError(f"sensor {sensor.sensor_id!r} references unknown node {sensor.node!r}")
    for runtime in runtimes.values():
        if runtime.robot is not None and runtime.robot not in robots:
            raise ConfigError(f"runtime {runtime.runtime_id!r} references unknown robot {runtime.robot!r}")
        if runtime.simulator is not None and runtime.simulator not in simulators:
            raise ConfigError(f"runtime {runtime.runtime_id!r} references unknown simulator {runtime.simulator!r}")
        if runtime.model is None:
            # ``parse_runtime`` already rejects this combination for
            # simulators.  Keeping the check here makes the relationship
            # explicit for callers constructing DeploymentConfig directly.
            if runtime.simulator is not None:
                raise ConfigError(f"runtime {runtime.runtime_id!r} simulator runtimes require a model and binding")
            wireless = False
        else:
            if runtime.model not in models:
                raise ConfigError(f"runtime {runtime.runtime_id!r} references unknown model {runtime.model!r}")
            wireless = models[runtime.model].transport == "wireless"
        if wireless and runtime.inference_client is None:
            raise ConfigError(f"runtimes.{runtime.runtime_id}.inference_client is required for wireless transport")
        if not wireless and runtime.inference_client is not None:
            raise ConfigError(f"runtimes.{runtime.runtime_id}.inference_client requires wireless transport")
        for input_name, sensor_id in runtime.inputs.items():
            if sensor_id not in sensors:
                raise ConfigError(
                    f"runtime {runtime.runtime_id!r} input {input_name!r} references unknown sensor {sensor_id!r}"
                )


def _validate_unique_robot_ports(robots: dict[str, RobotConfig]) -> None:
    owners: dict[tuple[str, str], list[RobotConfig]] = {}
    for robot in robots.values():
        ports = robot_ports(robot)
        for port in ports:
            key = (robot.node, port)
            previous = owners.setdefault(key, [])
            for owner in previous:
                # A resource label is only an alias for a shared adapter when
                # both declarations describe the same adapter and the same
                # complete set of physical ports.  In particular, a dual-arm
                # declaration must not hide a single-arm overlap by reusing a
                # broad resource label.
                same_adapter = (
                    robot.resource is not None
                    and robot.resource == owner.resource
                    and robot.kind == owner.kind
                    and ports == robot_ports(owner)
                )
                if not same_adapter:
                    raise ConfigError(
                        f"robots {owner.robot_id!r} and {robot.robot_id!r} share port {port!r} on node {robot.node!r}"
                    )
            previous.append(robot)


def _validate_unique_service_ports(
    models: dict[str, ModelConfig],
    runtimes: dict[str, RuntimeConfig],
    robots: dict[str, RobotConfig],
    simulators: dict[str, SimulatorConfig],
) -> None:
    owners: dict[tuple[str, int], str] = {}
    for model in models.values():
        if model.lifecycle == "external":
            continue
        key = (model.node, model.server.port)
        owner = owners.get(key)
        if owner is not None:
            raise ConfigError(
                f"models {owner!r} and {model.model_id!r} share port {model.server.port} on node {model.node!r}"
            )
        owners[key] = model.model_id
    for runtime in runtimes.values():
        node = robots[runtime.robot].node if runtime.robot is not None else simulators[runtime.simulator].node
        key = (node, runtime.server.port)
        owner = owners.get(key)
        if owner is not None:
            raise ConfigError(
                f"services {owner!r} and {runtime.runtime_id!r} share port {runtime.server.port} on node {node!r}"
            )
        owners[key] = runtime.runtime_id
        if runtime.inference_client is not None:
            port = runtime.inference_client.server.port
            key = (node, port)
            owner = owners.get(key)
            if owner is not None:
                raise ConfigError(
                    f"services {owner!r} and {runtime.runtime_id!r}.inference_client share port {port} on node {node!r}"
                )
            owners[key] = f"{runtime.runtime_id}.inference_client"


__all__ = ["DeploymentConfig", "config_digest", "load_config"]
