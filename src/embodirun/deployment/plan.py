"""Resolve deployment configuration into an executable service plan."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from embodirun.application.contracts import ControlRuntimeProfile, ControlServiceConfig
from embodirun.application.simulation.contracts import SimulationServiceConfig
from embodirun.bindings import BindingDefinition, binding_definition
from embodirun.devices import (
    ResourceIdentity,
    canonical_resource_identity,
)
from embodirun.model_services.providers import (
    ManagedCommandOptions,
    ProviderOptionsContext,
    provider,
)
from embodirun.robots.sensors import SensorInput
from embodirun.simulators import simulator_definition

from .config import DeploymentConfig, ModelConfig, RuntimeConfig
from .config.robot import robot_ports
from .environment import (
    EnvironmentProfile,
    environment_profiles,
    robot_environment_profile,
)
from .executor import Command
from .network import service_host


class ServiceError(ValueError):
    """A deployment configuration cannot be resolved into a valid plan."""


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """A long-running process that can be started on a deployment node."""

    service_id: str
    kind: Literal["model", "control", "simulation", "sensor"]
    node: str
    environment_id: str
    endpoint: str
    health_endpoint: str | None
    command: Command
    adapter_config_json: str | None = None
    control_config_json: str | None = None
    simulation_config_json: str | None = None
    wireless_config_json: str | None = None
    # Physical resources are descriptive plan data.  Control owns the actual
    # leases at runtime; keeping these fields here lets Host explain why two
    # runtime services share one owner without inventing a second lock layer.
    owner_id: str | None = None
    resource_ids: tuple[str, ...] = ()
    runtime_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """One policy binding executed beside a robot or simulator."""

    runtime_id: str
    node: str
    target_kind: Literal["robot", "simulator"]
    target_id: str
    model: str | None
    binding: str | None
    environment_id: str
    model_endpoint: str | None
    service_id: str
    service_endpoint: str


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """Deterministic host-side plan derived from one deployment config."""

    name: str
    deploy_commit: str
    environments: tuple[EnvironmentProfile, ...]
    services: tuple[ServiceSpec, ...]
    runtimes: tuple[RuntimeSpec, ...]
    device_resources: tuple[DeviceOwnerSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class DeviceOwnerSpec:
    """Host's resolved physical-resource owner and its runtime consumers."""

    resource_id: str
    node: str
    kind: str
    owner_service_id: str
    consumers: tuple[str, ...]
    external_owner: str | None = None


def build_plan(config: DeploymentConfig) -> DeploymentPlan:
    """Build a deterministic plan without contacting nodes or starting processes."""

    environments = environment_profiles(config)
    models = tuple(
        _model_service(config, model, environments)
        for model in sorted(config.models.values(), key=lambda item: item.model_id)
        if model.lifecycle == "managed"
    )
    runtimes: list[RuntimeSpec] = []
    controls: list[ServiceSpec] = []
    control_groups: dict[str, list[tuple[RuntimeSpec, ServiceSpec]]] = {}
    device_resources: dict[str, DeviceOwnerSpec] = {}
    resource_groups: dict[str, set[str]] = {}
    resource_owners: dict[str, set[str | None]] = {}
    simulations: list[ServiceSpec] = []
    for runtime in sorted(config.runtimes.values(), key=lambda item: item.runtime_id):
        if runtime.robot is not None:
            robot = config.robots[runtime.robot]
            if (
                robot.owner
                and robot.owner not in {"deploy", "control"}
                and robot.kind not in {"unitree.go2", "lerobot.xlerobot"}
            ):
                raise ServiceError(
                    f"robot {robot.robot_id!r} declares unsupported external owner "
                    f"{robot.owner!r}; only supported external proxy robot types "
                    "may declare an external owner"
                )
            _validate_sensor_nodes(config, runtime.runtime_id, runtime.inputs, robot.node)
            if runtime.model is None:
                environment = _find_environment(
                    environments,
                    node=robot.node,
                    project="deploy",
                    group=robot_environment_profile(robot.kind)[0],
                )
                runtime_spec, service = _device_control_service(
                    config,
                    runtime,
                    environment=environment,
                )
            else:
                model = config.models[runtime.model]
                assert runtime.binding is not None
                _validate_binding(runtime.binding, robot.kind, model.kind)
                model_endpoint = _endpoint(config, model, consumer_node=robot.node)
                environment = _find_environment(
                    environments,
                    node=robot.node,
                    project="deploy",
                    group=robot_environment_profile(robot.kind)[0],
                )
                runtime_spec, service = _control_service(
                    config,
                    runtime,
                    environment=environment,
                    model_endpoint=model_endpoint,
                )
            owner_resource = _robot_resource(config, robot)
            group_key = owner_resource.key
            control_groups.setdefault(group_key, []).append((runtime_spec, service))
            for resource in _control_resources(config, runtime):
                resource_groups.setdefault(resource["identity"], set()).add(group_key)
                resource_owners.setdefault(resource["identity"], set()).add(resource.get("owner"))
                existing = device_resources.get(resource["identity"])
                owner_service_id = existing.owner_service_id if existing else service.service_id
                consumers = tuple(dict.fromkeys((*(existing.consumers if existing else ()), runtime.runtime_id)))
                device_resources[resource["identity"]] = DeviceOwnerSpec(
                    resource_id=resource["identity"],
                    node=resource["node"],
                    kind=resource["kind"],
                    owner_service_id=owner_service_id,
                    consumers=consumers,
                    external_owner=resource.get("owner"),
                )
        else:
            if runtime.model is None or runtime.binding is None:
                raise ServiceError(f"simulation runtime {runtime.runtime_id!r} requires model and binding")
            simulator = config.simulators[runtime.simulator]
            definition = simulator_definition(simulator.kind)
            model = config.models[runtime.model]
            _validate_binding(runtime.binding, definition.embodiment_kind, model.kind)
            if runtime.inputs:
                raise ServiceError(
                    f"simulation runtime {runtime.runtime_id!r} receives rendered "
                    "inputs from its simulator and must not reference sensors"
                )
            model_endpoint = _endpoint(config, model, consumer_node=simulator.node)
            environment = _find_environment(
                environments,
                node=simulator.node,
                project="deploy",
                group=definition.environment_group,
            )
            runtime_spec, service = _simulation_service(
                config,
                runtime,
                environment=environment,
                model_endpoint=model_endpoint,
            )
            simulations.append(service)
        runtimes.append(runtime_spec)
    for resource_id, groups in resource_groups.items():
        owners = resource_owners.get(resource_id, set())
        if len(owners) > 1:
            raise ServiceError(f"resource {resource_id!r} has inconsistent owner declarations")
        if len(groups) > 1 and resource_id in {key for key, item in device_resources.items() if item.kind == "sensor"}:
            raise ServiceError(
                f"sensor resource {resource_id!r} is referenced by multiple robot "
                "owners; a unique sensor owner is required before startup"
            )
    for group_key in sorted(control_groups):
        pairs = control_groups[group_key]
        if len(pairs) == 1:
            controls.append(pairs[0][1])
            continue
        # One process owns one physical robot.  Profiles retain the model and
        # binding chosen by each runtime; aliases may differ, but incompatible
        # physical camera profiles are rejected before any service starts.
        first_runtime, first_service = pairs[0]
        service_id = f"control-{config.robots[config.runtimes[first_runtime.runtime_id].robot].robot_id}"
        merged = _merge_control_services(
            tuple(service for _, service in pairs),
            service_id=service_id,
            runtime_ids=tuple(item.runtime_id for item, _ in pairs),
        )
        controls.append(merged)
        for resource_id, item in tuple(device_resources.items()):
            if item.owner_service_id in {service.service_id for _, service in pairs}:
                device_resources[resource_id] = replace(
                    item,
                    owner_service_id=service_id,
                )
        for runtime_spec, _service in pairs:
            runtimes[runtimes.index(runtime_spec)] = replace(
                runtime_spec,
                service_id=service_id,
                service_endpoint=merged.endpoint,
            )
    services = (*models, *controls, *simulations)
    if len({service.service_id for service in services}) != len(services):
        raise ServiceError("model IDs must not collide with generated runtime service IDs")
    wireless = _wireless_endpoint_configs(config)
    return DeploymentPlan(
        name=config.metadata.name,
        deploy_commit=config.metadata.deploy_commit,
        environments=environments,
        services=tuple(replace(service, wireless_config_json=wireless.get(service.service_id)) for service in services),
        runtimes=tuple(runtimes),
        device_resources=tuple(device_resources[key] for key in sorted(device_resources)),
    )


def _model_service(
    config: DeploymentConfig,
    model: ModelConfig,
    environments: tuple[EnvironmentProfile, ...],
) -> ServiceSpec:
    try:
        descriptor = provider(model.backend)
    except ValueError as error:
        raise ServiceError(str(error)) from error
    if descriptor.managed_command is None or not descriptor.supports(model.transport):
        raise ServiceError(f"model {model.model_id!r} has no managed {model.backend}/{model.transport} service")
    if model.node is None or model.server is None:
        raise ServiceError(f"managed model {model.model_id!r} is missing node/server")
    environment_group = descriptor.environment_group or model.kind
    environment = _find_environment(
        environments,
        node=model.node,
        project=descriptor.source_project,
        group=environment_group,
        path=model.environment or f".venv-{model.backend}-{model.kind}",
    )
    source = _option_string(model, "source", required=descriptor.requires_checkpoint) or ""
    adapter_config_json = _binding_adapter_config(config, model)
    configured_adapter = _option_string(model, "adapter_config")
    gpu = _option_string(model, "gpu")
    if descriptor.adapter_config_owned and adapter_config_json is not None and configured_adapter is not None:
        raise ServiceError(f"models.{model.model_id}.adapter_config conflicts with binding configuration")
    options = ManagedCommandOptions(
        checkpoint=source,
        bind=model.server.bind,
        port=model.server.port,
        transport=model.transport,
        policy=model.kind,
        device=gpu,
        adapter_config=configured_adapter,
        comm_config=f"{model.model_id}.wireless.json",
        executable=(_option_string(model, "server_executable") or descriptor.default_executable),
        pipeline=_option_string(model, "pipeline"),
        pipeline_config=_option_string(model, "pipeline_config"),
        extra_args=_option_strings(model, "server_args"),
    )
    try:
        argv = descriptor.managed_command(options)
    except (TypeError, ValueError) as error:
        raise ServiceError(f"model {model.model_id!r} has invalid managed options: {error}") from error
    if descriptor.append_server_args:
        argv = (*argv, *options.extra_args)
    endpoint = _endpoint(config, model, consumer_node=model.node)
    command_environment = (
        descriptor.environment_builder(model.options) if descriptor.environment_builder is not None else {}
    )
    return ServiceSpec(
        service_id=model.model_id,
        kind="model",
        node=model.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=(
            f"{endpoint}{descriptor.health_suffix}" if descriptor.health_suffix and model.transport == "http" else None
        ),
        command=Command(argv, environment=command_environment),
        adapter_config_json=adapter_config_json if descriptor.adapter_config_owned else None,
    )


def _device_control_service(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    *,
    environment: EnvironmentProfile,
) -> tuple[RuntimeSpec, ServiceSpec]:
    """Plan a robot lifecycle service without inventing a model peer.

    A device-only runtime still has a normal control process and loopback
    endpoint.  Its generated contract carries ``inference.enabled=false``;
    Host therefore can start, describe, and observe the robot while task
    execution remains explicitly unavailable until a model runtime is added.
    """

    if runtime.robot is None or runtime.model is not None or runtime.binding is not None:
        raise ServiceError(f"runtime {runtime.runtime_id!r} is not a robot-only runtime")
    robot = config.robots[runtime.robot]
    endpoint = f"http://{_url_host(runtime.server.bind)}:{runtime.server.port}"
    inputs = tuple(
        SensorInput(
            sensor_id=sensor_id,
            name=input_name,
            kind=config.sensors[sensor_id].kind,
            options=config.sensors[sensor_id].options,
        )
        for input_name, sensor_id in runtime.inputs.items()
    )
    control_config = ControlServiceConfig.device_only(
        runtime_id=runtime.runtime_id,
        bind=runtime.server.bind,
        port=runtime.server.port,
        robot_id=robot.robot_id,
        robot_kind=robot.kind,
        robot_options=robot.options,
        node_id=robot.node,
        inputs=inputs,
        device_resources=tuple(_control_resources(config, runtime)),
    )
    runtime_spec = RuntimeSpec(
        runtime_id=runtime.runtime_id,
        node=robot.node,
        target_kind="robot",
        target_id=robot.robot_id,
        model=None,
        binding=None,
        environment_id=environment.environment_id,
        model_endpoint=None,
        service_id=f"control-{runtime.runtime_id}",
        service_endpoint=endpoint,
    )
    service = ServiceSpec(
        service_id=runtime_spec.service_id,
        kind="control",
        node=robot.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=f"{endpoint}/healthz",
        command=Command(("embodirun-control-serve",)),
        control_config_json=control_config.to_json(),
        owner_id=f"control:{robot.robot_id}",
        resource_ids=tuple(item["identity"] for item in _control_resources(config, runtime)),
        runtime_ids=(runtime.runtime_id,),
    )
    return runtime_spec, service


def _control_service(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    *,
    environment: EnvironmentProfile,
    model_endpoint: str,
) -> tuple[RuntimeSpec, ServiceSpec]:
    robot = config.robots[runtime.robot]
    model = config.models[runtime.model]
    service_id = f"control-{runtime.runtime_id}"
    endpoint = f"http://{_url_host(runtime.server.bind)}:{runtime.server.port}"
    inference_options = _inference_options(config, runtime, model)
    try:
        control_config = ControlServiceConfig(
            runtime_id=runtime.runtime_id,
            binding_kind=runtime.binding,
            bind=runtime.server.bind,
            port=runtime.server.port,
            inference_backend=model.backend,
            inference_transport=model.transport,
            inference_endpoint=model_endpoint,
            inference_options=inference_options,
            robot_id=robot.robot_id,
            robot_kind=robot.kind,
            robot_options=robot.options,
            inputs=tuple(
                SensorInput(
                    sensor_id=sensor_id,
                    name=input_name,
                    kind=config.sensors[sensor_id].kind,
                    options=config.sensors[sensor_id].options,
                )
                for input_name, sensor_id in runtime.inputs.items()
            ),
            runtime_options=runtime.options,
            node_id=robot.node,
            device_resources=tuple(_control_resources(config, runtime)),
        )
        control_config_json = control_config.to_json()
    except (TypeError, ValueError) as error:
        raise ServiceError(f"runtime {runtime.runtime_id!r} control configuration is invalid: {error}") from error
    runtime_spec = RuntimeSpec(
        runtime_id=runtime.runtime_id,
        node=robot.node,
        target_kind="robot",
        target_id=robot.robot_id,
        model=runtime.model,
        binding=runtime.binding,
        environment_id=environment.environment_id,
        model_endpoint=model_endpoint,
        service_id=service_id,
        service_endpoint=endpoint,
    )
    service = ServiceSpec(
        service_id=service_id,
        kind="control",
        node=robot.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=f"{endpoint}/healthz",
        command=Command(("embodirun-control-serve",)),
        control_config_json=control_config_json,
        owner_id=f"control:{robot.robot_id}",
        resource_ids=tuple(item["identity"] for item in _control_resources(config, runtime)),
        runtime_ids=(runtime.runtime_id,),
    )
    return runtime_spec, service


def _simulation_service(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    *,
    environment: EnvironmentProfile,
    model_endpoint: str,
) -> tuple[RuntimeSpec, ServiceSpec]:
    simulator = config.simulators[runtime.simulator]
    model = config.models[runtime.model]
    service_id = f"simulation-{runtime.runtime_id}"
    endpoint = f"http://{_url_host(runtime.server.bind)}:{runtime.server.port}"
    inference_options = _inference_options(config, runtime, model)
    try:
        service_config = SimulationServiceConfig(
            runtime_id=runtime.runtime_id,
            binding_kind=runtime.binding,
            bind=runtime.server.bind,
            port=runtime.server.port,
            inference_backend=model.backend,
            inference_transport=model.transport,
            inference_endpoint=model_endpoint,
            inference_options=inference_options,
            simulator_id=simulator.simulator_id,
            simulator_kind=simulator.kind,
            simulator_options=simulator.options,
        )
    except (TypeError, ValueError) as error:
        raise ServiceError(f"runtime {runtime.runtime_id!r} simulation configuration is invalid: {error}") from error
    runtime_spec = RuntimeSpec(
        runtime_id=runtime.runtime_id,
        node=simulator.node,
        target_kind="simulator",
        target_id=simulator.simulator_id,
        model=runtime.model,
        binding=runtime.binding,
        environment_id=environment.environment_id,
        model_endpoint=model_endpoint,
        service_id=service_id,
        service_endpoint=endpoint,
    )
    service = ServiceSpec(
        service_id=service_id,
        kind="simulation",
        node=simulator.node,
        environment_id=environment.environment_id,
        endpoint=endpoint,
        health_endpoint=f"{endpoint}/healthz",
        command=Command(("embodirun-simulation-serve",)),
        simulation_config_json=service_config.to_json(),
    )
    return runtime_spec, service


def _inference_options(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
    model: ModelConfig,
) -> dict[str, object]:
    options: dict[str, object] = {}
    token = _option_string(model, "token")
    if token is not None:
        options["token"] = token
    if model.transport == "wireless":
        prefix = "control" if runtime.robot is not None else "simulation"
        options.update(
            {
                "comm_config": f"{prefix}-{runtime.runtime_id}.wireless.json",
                "server_node_id": _wireless_model_peer_id(model),
            }
        )
    definition = _load_binding(runtime.binding)
    image_fields = _runtime_image_fields(config, runtime)
    descriptor = provider(model.backend)
    try:
        options.update(
            descriptor.build_options(
                ProviderOptionsContext(
                    model_id=model.model_id,
                    runtime_id=runtime.runtime_id,
                    transport=model.transport,
                    model_options=model.options,
                    binding_options=definition.adapter_config or {},
                    image_fields=image_fields,
                    maximum_chunk_steps=definition.maximum_chunk_steps,
                )
            )
        )
    except (TypeError, ValueError) as error:
        raise ServiceError(f"models.{model.model_id} provider options are invalid: {error}") from error
    return options


def _binding_adapter_config(
    config: DeploymentConfig,
    model: ModelConfig,
) -> str | None:
    policy_kwargs = _option_mapping(model, "policy_kwargs")
    if policy_kwargs is not None:
        if model.backend != "vvla":
            raise ServiceError(f"models.{model.model_id}.policy_kwargs requires VVLA")
        if any(not isinstance(key, str) or not key.strip() for key in policy_kwargs):
            raise ServiceError(f"models.{model.model_id}.policy_kwargs keys must be non-empty strings")
    runtime_ids: list[str] = []
    generated: set[str | None] = set()
    for runtime in config.runtimes.values():
        if runtime.model != model.model_id:
            continue
        runtime_ids.append(runtime.runtime_id)
        definition = _load_binding(runtime.binding)
        if definition.adapter_config is None:
            generated.add(None)
            continue
        adapter_config = dict(definition.adapter_config)
        if policy_kwargs is not None:
            adapter_config["policy_kwargs"] = dict(policy_kwargs)
        image_fields = _runtime_image_fields(config, runtime)
        if image_fields:
            adapter_config["image_fields"] = image_fields
        if provider(model.backend).adapter_config_owned:
            image_keys = _configured_image_keys(
                model,
                image_fields,
                runtime_id=runtime.runtime_id,
            )
            if image_keys is not None:
                resolved_image_keys = {field: image_keys.get(field, field) for field in image_fields}
                if len(resolved_image_keys.values()) != len(set(resolved_image_keys.values())):
                    raise ServiceError(f"models.{model.model_id}.image_keys values must be unique")
                adapter_config["image_keys"] = image_keys
        try:
            generated.add(
                json.dumps(
                    {
                        **adapter_config,
                        "return_steps": definition.maximum_chunk_steps,
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ServiceError(f"binding {runtime.binding!r} adapter configuration is not JSON") from error
    if len(generated) > 1:
        names = ", ".join(sorted(runtime_ids))
        raise ServiceError(
            f"model {model.model_id!r} is shared by runtimes with incompatible adapter configurations: {names}"
        )
    result = next(iter(generated), None)
    if policy_kwargs is not None and result is None:
        raise ServiceError(
            f"models.{model.model_id}.policy_kwargs requires a binding-generated "
            "adapter configuration; otherwise put policy_kwargs in the adapter_config JSON"
        )
    return result


def _runtime_image_fields(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
) -> tuple[str, ...]:
    if runtime.robot is not None:
        return tuple(runtime.inputs)
    simulator = config.simulators[runtime.simulator]
    return simulator_definition(simulator.kind).image_fields


def _wireless_model_peer_id(model: ModelConfig) -> str:
    return f"model.{model.model_id}"


def _runtime_node(config: DeploymentConfig, runtime: RuntimeConfig) -> str:
    if runtime.robot is not None:
        return config.robots[runtime.robot].node
    return config.simulators[runtime.simulator].node


def _wireless_endpoint_configs(config: DeploymentConfig) -> dict[str, str]:
    """Map service IDs to JSON (also valid YAML) understood by WirelessComm.

    A model knows all its runtimes; each runtime knows only its selected model.
    Peer IDs identify processes, not physical machines.
    """

    result: dict[str, str] = {}
    for model in config.models.values():
        if model.lifecycle == "external" or model.transport != "wireless":
            continue
        runtimes = sorted(
            (runtime for runtime in config.runtimes.values() if runtime.model == model.model_id),
            key=lambda runtime: runtime.runtime_id,
        )
        remote = any(_runtime_node(config, runtime) != model.node for runtime in runtimes)
        server = {
            "node_id": _wireless_model_peer_id(model),
            "host": service_host(config.nodes[model.node], model.server.bind, remote=remote),
            "port": model.server.port,
        }
        clients = []
        for runtime in runtimes:
            client = runtime.inference_client
            assert client is not None
            node = _runtime_node(config, runtime)
            peer = {
                "node_id": f"runtime.{runtime.runtime_id}",
                "host": service_host(config.nodes[node], client.server.bind, remote=node != model.node),
                "port": client.server.port,
            }
            clients.append(peer)
            prefix = "control" if runtime.robot is not None else "simulation"
            result[f"{prefix}-{runtime.runtime_id}"] = json.dumps(
                {
                    "local": {**peer, "bind_host": client.server.bind},
                    "peers": [server],
                    "comm": client.transport_options,
                },
                allow_nan=False,
                sort_keys=True,
            )
        result[model.model_id] = json.dumps(
            {
                "local": {**server, "bind_host": model.server.bind},
                "peers": clients,
                "comm": model.transport_options,
            },
            allow_nan=False,
            sort_keys=True,
        )
    return result


def _endpoint(
    config: DeploymentConfig,
    model: ModelConfig,
    *,
    consumer_node: str,
) -> str:
    if model.lifecycle == "external":
        assert model.endpoint is not None
        return model.endpoint
    if model.transport == "wireless":
        return f"wireless://{_wireless_model_peer_id(model)}"
    host = service_host(config.nodes[model.node], model.server.bind, remote=consumer_node != model.node)
    return f"http://{_url_host(host)}:{model.server.port}"


def _url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _find_environment(
    profiles: tuple[EnvironmentProfile, ...],
    *,
    node: str,
    project: str,
    group: str,
    path: str | None = None,
) -> EnvironmentProfile:
    matches = [
        profile
        for profile in profiles
        if profile.node == node
        and profile.project == project
        and profile.group == group
        and (path is None or profile.path == path)
    ]
    if len(matches) != 1:
        raise ServiceError(f"expected one {project}/{group} environment on node {node!r}, found {len(matches)}")
    return matches[0]


def _validate_binding(binding: str, robot_kind: str, model_kind: str) -> None:
    definition = _load_binding(binding)
    if definition.robot_kind != robot_kind or definition.model_kind != model_kind:
        raise ServiceError(f"binding {binding!r} does not match robot {robot_kind!r} and model {model_kind!r}")


def _load_binding(binding: str) -> BindingDefinition:
    try:
        definition = binding_definition(binding)
    except (KeyError, TypeError):
        raise ServiceError(f"binding {binding!r} is not available") from None
    return definition


def _validate_sensor_nodes(
    config: DeploymentConfig,
    runtime_id: str,
    inputs: dict[str, str],
    runtime_node: str,
) -> None:
    for input_name, sensor_id in inputs.items():
        sensor = config.sensors[sensor_id]
        if sensor.node != runtime_node:
            raise ServiceError(
                f"runtime {runtime_id!r} input {input_name!r} uses sensor "
                f"{sensor_id!r} on node {sensor.node!r}; cross-node sensor inputs "
                "are not supported yet"
            )


def _robot_resource(config: DeploymentConfig, robot) -> ResourceIdentity:
    """Resolve one robot's physical connection without opening it."""

    source = robot.resource
    if source is None:
        if robot.kind == "lerobot.so101":
            source = robot.port or robot.options.get("port")
        elif robot.kind == "arx.x5":
            # ARX5Config's normal driver default is the local ``can1`` bus;
            # keep the identity explicit even when legacy YAML omitted it.
            source = robot.options.get("can_port", "can1")
        elif robot.kind == "franka.fr3":
            source = robot.options.get("host")
        elif robot.kind == "unitree.go2":
            source = robot.options.get("control_url")
        else:
            source = robot.port or robot.options.get("port")
    if robot.kind == "lerobot.bi_so101":
        ports = robot_ports(robot)
        if len(ports) != 2:
            raise ServiceError(f"robot {robot.robot_id!r} must identify both SO-101 ports")
        port_identity = "|".join(ports)
        if isinstance(source, str) and source.strip():
            source = f"{source}|ports={port_identity}"
        else:
            source = f"ports={port_identity}"
    if not isinstance(source, str) or not source.strip():
        raise ServiceError(
            f"robot {robot.robot_id!r} has no explicit physical connection identity; set robots.<id>.resource"
        )
    return canonical_resource_identity(
        robot.node,
        "robot",
        source,
        resolve_path=False,
    )


def _sensor_resource(config: DeploymentConfig, sensor) -> ResourceIdentity:
    """Resolve camera/bus identity from explicit config or legacy device path."""

    source = sensor.resource
    if source is None:
        source = (
            sensor.options.get("device")
            or sensor.options.get("path")
            or sensor.options.get("serial")
            or sensor.options.get("serial_number")
        )
    if source is None:
        # Preserve custom logical sensors, whose adapter may own an endpoint
        # rather than a path; physical camera/bus kinds must be explicit.
        if any(token in sensor.kind.lower() for token in ("camera", "v4l", "realsense", "serial", "can")):
            raise ServiceError(
                f"sensor {sensor.sensor_id!r} has no physical identity; "
                "set sensors.<id>.resource or a driver device/serial field"
            )
        source = sensor.sensor_id
    return canonical_resource_identity(sensor.node, "sensor", source, resolve_path=False)


def _control_resources(
    config: DeploymentConfig,
    runtime: RuntimeConfig,
) -> list[dict[str, object]]:
    robot = config.robots[runtime.robot]
    robot_identity = _robot_resource(config, robot)
    resources: list[dict[str, object]] = [
        {
            "identity": robot_identity.key,
            "node": robot_identity.node,
            "kind": robot_identity.kind,
            "value": robot_identity.value,
            "owner": robot.owner,
            "external_owner": bool(robot.owner and robot.owner not in {"deploy", "control"}),
        }
    ]
    seen = {robot_identity.key}
    # The pair identity above groups one BiSO101 adapter, while each bus also
    # needs the ordinary robot lock used by a single SO-101.  Keeping those
    # component identities in the same resource list prevents an unrelated
    # single-arm runtime from sharing one side of the pair.
    for port in robot_ports(robot):
        identity = canonical_resource_identity(
            robot.node,
            "robot",
            port,
            resolve_path=False,
        )
        if identity.key in seen:
            continue
        seen.add(identity.key)
        resources.append(
            {
                "identity": identity.key,
                "node": identity.node,
                "kind": identity.kind,
                "value": identity.value,
                "owner": robot.owner,
                "external_owner": bool(robot.owner and robot.owner not in {"deploy", "control"}),
            }
        )
    for sensor_id in runtime.inputs.values():
        sensor = config.sensors[sensor_id]
        if sensor.owner and sensor.owner not in {"deploy", "control"}:
            raise ServiceError(
                f"sensor {sensor.sensor_id!r} declares unsupported external owner "
                f"{sensor.owner!r}; a camera proxy is not available"
            )
        identity = _sensor_resource(config, sensor)
        if identity.key in seen:
            # One physical source may have two logical input aliases.  Keep
            # those IDs in generated metadata so Control can match the
            # selected SensorInput instead of falling back to the first
            # resource declaration.
            existing = next(item for item in resources if item["identity"] == identity.key)
            if existing.get("kind") == "sensor":
                sensor_ids = existing.setdefault("sensor_ids", [existing.get("sensor_id")])
                if isinstance(sensor_ids, list) and sensor.sensor_id not in sensor_ids:
                    sensor_ids.append(sensor.sensor_id)
            continue
        seen.add(identity.key)
        resources.append(
            {
                "identity": identity.key,
                "node": identity.node,
                "kind": identity.kind,
                "value": identity.value,
                "sensor_id": sensor.sensor_id,
                "sensor_ids": [sensor.sensor_id],
                "owner": sensor.owner,
                "external_owner": bool(sensor.owner and sensor.owner not in {"deploy", "control"}),
            }
        )
    return resources


def _control_resource_ids(config: DeploymentConfig, runtime: RuntimeConfig) -> tuple[str, ...]:
    return tuple(item["identity"] for item in _control_resources(config, runtime))


def _merge_control_services(
    services: tuple[ServiceSpec, ...],
    *,
    service_id: str,
    runtime_ids: tuple[str, ...],
) -> ServiceSpec:
    if not services:
        raise ServiceError("cannot merge an empty control service group")
    configs = [ControlServiceConfig.from_json(service.control_config_json or "") for service in services]
    base = configs[0]
    if any(config.inference_enabled != base.inference_enabled for config in configs):
        raise ServiceError(
            "one physical robot cannot merge device-only and model runtimes; use a single configured lifecycle mode"
        )
    if not base.inference_enabled:
        raise ServiceError("multiple device-only runtimes for one robot are unsupported; use one robot-only runtime")
    if any(config.inference_transport == "wireless" for config in configs):
        raise ServiceError(
            "multiple wireless runtimes cannot share one control owner yet; use one runtime or separate physical owners"
        )
    profiles: dict[str, ControlRuntimeProfile] = {}
    resources: dict[str, Mapping[str, object]] = {}
    for config in configs:
        profile = config.profile_for_runtime(config.runtime_id)
        profiles[profile.runtime_id] = profile
        for item in config.device_resources:
            identity = str(item["identity"])
            previous = resources.get(identity)
            if previous is not None:
                # Names can differ per runtime, but physical profile and owner
                # declarations must agree before a single process is selected.
                if (
                    previous.get("kind") != item.get("kind")
                    or previous.get("value") != item.get("value")
                    or previous.get("owner") != item.get("owner")
                ):
                    raise ServiceError(f"shared resource {identity!r} has incompatible profiles")
                # Preserve logical aliases when two runtime profiles refer to
                # the same physical camera identity.
                merged_item = dict(previous)
                previous_ids = previous.get("sensor_ids", [previous.get("sensor_id")])
                incoming_ids = item.get("sensor_ids", [item.get("sensor_id")])
                aliases = [value for value in (*previous_ids, *incoming_ids) if isinstance(value, str) and value]
                merged_item["sensor_ids"] = list(dict.fromkeys(aliases))
                item = merged_item
            resources[identity] = item
    if len({config.robot_kind for config in configs}) != 1 or len({config.robot_id for config in configs}) != 1:
        raise ServiceError("runtimes sharing one physical robot must use one compatible robot owner")
    merged = replace(
        base,
        runtime_profiles=profiles,
        device_resources=tuple(resources[key] for key in sorted(resources)),
    )
    return replace(
        services[0],
        service_id=service_id,
        control_config_json=merged.to_json(),
        owner_id=services[0].owner_id,
        resource_ids=tuple(resources),
        runtime_ids=runtime_ids,
    )


def _option_string(
    model: ModelConfig,
    name: str,
    *,
    required: bool = False,
) -> str | None:
    value = model.options.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        message = "must be a non-empty string" if required else "must be a non-empty string when provided"
        raise ServiceError(f"models.{model.model_id}.{name} {message}")
    return value


def _option_strings(model: ModelConfig, name: str) -> tuple[str, ...]:
    value = model.options.get(name, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ServiceError(f"models.{model.model_id}.{name} must be a list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ServiceError(f"models.{model.model_id}.{name} must contain non-empty strings")
    reserved = (
        "--host",
        "--port",
        "--model-type",
        "--pipeline",
        "--pipeline-class-name",
        "--pipeline-config-path",
    )
    if any(item == option or item.startswith(f"{option}=") for item in result for option in reserved):
        raise ServiceError(
            f"models.{model.model_id}.{name} must not override host, port, model type, or explicit pipeline settings"
        )
    return result


def _option_mapping(
    model: ModelConfig,
    name: str,
) -> dict[str, object] | None:
    value = model.options.get(name)
    if value is None:
        return None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ServiceError(f"models.{model.model_id}.{name} must be an object")
    return dict(value)


def _configured_image_keys(
    model: ModelConfig,
    image_fields: tuple[str, ...],
    *,
    runtime_id: str,
) -> dict[str, str] | None:
    value = _option_mapping(model, "image_keys")
    if value is None:
        return None
    if any(not isinstance(target, str) or not target.strip() for target in value.values()):
        raise ServiceError(f"models.{model.model_id}.image_keys values must be non-empty strings")
    unknown = sorted(set(value) - set(image_fields))
    if unknown:
        raise ServiceError(
            f"models.{model.model_id}.image_keys contains fields not produced "
            f"by runtime {runtime_id!r}: {', '.join(unknown)}"
        )
    return {source: target for source, target in value.items() if isinstance(target, str)}


__all__ = [
    "DeploymentPlan",
    "RuntimeSpec",
    "ServiceError",
    "ServiceSpec",
    "build_plan",
]
