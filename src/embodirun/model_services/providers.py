"""Inference provider registry shared by clients and deployment validation.

Providers own protocol/client capability metadata.  Host and Control consume this
small registry instead of maintaining another backend enum.  Registration is
process-local and explicit; package discovery is intentionally out of scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .client import InferenceClient

ClientBuilder = Callable[[str, Mapping[str, Any], float], InferenceClient]


@dataclass(frozen=True, slots=True)
class ProviderOptionsContext:
    """Context used by a provider to derive runtime client options.

    ``model_options`` contains user supplied model keys, while
    ``binding_options`` contains the binding's model-to-device mapping.  The
    Host supplies rendered image fields and the binding chunk limit so a
    provider can validate its own options without importing Host internals.
    """

    model_id: str
    runtime_id: str
    transport: str
    model_options: Mapping[str, Any]
    binding_options: Mapping[str, Any]
    image_fields: tuple[str, ...]
    maximum_chunk_steps: int


ProviderOptionsBuilder = Callable[[ProviderOptionsContext], Mapping[str, Any]]
EnvironmentPackagesBuilder = Callable[[Mapping[str, Any]], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ManagedCommandOptions:
    """Typed inputs a managed provider may use to build its server command.

    ``checkpoint`` is a provider resource path, ``bind``/``port`` describe the
    service listener, and the remaining values are optional policy, device,
    adapter, pipeline, and provider-specific command arguments.  Providers
    that do not use one of these values may ignore it or reject it explicitly.
    """

    checkpoint: str
    bind: str
    port: int
    transport: str
    policy: str
    device: str | None
    adapter_config: str | None
    comm_config: str
    executable: str
    pipeline: str | None
    pipeline_config: str | None
    extra_args: tuple[str, ...]


ManagedCommandBuilder = Callable[[ManagedCommandOptions], tuple[str, ...]]
EnvironmentBuilder = Callable[[Mapping[str, Any]], dict[str, str]]


@dataclass(frozen=True, slots=True)
class InferenceProvider:
    """Public capability and lifecycle descriptor for one inference provider.

    The descriptor is the single source of truth consumed by client creation,
    configuration validation, environment preparation, and managed service
    planning.  ``external`` services use only client capabilities; managed
    services additionally require ``managed_command`` and their declared
    environment metadata.
    """

    name: str
    transports: frozenset[str]
    action_capable: bool
    client_builder: ClientBuilder
    environment_group: str | None = None
    requires_environment_packages: bool = False
    managed_command: ManagedCommandBuilder | None = None
    health_suffix: str | None = None
    adapter_config_owned: bool = False
    append_server_args: bool = False
    environment_builder: EnvironmentBuilder | None = None
    source_project: Literal["deploy", "inference"] = "inference"
    requires_source_checkout: bool = False
    requires_checkpoint: bool = True
    environment_packages_builder: EnvironmentPackagesBuilder | None = None
    allowed_options: frozenset[str] = frozenset()
    options_builder: ProviderOptionsBuilder | None = None
    default_python: str | None = None
    default_executable: str = ""
    source_repository: str | None = None
    source_submodule_path: str | None = None

    def supports(self, transport: str) -> bool:
        """Return whether the provider advertises ``transport`` capability.

        This is a pure metadata check used by configuration validation and
        client construction; unsupported values return ``False`` so callers
        can report the provider/transport pair before any process starts.
        """

        return transport in self.transports

    def environment_packages(self, options: Mapping[str, Any]) -> tuple[str, ...]:
        """Render optional provider-owned package arguments for managed setup.

        Providers without package metadata return an empty tuple.  A provider
        callback that returns non-string or empty arguments raises
        ``ValueError`` so malformed environment metadata cannot reach ``uv``.
        """

        if self.environment_packages_builder is None:
            return ()
        packages = self.environment_packages_builder(options)
        if isinstance(packages, (str, bytes)) or any(
            not isinstance(item, str) or not item.strip() for item in packages
        ):
            raise ValueError(f"provider {self.name!r} returned invalid environment package arguments")
        return tuple(packages)

    def build_options(self, context: ProviderOptionsContext) -> dict[str, Any]:
        """Build optional client metadata from the typed Host context.

        The default is an empty mapping for providers with no extra options.
        Provider callbacks may validate their own capability fields and raise
        ``ValueError`` before Control receives a malformed client contract.
        """

        if self.options_builder is None:
            return {}
        return dict(self.options_builder(context))


_PROVIDERS: dict[str, InferenceProvider] = {}


def register_provider(provider: InferenceProvider) -> None:
    """Register one process-local provider descriptor exactly once."""

    if not provider.name.strip():
        raise ValueError("provider name must not be empty")
    if not provider.transports:
        raise ValueError(f"provider {provider.name!r} must support a transport")
    if provider.name in _PROVIDERS:
        raise ValueError(f"inference provider {provider.name!r} is already registered")
    _PROVIDERS[provider.name] = provider


def provider(name: str) -> InferenceProvider:
    """Resolve a provider name or raise a local configuration error."""

    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unsupported inference provider {name!r}") from None


def providers() -> tuple[InferenceProvider, ...]:
    """Return registered providers in deterministic name order."""

    return tuple(_PROVIDERS[name] for name in sorted(_PROVIDERS))


def _register_builtins() -> None:
    if _PROVIDERS:
        return
    from .backends.embodiinfer import EmbodiInferHttpClient, EmbodiInferWirelessClient
    from .backends.sglang import SglangHttpClient

    def embodiinfer_builder(endpoint: str, options: Mapping[str, Any], timeout_s: float) -> InferenceClient:
        token = options.get("token")
        if token is not None and not isinstance(token, str):
            raise ValueError("inference token must be a string")
        if options.get("transport", "http") == "wireless":
            return EmbodiInferWirelessClient.from_config(
                _required_string(options, "comm_config"),
                server_node_id=_required_string(options, "server_node_id"),
                token=token,
                timeout_s=timeout_s,
            )
        return EmbodiInferHttpClient(endpoint, token=token, timeout_s=timeout_s)

    def sglang_builder(endpoint: str, options: Mapping[str, Any], timeout_s: float) -> InferenceClient:
        token = options.get("token")
        if token is not None and not isinstance(token, str):
            raise ValueError("inference token must be a string")
        return SglangHttpClient(
            endpoint,
            token=token,
            timeout_s=timeout_s,
            image_keys=_mapping(options, "image_keys"),
            state_fields=_strings(options, "state_fields"),
            action_feature_names=_strings(options, "action_feature_names"),
            output_action_dim=_positive_int(options, "output_action_dim"),
            parameters=_mapping(options, "parameters"),
            runtime=_mapping(options, "runtime"),
        )

    def embodiinfer_command(options: ManagedCommandOptions) -> tuple[str, ...]:
        from .backends.embodiinfer import (
            embodiinfer_http_server_command,
            embodiinfer_wireless_server_command,
        )

        common = {
            "policy": options.policy,
            "checkpoint": options.checkpoint,
            "device": options.device,
            "adapter_config": options.adapter_config,
        }
        if options.transport == "http":
            return embodiinfer_http_server_command(bind=options.bind, port=options.port, **common)
        return embodiinfer_wireless_server_command(comm_config=options.comm_config, **common)

    def sglang_command(options: ManagedCommandOptions) -> tuple[str, ...]:
        from .backends.sglang import sglang_server_command

        return sglang_server_command(
            checkpoint=options.checkpoint,
            bind=options.bind,
            port=options.port,
            executable=options.executable,
            pipeline=options.pipeline,
            pipeline_config=options.pipeline_config,
            extra_args=options.extra_args,
        )

    def sglang_environment(options: Mapping[str, Any]) -> dict[str, str]:
        gpu = options.get("gpu")
        if gpu is None:
            return {}
        if not isinstance(gpu, str) or not gpu.startswith("cuda:"):
            raise ValueError("gpu must use cuda:<device> for SGLang")
        devices = gpu.removeprefix("cuda:").split(",")
        if not devices or any(not item.isdigit() for item in devices):
            raise ValueError("gpu must use cuda:<device> for SGLang")
        return {"CUDA_VISIBLE_DEVICES": ",".join(devices)}

    def sglang_packages(_options: Mapping[str, Any]) -> tuple[str, ...]:
        # The SGLang integration imports the Deploy HTTP client and therefore
        # installs this checkout explicitly.  Keeping both paths here makes
        # ``uv pip install`` independent of PyPI's Deploy release and avoids
        # dragging EmbodiInfer or its optional engine into the environment.
        # SGLang keeps its isolated environment under ``sources/inference``;
        # these paths deliberately point at the active EmbodiRun checkout beside
        # it, rather than resolving a released ``embodirun`` release from PyPI.
        return ("-e", "../deploy", "-e", "../deploy/integrations/sglang_pi05")

    def sglang_options(context: ProviderOptionsContext) -> Mapping[str, Any]:
        model_options = context.model_options
        binding = context.binding_options
        options: dict[str, Any] = {}
        for name in ("state_fields", "action_feature_names"):
            value = binding.get(name)
            if value is not None:
                options[name] = list(value)
        image_keys = {field: field.rsplit(".", 1)[-1] for field in context.image_fields}
        configured = model_options.get("image_keys")
        if configured is not None:
            if not isinstance(configured, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str) or not value.strip()
                for key, value in configured.items()
            ):
                raise ValueError("image_keys must map non-empty field names to strings")
            unknown = sorted(set(configured) - set(context.image_fields))
            if unknown:
                raise ValueError("image_keys contains fields not produced by the runtime: " + ", ".join(unknown))
            image_keys.update(configured)
        if len(image_keys.values()) != len(set(image_keys.values())):
            raise ValueError("image_keys values must be unique")
        options["image_keys"] = image_keys
        parameters = model_options.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("parameters must be an object")  # noqa: TRY004
        parameters = dict(parameters)
        parameters.setdefault("action_horizon", context.maximum_chunk_steps)
        action_horizon = parameters["action_horizon"]
        if (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or not 1 <= action_horizon <= context.maximum_chunk_steps
        ):
            raise ValueError("parameters.action_horizon must be between 1 and the binding chunk limit")
        options["parameters"] = parameters
        runtime = model_options.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise ValueError("runtime must be an object")  # noqa: TRY004
        options["runtime"] = dict(runtime)
        output_action_dim = model_options.get("output_action_dim")
        if output_action_dim is not None:
            if isinstance(output_action_dim, bool) or not isinstance(output_action_dim, int) or output_action_dim <= 0:
                raise ValueError("output_action_dim must be a positive integer")
            options["output_action_dim"] = output_action_dim
        return options

    register_provider(
        InferenceProvider(
            "embodiinfer",
            frozenset({"http", "wireless"}),
            True,
            embodiinfer_builder,
            environment_group=None,
            managed_command=embodiinfer_command,
            health_suffix="/healthz",
            adapter_config_owned=True,
            append_server_args=True,
            requires_source_checkout=True,
            source_repository="https://github.com/BUAA-CI-LAB/EmbodiInfer.git",
            source_submodule_path="third_party/embodiinfer",
            allowed_options=frozenset(
                {
                    "source",
                    "gpu",
                    "adapter_config",
                    "server_args",
                    "token",
                    "image_keys",
                    "policy_kwargs",
                }
            ),
        )
    )
    register_provider(
        InferenceProvider(
            "sglang",
            frozenset({"http"}),
            True,
            sglang_builder,
            environment_group="sglang",
            requires_environment_packages=True,
            managed_command=sglang_command,
            health_suffix="/health",
            source_project="inference",
            append_server_args=False,
            environment_builder=sglang_environment,
            environment_packages_builder=sglang_packages,
            allowed_options=frozenset(
                {
                    "source",
                    "gpu",
                    "server_executable",
                    "pipeline",
                    "pipeline_config",
                    "server_args",
                    "token",
                    "image_keys",
                    "output_action_dim",
                    "parameters",
                    "runtime",
                }
            ),
            options_builder=sglang_options,
            default_python="3.12",
            default_executable="sglang",
        )
    )


def _required_string(options: Mapping[str, Any], name: str) -> str:
    value = options.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"inference option {name!r} must be non-empty")
    return value


def _mapping(options: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = options.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004
            f"inference option {name!r} must be an object"
        )
    return dict(value)


def _strings(options: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = options.get(name, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"inference option {name!r} must be a list")  # noqa: TRY004
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"inference option {name!r} must contain non-empty strings")
    return tuple(value)


def _positive_int(options: Mapping[str, Any], name: str) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"inference option {name!r} must be a positive integer")
    return value


_register_builtins()

__all__ = [
    "InferenceProvider",
    "ManagedCommandOptions",
    "ProviderOptionsContext",
    "provider",
    "providers",
    "register_provider",
]
