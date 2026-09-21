"""Inference service contracts and clients organized by backend and protocol."""

from .backends.sglang import SglangHttpClient, SglangHttpError
from .backends.vvla import (
    VvlaHttpClient,
    VvlaHttpError,
    VvlaWirelessClient,
    VvlaWirelessError,
)
from .client import InferenceClient, PolicyClient
from .contracts import (
    ImagePayload,
    PolicyAction,
    PolicyObservation,
    PolicyResult,
    Session,
)
from .factory import build_inference_client
from .protocols.http import (
    HttpResponse,
    HttpTransport,
    HttpTransportError,
    UrllibHttpTransport,
)
from .protocols.wireless import (
    WirelessProtocolError,
    WirelessRpcTransport,
    WirelessTransport,
)
from .providers import (
    InferenceProvider,
    ManagedCommandOptions,
    ProviderOptionsContext,
    provider,
    providers,
    register_provider,
)

__all__ = [
    "HttpResponse",
    "HttpTransport",
    "HttpTransportError",
    "ImagePayload",
    "InferenceClient",
    "PolicyAction",
    "PolicyClient",
    "PolicyObservation",
    "PolicyResult",
    "Session",
    "SglangHttpClient",
    "SglangHttpError",
    "UrllibHttpTransport",
    "VvlaHttpClient",
    "VvlaHttpError",
    "VvlaWirelessClient",
    "VvlaWirelessError",
    "WirelessProtocolError",
    "WirelessRpcTransport",
    "WirelessTransport",
    "build_inference_client",
    "InferenceProvider",
    "ManagedCommandOptions",
    "ProviderOptionsContext",
    "provider",
    "providers",
    "register_provider",
]
