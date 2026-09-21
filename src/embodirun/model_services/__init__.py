"""Inference service contracts and clients organized by backend and protocol."""

from .backends.embodiinfer import (
    EmbodiInferHttpClient,
    EmbodiInferHttpError,
    EmbodiInferWirelessClient,
    EmbodiInferWirelessError,
)
from .backends.sglang import SglangHttpClient, SglangHttpError
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
    "EmbodiInferHttpClient",
    "EmbodiInferHttpError",
    "EmbodiInferWirelessClient",
    "EmbodiInferWirelessError",
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
