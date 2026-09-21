"""EmbodiInfer inference service integrations."""

from .http import EmbodiInferHttpClient, EmbodiInferHttpError, embodiinfer_http_server_command
from .wireless import (
    EmbodiInferWirelessClient,
    EmbodiInferWirelessError,
    embodiinfer_wireless_server_command,
)

__all__ = [
    "EmbodiInferHttpClient",
    "EmbodiInferHttpError",
    "EmbodiInferWirelessClient",
    "EmbodiInferWirelessError",
    "embodiinfer_http_server_command",
    "embodiinfer_wireless_server_command",
]
