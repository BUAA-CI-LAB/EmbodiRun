"""Connect to the frozen benchmark through the shared v1 compatibility path."""

from embodirun.model_services.backends.embodiinfer.compat import connect_client as _connect_client
from embodirun.model_services.backends.embodiinfer.http import EmbodiInferHttpClient
from embodirun.model_services.protocols.http import HttpTransport


def connect_client(
    endpoint: str, token: str | None, *, transport: HttpTransport | None = None
) -> EmbodiInferHttpClient:
    """Retain the benchmark timeout while negotiating its frozen v1 namespace."""
    return _connect_client(endpoint, token, timeout_s=240, transport=transport)
