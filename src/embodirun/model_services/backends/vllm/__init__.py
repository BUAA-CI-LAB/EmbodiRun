"""vLLM policy service over the versioned action HTTP protocol.

The optional service adapter translates model output into actions. This client
does not treat an OpenAI chat-completions endpoint as an action service.
"""

from embodirun.model_services.backends.embodiinfer import EmbodiInferHttpClient


class VllmHttpClient(EmbodiInferHttpClient):
    """Connect to a vLLM action adapter with session/reset/idempotency support."""


__all__ = ["VllmHttpClient"]
