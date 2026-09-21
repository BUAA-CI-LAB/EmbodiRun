"""Compatibility facade for :mod:`embodirun.model_services`.

Inference service contracts now live under the model-services domain. The
legacy import remains public so existing Control, Agent, and user code keeps
the same types and the provider registry remains process-global and unique.
"""

from embodirun.model_services import *  # noqa: F401,F403
from embodirun.model_services import __all__ as __all__
