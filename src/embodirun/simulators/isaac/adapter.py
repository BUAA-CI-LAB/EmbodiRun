"""Deploy adapter for Isaac Sim visual navigation."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..navigation import NavigationEnvironment, NavigationSimulatorAdapter
from .config import IsaacConfig
from .environment import make_isaac_environment


class IsaacAdapter(NavigationSimulatorAdapter):
    def __init__(
        self,
        config: IsaacConfig,
        *,
        environment_factory: Callable[[IsaacConfig], NavigationEnvironment] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, IsaacConfig):
            raise TypeError("config must be an IsaacConfig")
        super().__init__(
            config,
            environment_factory=environment_factory or make_isaac_environment,
            engine_name="Isaac",
            clock=clock,
        )


__all__ = ["IsaacAdapter"]
