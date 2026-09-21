"""Deploy adapter for Habitat visual navigation."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..navigation import NavigationEnvironment, NavigationSimulatorAdapter
from .config import HabitatConfig
from .environment import make_habitat_environment


class HabitatAdapter(NavigationSimulatorAdapter):
    def __init__(
        self,
        config: HabitatConfig,
        *,
        environment_factory: Callable[[HabitatConfig], NavigationEnvironment] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, HabitatConfig):
            raise TypeError("config must be a HabitatConfig")
        super().__init__(
            config,
            environment_factory=environment_factory or make_habitat_environment,
            engine_name="Habitat",
            clock=clock,
        )


__all__ = ["HabitatAdapter"]
