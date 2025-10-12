"""Factory helpers for environment adapters."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, TypeVar

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
)
from gym_gui.core.adapters.base import AdapterContext, EnvironmentAdapter
from gym_gui.core.adapters.toy_text import TOY_TEXT_ADAPTERS
from gym_gui.core.adapters.box2d import BOX2D_ADAPTERS
from gym_gui.core.enums import GameId

AdapterT = TypeVar("AdapterT", bound=EnvironmentAdapter)


class AdapterFactoryError(KeyError):
    """Raised when an adapter cannot be created for the requested game."""


@lru_cache(maxsize=None)
def _registry() -> Mapping[GameId, type[EnvironmentAdapter]]:
    """Build the adapter registry on first use.

    Toy-text adapters are the initial entries; future phases can extend this by
    importing additional modules and updating the mapping.
    """

    return {
        **TOY_TEXT_ADAPTERS,
        **BOX2D_ADAPTERS,
    }


def available_games() -> Iterable[GameId]:
    """Return the set of :class:`GameId` values with registered adapters."""

    return _registry().keys()


def get_adapter_cls(game_id: GameId) -> type[EnvironmentAdapter]:
    """Look up the adapter class registered for ``game_id``."""

    try:
        return _registry()[game_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AdapterFactoryError(f"No adapter registered for '{game_id.value}'") from exc


def create_adapter(
    game_id: GameId,
    context: AdapterContext | None = None,
    *,
    game_config: (
        FrozenLakeConfig
        | TaxiConfig
        | CliffWalkingConfig
        | LunarLanderConfig
        | CarRacingConfig
        | BipedalWalkerConfig
        | None
    ) = None,
) -> EnvironmentAdapter:
    """Instantiate the adapter bound to the optional ``context`` and ``game_config``."""

    adapter_cls = get_adapter_cls(game_id)
    
    # Import adapter classes to check if game_config is supported
    from gym_gui.core.adapters.toy_text import (
        CliffWalkingAdapter,
        FrozenLakeAdapter,
        TaxiAdapter,
    )
    from gym_gui.core.adapters.box2d import (
        BipedalWalkerAdapter,
        CarRacingAdapter,
        LunarLanderAdapter,
    )
    
    # Pass game config to appropriate adapter constructor
    if game_config is not None:
        if adapter_cls is FrozenLakeAdapter and isinstance(game_config, FrozenLakeConfig):
            adapter = FrozenLakeAdapter(context, game_config=game_config)
        elif adapter_cls is TaxiAdapter and isinstance(game_config, TaxiConfig):
            adapter = TaxiAdapter(context, game_config=game_config)
        elif adapter_cls is CliffWalkingAdapter and isinstance(game_config, CliffWalkingConfig):
            adapter = CliffWalkingAdapter(context, game_config=game_config)
        elif adapter_cls is LunarLanderAdapter and isinstance(game_config, LunarLanderConfig):
            adapter = LunarLanderAdapter(context, config=game_config)
        elif adapter_cls is CarRacingAdapter and isinstance(game_config, CarRacingConfig):
            adapter = CarRacingAdapter(context, config=game_config)
        elif adapter_cls is BipedalWalkerAdapter and isinstance(game_config, BipedalWalkerConfig):
            adapter = BipedalWalkerAdapter(context, config=game_config)
        else:
            adapter = adapter_cls(context)
    else:
        adapter = adapter_cls(context)
    
    if context is not None:
        adapter.ensure_control_mode(context.control_mode)
    return adapter


__all__ = [
    "AdapterFactoryError",
    "available_games",
    "create_adapter",
    "get_adapter_cls",
]
