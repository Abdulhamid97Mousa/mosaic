"""Factory helpers for environment adapters."""

from __future__ import annotations

from gym_gui.cache.memory import memoize
from typing import Iterable, Mapping, TypeVar

from gym_gui.cache.memory import memoize
from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    MiniGridConfig,
    ALEConfig,
    TaxiConfig,
    BlackjackConfig,
)
from gym_gui.core.adapters.base import AdapterContext, EnvironmentAdapter
from gym_gui.core.adapters.toy_text import TOY_TEXT_ADAPTERS
from gym_gui.core.adapters.box2d import BOX2D_ADAPTERS
from gym_gui.core.adapters.minigrid import MINIGRID_ADAPTERS
from gym_gui.core.adapters.ale import ALE_ADAPTERS, ALEAdapter

try:  # Optional dependency
    from gym_gui.core.adapters.vizdoom import (  # pragma: no cover - optional
        VIZDOOM_ADAPTERS,
        ViZDoomAdapter,
        ViZDoomConfig,
    )
except Exception:  # pragma: no cover - vizdoom optional
    VIZDOOM_ADAPTERS = {}
    ViZDoomAdapter = None  # type: ignore
    ViZDoomConfig = None  # type: ignore

try:  # Optional dependency - PettingZoo Classic
    from gym_gui.core.adapters.pettingzoo_classic import (
        PETTINGZOO_CLASSIC_ADAPTERS,
    )
except Exception:  # pragma: no cover - pettingzoo optional
    PETTINGZOO_CLASSIC_ADAPTERS = {}
from gym_gui.core.enums import GameId

AdapterT = TypeVar("AdapterT", bound=EnvironmentAdapter)


class AdapterFactoryError(KeyError):
    """Raised when an adapter cannot be created for the requested game."""


@memoize()
def _registry() -> Mapping[GameId, type[EnvironmentAdapter]]:
    """Build the adapter registry on first use.

    Toy-text adapters are the initial entries; future phases can extend this by
    importing additional modules and updating the mapping.
    """

    return {
        **TOY_TEXT_ADAPTERS,
        **BOX2D_ADAPTERS,
        **MINIGRID_ADAPTERS,
        **ALE_ADAPTERS,
        **VIZDOOM_ADAPTERS,
        **PETTINGZOO_CLASSIC_ADAPTERS,
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
        | BlackjackConfig
        | LunarLanderConfig
        | CarRacingConfig
        | BipedalWalkerConfig
        | MiniGridConfig
        | ALEConfig
        | ViZDoomConfig
        | None
    ) = None,
) -> EnvironmentAdapter:
    """Instantiate the adapter bound to the optional ``context`` and ``game_config``."""

    adapter_cls = get_adapter_cls(game_id)
    
    # Import adapter classes to check if game_config is supported
    from gym_gui.core.adapters.toy_text import (
        CliffWalkingAdapter,
        FrozenLakeAdapter,
        FrozenLakeV2Adapter,
        TaxiAdapter,
        BlackjackAdapter,
    )
    from gym_gui.core.adapters.minigrid import (
        MiniGridAdapter,
        MiniGridEmpty5x5Adapter,
        MiniGridEmptyRandom5x5Adapter,
        MiniGridEmpty6x6Adapter,
        MiniGridEmptyRandom6x6Adapter,
        MiniGridEmpty8x8Adapter,
        MiniGridEmpty16x16Adapter,
        MiniGridDoorKey5x5Adapter,
        MiniGridDoorKey6x6Adapter,
        MiniGridDoorKey8x8Adapter,
        MiniGridDoorKey16x16Adapter,
        MiniGridLavaGapS5Adapter,
        MiniGridLavaGapS6Adapter,
        MiniGridLavaGapS7Adapter,
    )
    from gym_gui.core.adapters.box2d import (
        BipedalWalkerAdapter,
        CarRacingAdapter,
        LunarLanderAdapter,
    )
    from gym_gui.core.adapters.ale import (
        AdventureV4Adapter,
        AdventureV5Adapter,
    )
    
    # Pass game config to appropriate adapter constructor
    if game_config is not None:
        if adapter_cls is FrozenLakeAdapter and isinstance(game_config, FrozenLakeConfig):
            adapter = FrozenLakeAdapter(context, game_config=game_config)
        elif adapter_cls is FrozenLakeV2Adapter and isinstance(game_config, FrozenLakeConfig):
            adapter = FrozenLakeV2Adapter(context, game_config=game_config)
        elif adapter_cls is TaxiAdapter and isinstance(game_config, TaxiConfig):
            adapter = TaxiAdapter(context, game_config=game_config)
        elif adapter_cls is CliffWalkingAdapter and isinstance(game_config, CliffWalkingConfig):
            adapter = CliffWalkingAdapter(context, game_config=game_config)
        elif adapter_cls is BlackjackAdapter and isinstance(game_config, BlackjackConfig):
            adapter = BlackjackAdapter(context, game_config=game_config)
        elif adapter_cls is LunarLanderAdapter and isinstance(game_config, LunarLanderConfig):
            adapter = LunarLanderAdapter(context, config=game_config)
        elif adapter_cls is CarRacingAdapter and isinstance(game_config, CarRacingConfig):
            adapter = CarRacingAdapter(context, config=game_config)
        elif adapter_cls is BipedalWalkerAdapter and isinstance(game_config, BipedalWalkerConfig):
            adapter = BipedalWalkerAdapter(context, config=game_config)
        elif issubclass(adapter_cls, MiniGridAdapter) and isinstance(game_config, MiniGridConfig):
            adapter = adapter_cls(context, config=game_config)
        elif issubclass(adapter_cls, ALEAdapter) and isinstance(game_config, ALEConfig):
            adapter = adapter_cls(context, config=game_config)
        elif (
            ViZDoomAdapter is not None
            and issubclass(adapter_cls, ViZDoomAdapter)
            and isinstance(game_config, ViZDoomConfig)
        ):
            adapter = adapter_cls(context, config=game_config)
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
