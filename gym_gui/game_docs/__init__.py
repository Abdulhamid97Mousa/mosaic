"""Aggregated environment documentation helpers."""

from __future__ import annotations

from typing import Dict

from gym_gui.core.enums import GameId
from gym_gui.game_docs.Gymnasium.ToyText import (
    TAXI_HTML,
    FROZEN_HTML,
    FROZEN_V2_HTML,
    CLIFF_HTML,
    BLACKJACK_HTML,
)
from gym_gui.game_docs.Gymnasium.Box2D import (
    LUNAR_LANDER_HTML,
    CAR_RACING_HTML,
    BIPEDAL_WALKER_HTML,
)

from gym_gui.game_docs.MiniGrid import (
    MINIGRID_EMPTY_HTML,
    MINIGRID_DOORKEY_HTML,
    MINIGRID_LAVAGAP_HTML,
)

_DEFAULT_DOC = (
    "<h3>Documentation unavailable</h3>"
    "<p>This environment does not yet have a descriptive blurb."
    " Check the upstream project for details.</p>"
)

GAME_INFO: Dict[GameId, str] = {
    GameId.TAXI: TAXI_HTML,
    GameId.FROZEN_LAKE: FROZEN_HTML,
    GameId.FROZEN_LAKE_V2: FROZEN_V2_HTML,
    GameId.CLIFF_WALKING: CLIFF_HTML,
    GameId.BLACKJACK: BLACKJACK_HTML,
    GameId.LUNAR_LANDER: LUNAR_LANDER_HTML,
    GameId.CAR_RACING: CAR_RACING_HTML,
    GameId.BIPEDAL_WALKER: BIPEDAL_WALKER_HTML,
}

_MINIGRID_EMPTY_VARIANTS = (
    GameId.MINIGRID_EMPTY_5x5,
    GameId.MINIGRID_EMPTY_RANDOM_5x5,
    GameId.MINIGRID_EMPTY_6x6,
    GameId.MINIGRID_EMPTY_RANDOM_6x6,
    GameId.MINIGRID_EMPTY_8x8,
    GameId.MINIGRID_EMPTY_16x16,
)
GAME_INFO.update({gid: MINIGRID_EMPTY_HTML for gid in _MINIGRID_EMPTY_VARIANTS})

_MINIGRID_DOORKEY_VARIANTS = (
    GameId.MINIGRID_DOORKEY_5x5,
    GameId.MINIGRID_DOORKEY_6x6,
    GameId.MINIGRID_DOORKEY_8x8,
    GameId.MINIGRID_DOORKEY_16x16,
)
GAME_INFO.update({gid: MINIGRID_DOORKEY_HTML for gid in _MINIGRID_DOORKEY_VARIANTS})

_MINIGRID_LAVAGAP_VARIANTS = (
    GameId.MINIGRID_LAVAGAP_S5,
    GameId.MINIGRID_LAVAGAP_S6,
    GameId.MINIGRID_LAVAGAP_S7,
)
GAME_INFO.update({gid: MINIGRID_LAVAGAP_HTML for gid in _MINIGRID_LAVAGAP_VARIANTS})


def get_game_info(game_id: GameId) -> str:
    """Return HTML documentation for the specified environment."""

    return GAME_INFO.get(game_id, _DEFAULT_DOC)


__all__ = ["GAME_INFO", "get_game_info"]
