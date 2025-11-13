from __future__ import annotations

from typing import Any

from gym_gui.core.enums import (
    ENVIRONMENT_FAMILY_BY_GAME,
    EnvironmentFamily,
    GameId,
    RenderMode,
    DEFAULT_RENDER_MODES,
)
from gym_gui.rendering import RendererRegistry
from gym_gui.controllers.interaction import (
    Box2DInteractionController,
    TurnBasedInteractionController,
    AleInteractionController,
)
from gym_gui.core.factories.adapters import create_adapter
from gym_gui.core.session_components import SessionComponents


def create_session(game_id: GameId, context: Any, *, game_config: Any | None, renderer_registry: RendererRegistry, owner: Any) -> SessionComponents:
    family = ENVIRONMENT_FAMILY_BY_GAME[game_id]
    adapter = create_adapter(game_id, context, game_config=game_config)

    render_mode = getattr(game_config, "render_mode", None) or DEFAULT_RENDER_MODES.get(game_id, RenderMode.RGB_ARRAY)
    renderer = renderer_registry.create(render_mode)

    if family in (EnvironmentFamily.BOX2D, EnvironmentFamily.MUJOCO):
        interaction = Box2DInteractionController(owner, target_hz=50)
    elif family in (EnvironmentFamily.ATARI, EnvironmentFamily.ALE):
        interaction = AleInteractionController(owner, target_hz=60)
    else:
        interaction = TurnBasedInteractionController()

    return SessionComponents(adapter=adapter, renderer=renderer, interaction=interaction)
