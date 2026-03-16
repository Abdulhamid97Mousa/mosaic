"""MalmoEnv loader for FPS-style mouse capture.

Mouse horizontal movement → turn left (action 4) / turn right (action 5).
This mirrors the VizdoomEnvLoader pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs

from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME, EnvironmentFamily, GameId

_LOG = logging.getLogger(__name__)

# MalmoEnv turn action indices (consistent across all 13 missions)
# Action list: 0=move 1, 1=move -1, 2=turn 1 (right), 3=turn -1 (left), 4+=mission-specific
_TURN_LEFT_ACTION = 3   # "turn -1"
_TURN_RIGHT_ACTION = 2  # "turn 1"


class MalmoEnvLoader:
    """Loader for MalmoEnv (Microsoft Malmo / Minecraft) environment setup.

    Configures FPS-style mouse capture so that horizontal mouse movement steers
    the Minecraft player view, exactly as physical Minecraft mouse-look works.

    Args:
        render_tabs: The render tabs widget for configuring mouse capture.
    """

    def __init__(self, render_tabs: "RenderTabs") -> None:
        self._render_tabs = render_tabs

    def configure_mouse_capture(
        self,
        session: "SessionController",
        sensitivity: float = 5.0,
    ) -> bool:
        """Configure FPS-style mouse capture for MalmoEnv games.

        Click on the Video tab to capture the mouse; ESC or focus-loss releases it.
        Horizontal mouse movement maps to turn-left / turn-right discrete actions.

        Args:
            session: The session controller with the current game.
            sensitivity: Pixels of horizontal mouse movement per turn action (default 5.0).

        Returns:
            True if mouse capture was enabled (MalmoEnv game), False otherwise.
        """
        game_id = session.game_id
        if game_id is None:
            self._render_tabs.configure_mouse_capture(enabled=False)
            return False

        try:
            gid = GameId(game_id)
        except ValueError:
            self._render_tabs.configure_mouse_capture(enabled=False)
            return False

        family = ENVIRONMENT_FAMILY_BY_GAME.get(gid)
        if family != EnvironmentFamily.MALMOENV:
            self._render_tabs.configure_mouse_capture(enabled=False)
            return False

        def mouse_action_callback(action: int) -> None:
            session.perform_human_action(action, key_label="mouse_turn")

        self._render_tabs.configure_mouse_capture(
            enabled=True,
            action_callback=mouse_action_callback,
            turn_left_action=_TURN_LEFT_ACTION,
            turn_right_action=_TURN_RIGHT_ACTION,
            sensitivity=sensitivity,
        )
        _LOG.debug("MalmoEnv mouse capture enabled for %s", game_id)
        return True

    def disable_mouse_capture(self) -> None:
        """Disable mouse capture."""
        self._render_tabs.configure_mouse_capture(enabled=False)


__all__ = ["MalmoEnvLoader"]
