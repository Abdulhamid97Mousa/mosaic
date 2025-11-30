"""Connect Four game handlers using composition pattern.

This module provides a handler class that manages Connect Four game interactions
in Human Control Mode, including column selection and board state updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.core.adapters.pettingzoo_classic import ConnectFourEnvironmentAdapter

_LOG = logging.getLogger(__name__)


class ConnectFourHandler:
    """Handles Connect Four game interactions in Human Control Mode.

    This class encapsulates all Connect Four-related logic for the standard session
    flow (Human Only mode where both players are human).

    Args:
        session: The session controller managing the environment.
        render_tabs: The render tabs widget for displaying the Connect Four board.
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(
        self,
        session: "SessionController",
        render_tabs: "RenderTabs",
        status_bar: "QStatusBar",
    ) -> None:
        self._session = session
        self._render_tabs = render_tabs
        self._status_bar = status_bar

    def on_column_clicked(self, column: int) -> None:
        """Handle column click from RenderTabs (Human Control Mode).

        This is called when playing Connect Four in Human Only mode through
        the standard SessionController flow.

        Args:
            column: Column index (0-6)
        """
        from gym_gui.core.enums import GameId

        # Check if we're in Connect Four game
        if self._session.game_id != GameId.CONNECT_FOUR:
            return

        # Get the Connect Four adapter (type narrowing for pyright)
        raw_adapter = self._session._adapter
        if raw_adapter is None:
            _LOG.warning("Connect Four adapter is None")
            return

        # Cast to ConnectFourEnvironmentAdapter for type safety
        adapter = cast("ConnectFourEnvironmentAdapter", raw_adapter)

        # Validate column
        if not adapter.is_column_legal(column):
            self._status_bar.showMessage(f"Column {column + 1} is full", 3000)
            return

        # Execute move
        try:
            step = adapter.step(column)
            # Update display
            self._render_tabs.display_payload(step.render_payload)
            # Update status
            self._update_status(adapter)
        except Exception as e:
            _LOG.error(f"Connect Four move failed: {e}")
            self._status_bar.showMessage(f"Move failed: {e}", 5000)

    def _update_status(self, adapter: "ConnectFourEnvironmentAdapter") -> None:
        """Update status bar with current game state.

        Args:
            adapter: The Connect Four adapter.
        """
        try:
            game_state = adapter.get_connect_four_state()
            player_name = "Red" if game_state.current_player == "player_0" else "Yellow"
            status = f"{player_name}'s turn (Player {1 if game_state.current_player == 'player_0' else 2})"

            if game_state.is_game_over:
                if game_state.winner == "draw":
                    status = "Game Over - Draw!"
                elif game_state.winner:
                    winner_name = "Red" if game_state.winner == "player_0" else "Yellow"
                    winner_num = 1 if game_state.winner == "player_0" else 2
                    status = f"Game Over - {winner_name} (Player {winner_num}) wins!"

            self._status_bar.showMessage(status, 5000)
        except Exception as e:
            _LOG.warning(f"Failed to update Connect Four status: {e}")

    @staticmethod
    def is_connect_four_payload(payload: dict) -> bool:
        """Check if a payload is a Connect Four render payload.

        Args:
            payload: The render payload to check.

        Returns:
            True if this is a Connect Four payload.
        """
        return payload.get("game_type") == "connect_four" and "board" in payload


__all__ = ["ConnectFourHandler"]
