"""Go game handlers using composition pattern.

This module provides a handler class that manages Go game interactions
in Human Control Mode, including stone placement and pass handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

_LOG = logging.getLogger(__name__)


class GoHandler:
    """Handles Go game interactions in Human Control Mode.

    This class encapsulates all Go-related logic for the standard session
    flow (Human Only mode where both players are human).

    Args:
        session: The session controller managing the environment.
        render_tabs: The render tabs widget for displaying the Go board.
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

    def on_intersection_clicked(self, row: int, col: int) -> None:
        """Handle intersection click from RenderTabs (Human Control Mode).

        This is called when playing Go in Human Only mode through
        the standard SessionController flow.

        Args:
            row: Row index (0 to board_size-1)
            col: Column index (0 to board_size-1)
        """
        from gym_gui.core.enums import GameId

        # Check if we're in Go game
        if self._session.game_id != GameId.GO:
            return

        # Get the Go adapter (type narrowing for pyright)
        raw_adapter = self._session._adapter
        if raw_adapter is None:
            _LOG.warning("Go adapter is None")
            return

        # Cast to GoEnvironmentAdapter for type safety
        adapter = cast("GoEnvironmentAdapter", raw_adapter)

        # Convert to action
        action = adapter.coords_to_action(row, col)

        # Validate move
        if not adapter.is_move_legal(action):
            self._status_bar.showMessage(f"Illegal move at ({row}, {col})", 3000)
            return

        # Execute move
        try:
            step = adapter.step(action)
            # Update display
            self._render_tabs.display_payload(step.render_payload)
            # Update status
            self._update_status(adapter)
        except Exception as e:
            _LOG.error(f"Go move failed: {e}")
            self._status_bar.showMessage(f"Move failed: {e}", 5000)

    def on_pass_requested(self) -> None:
        """Handle pass action request."""
        from gym_gui.core.enums import GameId

        # Check if we're in Go game
        if self._session.game_id != GameId.GO:
            return

        raw_adapter = self._session._adapter
        if raw_adapter is None:
            _LOG.warning("Go adapter is None")
            return

        adapter = cast("GoEnvironmentAdapter", raw_adapter)

        # Get pass action
        pass_action = adapter.get_pass_action()

        # Validate pass
        if not adapter.is_move_legal(pass_action):
            self._status_bar.showMessage("Pass not allowed", 3000)
            return

        # Execute pass
        try:
            step = adapter.step(pass_action)
            self._render_tabs.display_payload(step.render_payload)
            self._status_bar.showMessage("Player passed", 3000)
            self._update_status(adapter)
        except Exception as e:
            _LOG.error(f"Go pass failed: {e}")
            self._status_bar.showMessage(f"Pass failed: {e}", 5000)

    def _update_status(self, adapter: "GoEnvironmentAdapter") -> None:
        """Update status bar with current game state.

        Args:
            adapter: The Go adapter.
        """
        try:
            game_state = adapter.get_go_state()
            player_name = "Black" if game_state.current_player == "black_0" else "White"
            status = f"{player_name}'s turn | Move {game_state.move_count}"

            if game_state.is_game_over:
                if game_state.winner == "draw":
                    status = "Game Over - Draw!"
                elif game_state.winner:
                    winner_name = "Black" if "black" in game_state.winner else "White"
                    status = f"Game Over - {winner_name} wins!"

            self._status_bar.showMessage(status, 5000)
        except Exception as e:
            _LOG.warning(f"Failed to update Go status: {e}")

    @staticmethod
    def is_go_payload(payload: dict) -> bool:
        """Check if a payload is a Go render payload.

        Args:
            payload: The render payload to check.

        Returns:
            True if this is a Go payload.
        """
        return payload.get("game_type") == "go" and "board" in payload


__all__ = ["GoHandler"]
