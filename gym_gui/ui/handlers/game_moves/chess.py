"""Chess game handlers using composition pattern.

This module provides a handler class that manages Chess game interactions
in Human Control Mode, including move handling and board state updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.core.adapters.pettingzoo_classic import ChessEnvironmentAdapter

_LOG = logging.getLogger(__name__)


class ChessHandler:
    """Handles Chess game interactions in Human Control Mode.

    This class encapsulates all Chess-related logic for the standard session
    flow (Human Only mode where both players are human).

    Args:
        session: The session controller managing the environment.
        render_tabs: The render tabs widget for displaying the chess board.
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

    def on_chess_move(self, from_sq: str, to_sq: str) -> None:
        """Handle chess move from RenderTabs (Human Control Mode).

        This is called when playing Chess in Human Only mode through
        the standard SessionController flow.

        Args:
            from_sq: Source square (e.g., "e2")
            to_sq: Destination square (e.g., "e4")
        """
        from gym_gui.core.enums import GameId

        # Check if we're in Chess game
        if self._session.game_id != GameId.CHESS:
            return

        # Get the chess adapter (type narrowing for pyright)
        raw_adapter = self._session._adapter
        if raw_adapter is None:
            _LOG.warning("Chess adapter is None")
            return

        # Cast to ChessEnvironmentAdapter for type safety
        adapter = cast("ChessEnvironmentAdapter", raw_adapter)

        # Build UCI move
        uci_move = f"{from_sq}{to_sq}"

        # Check for pawn promotion (simple: always promote to queen)
        uci_move = self._handle_promotion(adapter, from_sq, to_sq, uci_move)

        # Validate move
        if not adapter.is_move_legal(uci_move):
            self._status_bar.showMessage(f"Illegal move: {uci_move}", 3000)
            return

        # Execute move
        try:
            step = adapter.step_uci(uci_move)
            # Update display
            self._render_tabs.display_payload(step.render_payload)
            # Update status
            self._update_status(adapter)
        except Exception as e:
            _LOG.error(f"Chess move failed: {e}")
            self._status_bar.showMessage(f"Move failed: {e}", 5000)

    def _handle_promotion(
        self, adapter: Any, from_sq: str, to_sq: str, uci_move: str
    ) -> str:
        """Check for pawn promotion and append promotion piece.

        Args:
            adapter: The chess adapter.
            from_sq: Source square.
            to_sq: Destination square.
            uci_move: Current UCI move string.

        Returns:
            UCI move string, possibly with promotion suffix.
        """
        try:
            from_row = int(from_sq[1])
            to_row = int(to_sq[1])
            # Get piece at from_sq
            chess_state = adapter.get_chess_state()
            fen = chess_state.fen
            piece = self._get_piece_from_fen(fen, from_sq)
            if piece and piece.upper() == "P":
                if (piece == "P" and to_row == 8) or (piece == "p" and to_row == 1):
                    uci_move += "q"  # Auto-promote to queen
        except Exception:
            pass
        return uci_move

    def _update_status(self, adapter: Any) -> None:
        """Update status bar with current game state.

        Args:
            adapter: The chess adapter.
        """
        try:
            chess_state = adapter.get_chess_state()
            status = f"{chess_state.current_player.capitalize()}'s turn"
            if chess_state.is_check:
                status += " - CHECK!"
            if chess_state.is_game_over:
                if chess_state.winner == "draw":
                    status = "Game Over - Draw!"
                elif chess_state.winner:
                    status = f"Game Over - {chess_state.winner.capitalize()} wins!"
            self._status_bar.showMessage(status, 5000)
        except Exception as e:
            _LOG.warning(f"Failed to update chess status: {e}")

    @staticmethod
    def _get_piece_from_fen(fen: str, square: str) -> Optional[str]:
        """Get piece at a square from FEN.

        Args:
            fen: FEN position string.
            square: Square in algebraic notation (e.g., "e2").

        Returns:
            Piece character or None if empty/invalid.
        """
        position = fen.split()[0] if fen else ""
        col = ord(square[0]) - ord("a")
        row = int(square[1]) - 1

        current_row = 7
        current_col = 0

        for char in position:
            if char == "/":
                current_row -= 1
                current_col = 0
            elif char.isdigit():
                current_col += int(char)
            else:
                if current_row == row and current_col == col:
                    return char
                current_col += 1

        return None

    @staticmethod
    def is_chess_payload(payload: Mapping[str, Any]) -> bool:
        """Check if a payload is a chess render payload.

        Args:
            payload: The render payload to check.

        Returns:
            True if this is a chess payload (has 'fen' and 'legal_moves').
        """
        return "fen" in payload and "legal_moves" in payload


__all__ = ["ChessHandler"]
