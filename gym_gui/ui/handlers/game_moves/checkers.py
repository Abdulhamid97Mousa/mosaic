"""Checkers game handlers using composition pattern.

This module provides a handler class that manages Checkers game interactions
in Human Control Mode, including move handling and board state updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, cast

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_CHECKERS_HANDLER_CLICK_RECEIVED,
    LOG_CHECKERS_HANDLER_GAME_MISMATCH,
    LOG_CHECKERS_MOVE_EXECUTED,
    LOG_CHECKERS_MOVE_FAILED,
    LOG_CHECKERS_PIECE_SELECTED,
)

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.core.adapters.open_spiel import CheckersEnvironmentAdapter

_LOG = logging.getLogger(__name__)


class CheckersHandler:
    """Handles Checkers game interactions in Human Control Mode.

    This class encapsulates all Checkers-related logic for the standard session
    flow (Human Only mode where both players are human).

    Uses a two-click selection system:
    1. First click selects a piece (source square)
    2. Second click selects the destination

    Args:
        session: The session controller managing the environment.
        render_tabs: The render tabs widget for displaying the checkers board.
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
        self._selected_square: Optional[str] = None  # Currently selected piece

    def on_checkers_cell_clicked(self, row: int, col: int) -> None:
        """Handle checkers cell click from RenderTabs (Human Control Mode).

        This is called when playing Checkers in Human Only mode through
        the standard SessionController flow.

        Args:
            row: Row index (0 = top, 7 = bottom)
            col: Column index (0 = left/a, 7 = right/h)
        """
        from gym_gui.core.enums import GameId

        log_constant(
            _LOG,
            LOG_CHECKERS_HANDLER_CLICK_RECEIVED,
            extra={"row": row, "col": col, "game_id": str(self._session.game_id)},
        )

        # Check if we're in Checkers game
        if self._session.game_id != GameId.OPEN_SPIEL_CHECKERS:
            log_constant(
                _LOG,
                LOG_CHECKERS_HANDLER_GAME_MISMATCH,
                extra={
                    "expected": str(GameId.OPEN_SPIEL_CHECKERS),
                    "got": str(self._session.game_id),
                },
            )
            return

        # Get the checkers adapter (type narrowing for pyright)
        raw_adapter = self._session._adapter
        if raw_adapter is None:
            _LOG.warning("Checkers adapter is None")
            return

        # Cast to CheckersEnvironmentAdapter for type safety
        adapter = cast("CheckersEnvironmentAdapter", raw_adapter)

        # Convert cell to algebraic notation
        clicked_square = adapter.cell_to_algebraic(row, col)

        # Two-click selection logic
        if self._selected_square is None:
            # First click - select piece
            self._handle_piece_selection(adapter, clicked_square, row, col)
        else:
            # Second click - try to move
            self._handle_move_attempt(adapter, clicked_square)

    def _handle_piece_selection(
        self,
        adapter: "CheckersEnvironmentAdapter",
        square: str,
        row: int,
        col: int,
    ) -> None:
        """Handle selection of a piece (first click).

        Args:
            adapter: The checkers adapter.
            square: Square in algebraic notation.
            row: Row index.
            col: Column index.
        """
        # Check if there's a piece on this square that can move
        moves_from = adapter.get_moves_from_square(square)

        if moves_from:
            self._selected_square = square
            self._status_bar.showMessage(
                f"Selected {square} - click destination to move", 5000
            )
            log_constant(
                _LOG,
                LOG_CHECKERS_PIECE_SELECTED,
                extra={"square": square, "legal_moves": moves_from},
            )
            # Update visual highlights
            self._update_selection_highlight(adapter, row, col, moves_from)
        else:
            # No piece or no legal moves from this square
            # Get all legal moves to provide helpful feedback
            all_moves = adapter.get_legal_move_strings()
            if not all_moves:
                self._status_bar.showMessage("No legal moves available!", 3000)
                self._clear_selection_highlight()
            else:
                # Get source squares that can move
                sources = sorted(set(m[:2] for m in all_moves))

                if len(all_moves) == 1:
                    # Mandatory jump - tell user where to click
                    from_sq = all_moves[0][:2]
                    self._status_bar.showMessage(
                        f"Mandatory jump! Click {from_sq} to move", 3000
                    )
                else:
                    # Multiple moves available, but not from this square
                    self._status_bar.showMessage(
                        f"No moves from {square}. Try: {', '.join(sources)}", 3000
                    )

                # Highlight the pieces that CAN move
                self._highlight_moveable_pieces(adapter, sources)

    def _handle_move_attempt(
        self,
        adapter: "CheckersEnvironmentAdapter",
        dest_square: str,
    ) -> None:
        """Handle move attempt (second click).

        Args:
            adapter: The checkers adapter.
            dest_square: Destination square in algebraic notation.
        """
        from_sq = self._selected_square
        assert from_sq is not None  # We only call this when selected

        # Check if clicking the same square (deselect)
        if dest_square == from_sq:
            self._selected_square = None
            self._status_bar.showMessage("Selection cleared", 2000)
            self._clear_selection_highlight()
            return

        # Check if clicking another piece with legal moves (reselect)
        moves_from_dest = adapter.get_moves_from_square(dest_square)
        if moves_from_dest:
            # Reselect to the new piece
            self._selected_square = dest_square
            self._status_bar.showMessage(
                f"Reselected {dest_square} - click destination to move", 5000
            )
            # Update visual highlights for new selection
            dest_row, dest_col = adapter.algebraic_to_cell(dest_square)
            self._update_selection_highlight(adapter, dest_row, dest_col, moves_from_dest)
            return

        # Try to find action for this move
        action = adapter.find_action_for_move(from_sq, dest_square)

        if action is None:
            self._status_bar.showMessage(
                f"Illegal move: {from_sq} to {dest_square}", 3000
            )
            self._selected_square = None
            self._clear_selection_highlight()
            return

        # Execute move
        try:
            step = adapter.step(action)
            self._selected_square = None

            log_constant(
                _LOG,
                LOG_CHECKERS_MOVE_EXECUTED,
                extra={"from": from_sq, "to": dest_square, "action": action},
            )

            # Update display
            self._render_tabs.display_payload(step.render_payload)

            # Update status
            self._update_status(adapter, from_sq, dest_square)

        except Exception as e:
            log_constant(
                _LOG,
                LOG_CHECKERS_MOVE_FAILED,
                exc_info=e,
                extra={"from": from_sq, "to": dest_square},
            )
            self._status_bar.showMessage(f"Move failed: {e}", 5000)
            self._selected_square = None
            self._clear_selection_highlight()

    def _update_status(
        self,
        adapter: "CheckersEnvironmentAdapter",
        from_sq: str,
        to_sq: str,
    ) -> None:
        """Update status bar with current game state.

        Args:
            adapter: The checkers adapter.
            from_sq: Source square of last move.
            to_sq: Destination square of last move.
        """
        try:
            checkers_state = adapter.get_checkers_state()
            player = "Black" if checkers_state.current_player == "player_0" else "White"

            if checkers_state.is_game_over:
                if checkers_state.winner == "player_0":
                    status = "Game Over - Black wins!"
                elif checkers_state.winner == "player_1":
                    status = "Game Over - White wins!"
                else:
                    status = "Game Over - Draw!"
            else:
                status = f"Move: {from_sq}{to_sq} | {player}'s turn"

            self._status_bar.showMessage(status, 5000)
        except Exception as e:
            _LOG.warning(f"Failed to update checkers status: {e}")

    def _update_selection_highlight(
        self,
        adapter: "CheckersEnvironmentAdapter",
        row: int,
        col: int,
        destinations: List[str],
    ) -> None:
        """Update the visual selection highlight on the board.

        Args:
            adapter: The checkers adapter.
            row: Row of selected piece.
            col: Column of selected piece.
            destinations: List of destination squares in algebraic notation.
        """
        try:
            strategy = self._render_tabs.get_board_game_strategy()
            if strategy is None:
                return

            # Convert destination squares to cell coordinates
            dest_cells: List[tuple[int, int]] = []
            for dest_sq in destinations:
                dest_row, dest_col = adapter.algebraic_to_cell(dest_sq)
                dest_cells.append((dest_row, dest_col))

            strategy.set_checkers_selection((row, col), dest_cells)
        except Exception as e:
            _LOG.debug(f"Failed to update selection highlight: {e}")

    def _clear_selection_highlight(self) -> None:
        """Clear the visual selection highlight on the board."""
        try:
            strategy = self._render_tabs.get_board_game_strategy()
            if strategy is not None:
                strategy.set_checkers_selection(None)
                strategy.set_checkers_moveable_cells(None)
        except Exception as e:
            _LOG.debug(f"Failed to clear selection highlight: {e}")

    def _highlight_moveable_pieces(
        self,
        adapter: "CheckersEnvironmentAdapter",
        source_squares: List[str],
    ) -> None:
        """Highlight pieces that can move (for mandatory jump hints).

        Args:
            adapter: The checkers adapter.
            source_squares: List of source squares in algebraic notation (e.g., ['a3', 'c5']).
        """
        try:
            strategy = self._render_tabs.get_board_game_strategy()
            if strategy is None:
                return

            # Convert source squares to cell coordinates
            cells: List[tuple[int, int]] = []
            for sq in source_squares:
                row, col = adapter.algebraic_to_cell(sq)
                cells.append((row, col))

            # Clear any selection and show moveable hints
            strategy.set_checkers_selection(None)
            strategy.set_checkers_moveable_cells(cells)
        except Exception as e:
            _LOG.debug(f"Failed to highlight moveable pieces: {e}")

    def clear_selection(self) -> None:
        """Clear the current piece selection."""
        self._selected_square = None

    @property
    def has_selection(self) -> bool:
        """Check if a piece is currently selected."""
        return self._selected_square is not None

    @property
    def selected_square(self) -> Optional[str]:
        """Get the currently selected square."""
        return self._selected_square

    @staticmethod
    def is_checkers_payload(payload: Mapping[str, Any]) -> bool:
        """Check if a payload is a checkers render payload.

        Args:
            payload: The render payload to check.

        Returns:
            True if this is a checkers payload (has 'game_type' == 'checkers').
        """
        return payload.get("game_type") == "checkers"


__all__ = ["CheckersHandler"]
