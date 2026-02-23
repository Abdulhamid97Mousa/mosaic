"""Sudoku game handlers using composition pattern.

This module provides a handler class that manages Sudoku game interactions
in Human Control Mode, including cell selection and digit entry.

Interaction flow:
1. User clicks cell to select it
2. User presses 1-9 to enter digit (validated against action mask)
3. Handler computes action and executes step through adapter
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.core.adapters.jumanji import JumanjiSudokuAdapter

_LOG = logging.getLogger(__name__)


class SudokuHandler:
    """Handles Sudoku game interactions in Human Control Mode.

    This class encapsulates all Sudoku-related logic for the standard session
    flow (Human Only mode where the human solves the puzzle).

    The handler:
    - Tracks the currently selected cell
    - Validates digit entries against Jumanji's action mask
    - Executes steps through the adapter

    Args:
        session: The session controller managing the environment.
        render_tabs: The render tabs widget for displaying the Sudoku board.
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

        # Track selected cell (row, col)
        self._selected_cell: tuple[int, int] | None = None

    @property
    def selected_cell(self) -> tuple[int, int] | None:
        """Currently selected cell, or None if no selection."""
        return self._selected_cell

    def on_cell_selected(self, row: int, col: int) -> None:
        """Handle cell selection from the board renderer.

        Args:
            row: Row index (0-8)
            col: Column index (0-8)
        """
        from gym_gui.core.enums import GameId

        # Check if we're in Sudoku game
        if self._session.game_id != GameId.JUMANJI_SUDOKU:
            return

        self._selected_cell = (row, col)
        _LOG.debug(f"Sudoku cell selected: ({row}, {col})")

    def on_digit_entered(self, row: int, col: int, digit: int) -> None:
        """Handle digit entry from keyboard.

        This is called when the user presses 1-9 while a cell is selected.
        The renderer validates against action_mask before emitting.

        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            digit: Digit to place (1-9)
        """
        from gym_gui.core.enums import GameId

        # Check if we're in Sudoku game
        if self._session.game_id != GameId.JUMANJI_SUDOKU:
            return

        # Get the Sudoku adapter
        raw_adapter = self._session._adapter
        if raw_adapter is None:
            _LOG.warning("Sudoku adapter is None")
            return

        # Cast to JumanjiSudokuAdapter for type safety
        adapter = cast("JumanjiSudokuAdapter", raw_adapter)

        # Compute action
        action = adapter.compute_action(row, col, digit)
        _LOG.debug(f"Sudoku digit entered: ({row}, {col}) = {digit} -> action {action}")

        # Execute step
        try:
            step = adapter.step(action)

            # Update display
            render_payload = adapter.render()
            self._render_tabs.display_payload(render_payload)

            # Update status
            self._update_status(adapter, row, col, digit, step)

        except Exception as e:
            _LOG.error(f"Sudoku move failed: {e}")
            self._status_bar.showMessage(f"Invalid move: {e}", 5000)

    def on_cell_cleared(self, row: int, col: int) -> None:
        """Handle cell clear request (Delete/Backspace key).

        Note: Jumanji Sudoku may not support clearing cells - this depends
        on the environment implementation. We log the request but may not
        be able to execute it.

        Args:
            row: Row index (0-8)
            col: Column index (0-8)
        """
        _LOG.debug(f"Sudoku cell clear requested: ({row}, {col})")
        self._status_bar.showMessage(
            "Note: Jumanji Sudoku doesn't support clearing cells", 3000
        )

    def _update_status(
        self,
        adapter: Any,
        row: int,
        col: int,
        digit: int,
        step: Any,
    ) -> None:
        """Update status bar with current game state.

        Args:
            adapter: The Sudoku adapter.
            row: Row of placed digit.
            col: Column of placed digit.
            digit: Digit that was placed.
            step: Step result from adapter.
        """
        try:
            # Get metrics from step state
            metrics = step.step_state.metrics if step.step_state else {}
            cells_filled = metrics.get("cells_filled", "?")
            cells_remaining = metrics.get("cells_remaining", "?")

            status = f"Placed {digit} at ({row+1}, {col+1})"
            status += f" | Filled: {cells_filled}/81"

            if cells_remaining == 0:
                status = "Puzzle Complete!"

            if step.terminated:
                if step.reward > 0:
                    status = "Puzzle Solved! Congratulations!"
                else:
                    status = "Game Over - Invalid state"

            self._status_bar.showMessage(status, 5000)

        except Exception as e:
            _LOG.warning(f"Failed to update Sudoku status: {e}")

    def reset(self) -> None:
        """Reset handler state."""
        self._selected_cell = None


__all__ = ["SudokuHandler"]
