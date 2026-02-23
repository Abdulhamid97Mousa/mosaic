"""Tic-Tac-Toe environment loader for Human vs Agent mode.

This loader handles:
- Tic-Tac-Toe game initialization with interactive board
- AI opponent setup
- Game lifecycle management
- Signal connections between components
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from qtpy import QtCore, QtWidgets

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget

from gym_gui.controllers.tictactoe_controller import TicTacToeGameController
from gym_gui.core.adapters.tictactoe_adapter import TicTacToeState
from gym_gui.core.enums import GameId
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

_LOG = logging.getLogger(__name__)


class TicTacToeEnvLoader:
    """Loader for Human vs Agent Tic-Tac-Toe games.

    This class encapsulates all Tic-Tac-Toe-specific loading and lifecycle management,
    keeping MainWindow clean of environment-specific code.

    Uses the existing BoardGameRendererStrategy for rendering, connecting its signals
    to the game controller for move handling.

    Args:
        render_tabs: The render tabs widget for displaying the board.
        control_panel: The control panel for accessing game configuration.
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(
        self,
        render_tabs: "RenderTabs",
        control_panel: "ControlPanelWidget",
        status_bar: "QStatusBar",
    ) -> None:
        self._render_tabs = render_tabs
        self._control_panel = control_panel
        self._status_bar = status_bar

        # Game components (created on load)
        self._controller: Optional[TicTacToeGameController] = None
        self._renderer_strategy: Optional[BoardGameRendererStrategy] = None
        self._tab_index: int = -1
        self._signal_connected: bool = False

    @property
    def controller(self) -> Optional[TicTacToeGameController]:
        """The active Tic-Tac-Toe game controller, if any."""
        return self._controller

    @property
    def is_loaded(self) -> bool:
        """Whether a Tic-Tac-Toe game is currently loaded."""
        return self._controller is not None

    def load(self, seed: int, parent: Optional[QtCore.QObject] = None) -> str:
        """Load and initialize the Tic-Tac-Toe game with interactive board.

        Args:
            seed: Random seed for game initialization.
            parent: Parent QObject for TicTacToeGameController (usually MainWindow).

        Returns:
            AI opponent display name.
        """
        # Clean up existing game
        self.cleanup()

        # Get human vs agent tab reference
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent

        # Create renderer strategy
        self._renderer_strategy = BoardGameRendererStrategy(self._render_tabs)

        # Create game controller
        self._controller = TicTacToeGameController(parent)

        # Connect controller signals
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.game_started.connect(self._on_game_started)
        self._controller.game_over.connect(self._on_game_over)
        self._controller.status_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._controller.error_occurred.connect(
            lambda msg: self._status_bar.showMessage(f"Tic-Tac-Toe error: {msg}", 5000)
        )

        # Connect renderer signal to controller for human moves
        self._renderer_strategy.tictactoe_cell_clicked.connect(
            self._on_cell_clicked
        )
        self._signal_connected = True

        # Add board tab with proper Human vs Agent naming
        self._tab_index = self._render_tabs.addTab(
            self._renderer_strategy.widget, "Human vs Agent - Tic-Tac-Toe"
        )
        self._render_tabs.setCurrentIndex(self._tab_index)

        _LOG.info(
            f"Tic-Tac-Toe board added: tab_index={self._tab_index}, "
            f"widget_visible={self._renderer_strategy.widget.isVisible()}"
        )

        # Get human player selection
        human_agent = human_vs_agent_tab._human_player_combo.currentData()
        # Map player_0/player_1 to player_1/player_2 (PettingZoo naming)
        human_player = "player_1" if human_agent == "player_0" else "player_2"

        # AI is Random for Tic-Tac-Toe (no Stockfish-like engine)
        ai_name = "Random AI"

        # Start the game
        self._controller.start_game(human_player=human_player, seed=seed)

        # Update control panel state
        human_vs_agent_tab.set_environment_loaded("tictactoe_v3", seed)
        human_vs_agent_tab.set_policy_loaded(ai_name)
        human_vs_agent_tab._reset_btn.setEnabled(True)

        human_mark = "X" if human_player == "player_1" else "O"
        self._status_bar.showMessage(
            f"Tic-Tac-Toe loaded (seed={seed}). You play as {human_mark} vs {ai_name}. Click cell to place mark.",
            5000
        )

        return ai_name

    def cleanup(self) -> None:
        """Clean up Tic-Tac-Toe game resources."""
        # Disconnect signals
        if self._signal_connected and self._renderer_strategy is not None:
            try:
                self._renderer_strategy.tictactoe_cell_clicked.disconnect(
                    self._on_cell_clicked
                )
            except Exception:
                pass
            self._signal_connected = False

        # Clean up controller
        if self._controller is not None:
            self._controller.close()
            self._controller = None

        # Remove tab
        if self._tab_index >= 0:
            self._render_tabs.removeTab(self._tab_index)
            self._tab_index = -1

        self._renderer_strategy = None

    def on_start_requested(self, human_agent: str, seed: int) -> None:
        """Handle start game request.

        Args:
            human_agent: Which agent the human plays ("player_0" or "player_1" from GUI).
            seed: Random seed.
        """
        if self._controller is not None:
            # Map GUI generic names to PettingZoo agent names
            # GUI: player_0=first, player_1=second
            # PettingZoo Tic-Tac-Toe: player_1=X (first), player_2=O (second)
            human_player = "player_1" if human_agent == "player_0" else "player_2"
            human_mark = "X" if human_player == "player_1" else "O"
            self._controller.start_game(human_player=human_player, seed=seed)
            self._status_bar.showMessage(f"Tic-Tac-Toe game started. You play as {human_mark}.", 3000)

    def on_reset_requested(self, seed: int) -> None:
        """Handle reset game request.

        Args:
            seed: New random seed for reset.
        """
        if self._controller is not None:
            self._controller.reset_game(seed=seed)
            self._status_bar.showMessage(f"Tic-Tac-Toe game reset with seed={seed}", 3000)
        else:
            self._status_bar.showMessage("No active game to reset", 3000)

    # -------------------------------------------------------------------------
    # Internal Event Handlers
    # -------------------------------------------------------------------------

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Handle cell click from the board renderer.

        Args:
            row: Row index clicked (0-2)
            col: Column index clicked (0-2)
        """
        if self._controller is not None and self._controller.is_human_turn():
            self._controller.submit_human_move(row, col)

    def _on_state_changed(self, state: TicTacToeState) -> None:
        """Handle game state update from controller.

        Args:
            state: New Tic-Tac-Toe game state.
        """
        if self._renderer_strategy is None:
            return

        # Update board via payload (include human_player for win highlight colors)
        payload = state.to_dict()
        if self._controller is not None:
            payload["human_player"] = self._controller._human_player
        self._renderer_strategy._widget.render_game(GameId.TIC_TAC_TOE, payload)

        # Update Human vs Agent tab status
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent

        player_display = "X" if state.current_player == "player_1" else "O"
        turn_text = f"{player_display}'s turn"

        score_text = f"Move {state.move_count}"

        result = None
        if state.is_game_over:
            if state.winner == "draw":
                result = "Draw"
            elif state.winner:
                winner_mark = "X" if state.winner == "player_1" else "O"
                result = f"{winner_mark} wins!"

        human_vs_agent_tab.update_game_status(
            current_turn=turn_text,
            score=score_text,
            result=result,
        )

    def _on_game_started(self) -> None:
        """Handle game start event."""
        _LOG.info("Tic-Tac-Toe game started")
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent
        human_vs_agent_tab._start_btn.setEnabled(True)
        human_vs_agent_tab._reset_btn.setEnabled(True)

    def _on_game_over(self, winner: str) -> None:
        """Handle game end event.

        Args:
            winner: "player_1", "player_2", or "draw"
        """
        _LOG.info(f"Tic-Tac-Toe game over: winner={winner}")


__all__ = ["TicTacToeEnvLoader"]
