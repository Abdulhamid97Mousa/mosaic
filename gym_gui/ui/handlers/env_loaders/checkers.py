"""Checkers environment loader for Human vs Agent mode.

This loader handles:
- Checkers game initialization with interactive board
- AI opponent setup (Random for now)
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

from gym_gui.controllers.checkers_controller import CheckersGameController
from gym_gui.core.adapters.checkers_adapter import CheckersState
from gym_gui.core.enums import GameId
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

_LOG = logging.getLogger(__name__)


class CheckersEnvLoader:
    """Loader for Human vs Agent Checkers games.

    This class encapsulates all Checkers-specific loading and lifecycle management,
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
        self._controller: Optional[CheckersGameController] = None
        self._renderer_strategy: Optional[BoardGameRendererStrategy] = None
        self._tab_index: int = -1
        self._signal_connected: bool = False

        # AI settings
        self._current_ai_name: str = "Random AI"
        self._is_fallback: bool = False

    @property
    def controller(self) -> Optional[CheckersGameController]:
        """The active Checkers game controller, if any."""
        return self._controller

    @property
    def is_loaded(self) -> bool:
        """Whether a Checkers game is currently loaded."""
        return self._controller is not None

    def load(self, seed: int, parent: Optional[QtCore.QObject] = None) -> str:
        """Load and initialize the Checkers game with interactive board.

        Args:
            seed: Random seed for game initialization.
            parent: Parent QObject for CheckersGameController (usually MainWindow).

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
        self._controller = CheckersGameController(parent)

        # Connect controller signals
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.game_started.connect(self._on_game_started)
        self._controller.game_over.connect(self._on_game_over)
        self._controller.status_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._controller.error_occurred.connect(
            lambda msg: self._status_bar.showMessage(f"Checkers error: {msg}", 5000)
        )

        # Connect renderer signals to controller for human moves
        self._renderer_strategy.checkers_cell_clicked.connect(
            self._on_cell_clicked
        )
        self._signal_connected = True

        # Add board tab with proper Human vs Agent naming
        self._tab_index = self._render_tabs.addTab(
            self._renderer_strategy.widget, "Human vs Agent - Checkers"
        )
        self._render_tabs.setCurrentIndex(self._tab_index)

        _LOG.info(
            f"Checkers board added: tab_index={self._tab_index}, "
            f"widget_visible={self._renderer_strategy.widget.isVisible()}"
        )

        # Get human player selection
        human_agent = human_vs_agent_tab._human_player_combo.currentData()
        human_player = "player_0" if human_agent == "player_0" else "player_1"

        # For now, use Random AI (no dedicated Checkers engine like Stockfish/KataGo)
        ai_name = "Random AI"
        self._current_ai_name = ai_name
        self._is_fallback = False

        # Start the game
        self._controller.start_game(human_player=human_player, seed=seed)

        # Update control panel state
        human_vs_agent_tab.set_environment_loaded("checkers", seed)
        human_vs_agent_tab.set_policy_loaded(ai_name)
        human_vs_agent_tab.set_active_ai(ai_name, self._is_fallback)
        human_vs_agent_tab._reset_btn.setEnabled(True)

        human_color = "Black" if human_player == "player_0" else "White"
        self._status_bar.showMessage(
            f"Checkers loaded (seed={seed}). "
            f"You play as {human_color} vs {ai_name}. Click piece to select, then click destination.",
            5000
        )

        return ai_name

    def cleanup(self) -> None:
        """Clean up Checkers game resources."""
        # Disconnect signals
        if self._signal_connected and self._renderer_strategy is not None:
            try:
                self._renderer_strategy.checkers_cell_clicked.disconnect(
                    self._on_cell_clicked
                )
            except Exception:
                pass
            self._signal_connected = False

        # Close controller
        if self._controller is not None:
            self._controller.close()
            self._controller = None

        # Remove tab
        if self._renderer_strategy is not None and self._tab_index >= 0:
            try:
                self._render_tabs.removeTab(self._tab_index)
            except Exception:
                pass
            self._renderer_strategy.cleanup()
            self._renderer_strategy = None
            self._tab_index = -1

        _LOG.info("Checkers game cleaned up")

    def reset(self, seed: int) -> None:
        """Reset the current Checkers game.

        Args:
            seed: New random seed for the game.
        """
        if self._controller is not None:
            self._controller.reset_game(seed=seed)
            human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent
            human_vs_agent_tab.set_environment_loaded("checkers", seed)
            self._status_bar.showMessage(f"Checkers reset with seed={seed}", 3000)

    # -------------------------------------------------------------------------
    # Internal Event Handlers
    # -------------------------------------------------------------------------

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Handle cell click from the board renderer.

        Args:
            row: Row index clicked
            col: Column index clicked
        """
        if self._controller is not None:
            self._controller.handle_cell_click(row, col)

    def _on_state_changed(self, state: CheckersState) -> None:
        """Handle game state update from controller.

        Args:
            state: New Checkers game state.
        """
        if self._renderer_strategy is None:
            return

        # Update board via payload
        payload = state.to_dict()
        self._renderer_strategy._widget.render_game(GameId.OPEN_SPIEL_CHECKERS, payload)

        # Update Human vs Agent tab status
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent

        player_display = "Black" if state.current_player == "player_0" else "White"
        turn_text = f"{player_display}'s turn"

        score_text = f"Move {state.move_count}"

        result = None
        if state.is_game_over:
            if state.winner == "draw":
                result = "Draw"
            elif state.winner:
                winner_color = "Black" if state.winner == "player_0" else "White"
                result = f"{winner_color} wins!"

        human_vs_agent_tab.update_game_status(
            current_turn=turn_text,
            score=score_text,
            result=result,
        )

    def _on_game_started(self) -> None:
        """Handle game start event."""
        _LOG.info("Checkers game started")
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent
        human_vs_agent_tab._start_btn.setEnabled(True)
        human_vs_agent_tab._reset_btn.setEnabled(True)

    def _on_game_over(self, winner: str) -> None:
        """Handle game end event.

        Args:
            winner: "player_0", "player_1", or "draw"
        """
        _LOG.info(f"Checkers game over: winner={winner}")


__all__ = ["CheckersEnvLoader"]
