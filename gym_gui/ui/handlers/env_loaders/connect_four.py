"""Connect Four environment loader for Human vs Agent mode.

This loader handles:
- Connect Four game initialization with interactive board
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

from gym_gui.controllers.connect_four_controller import ConnectFourGameController
from gym_gui.core.adapters.connect_four_adapter import ConnectFourState
from gym_gui.core.enums import GameId
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

_LOG = logging.getLogger(__name__)


class ConnectFourEnvLoader:
    """Loader for Human vs Agent Connect Four games.

    This class encapsulates all Connect Four-specific loading and lifecycle management,
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
        self._controller: Optional[ConnectFourGameController] = None
        self._renderer_strategy: Optional[BoardGameRendererStrategy] = None
        self._tab_index: int = -1
        self._signal_connected: bool = False

    @property
    def controller(self) -> Optional[ConnectFourGameController]:
        """The active Connect Four game controller, if any."""
        return self._controller

    @property
    def is_loaded(self) -> bool:
        """Whether a Connect Four game is currently loaded."""
        return self._controller is not None

    def load(self, seed: int, parent: Optional[QtCore.QObject] = None) -> str:
        """Load and initialize the Connect Four game with interactive board.

        Args:
            seed: Random seed for game initialization.
            parent: Parent QObject for ConnectFourGameController (usually MainWindow).

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
        self._controller = ConnectFourGameController(parent)

        # Connect controller signals
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.game_started.connect(self._on_game_started)
        self._controller.game_over.connect(self._on_game_over)
        self._controller.status_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._controller.error_occurred.connect(
            lambda msg: self._status_bar.showMessage(f"Connect Four error: {msg}", 5000)
        )

        # Connect renderer signal to controller for human moves
        self._renderer_strategy.connect_four_column_clicked.connect(
            self._on_column_clicked
        )
        self._signal_connected = True

        # Add board tab with proper Human vs Agent naming
        self._tab_index = self._render_tabs.addTab(
            self._renderer_strategy.widget, "Human vs Agent - Connect Four"
        )
        self._render_tabs.setCurrentIndex(self._tab_index)

        _LOG.info(
            f"Connect Four board added: tab_index={self._tab_index}, "
            f"widget_visible={self._renderer_strategy.widget.isVisible()}"
        )

        # Get human player selection
        human_agent = human_vs_agent_tab._human_player_combo.currentData()
        human_player = "player_0" if human_agent == "player_0" else "player_1"

        # AI is Random for Connect Four (no Stockfish-like engine)
        ai_name = "Random AI"

        # Start the game
        self._controller.start_game(human_player=human_player, seed=seed)

        # Update control panel state
        human_vs_agent_tab.set_environment_loaded("connect_four_v3", seed)
        human_vs_agent_tab.set_policy_loaded(ai_name)
        human_vs_agent_tab._reset_btn.setEnabled(True)

        human_color = "Red" if human_player == "player_0" else "Yellow"
        self._status_bar.showMessage(
            f"Connect Four loaded (seed={seed}). You play as {human_color} vs {ai_name}. Click column to drop piece.",
            5000
        )

        return ai_name

    def cleanup(self) -> None:
        """Clean up Connect Four game resources."""
        # Disconnect signals
        if self._signal_connected and self._renderer_strategy is not None:
            try:
                self._renderer_strategy.connect_four_column_clicked.disconnect(
                    self._on_column_clicked
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
            human_agent: Which agent the human plays ("player_0" or "player_1").
            seed: Random seed.
        """
        if self._controller is not None:
            human_player = "player_0" if human_agent == "player_0" else "player_1"
            human_color = "Red" if human_player == "player_0" else "Yellow"
            self._controller.start_game(human_player=human_player, seed=seed)
            self._status_bar.showMessage(f"Connect Four game started. You play as {human_color}.", 3000)

    def on_reset_requested(self, seed: int) -> None:
        """Handle reset game request.

        Args:
            seed: New random seed for reset.
        """
        if self._controller is not None:
            self._controller.reset_game(seed=seed)
            self._status_bar.showMessage(f"Connect Four game reset with seed={seed}", 3000)
        else:
            self._status_bar.showMessage("No active game to reset", 3000)

    # -------------------------------------------------------------------------
    # Internal Event Handlers
    # -------------------------------------------------------------------------

    def _on_column_clicked(self, column: int) -> None:
        """Handle column click from the board renderer.

        Args:
            column: Column index clicked (0-6)
        """
        if self._controller is not None and self._controller.is_human_turn():
            self._controller.submit_human_move(column)

    def _on_state_changed(self, state: ConnectFourState) -> None:
        """Handle game state update from controller.

        Args:
            state: New Connect Four game state.
        """
        if self._renderer_strategy is None:
            return

        # Update board via payload
        payload = state.to_dict()
        self._renderer_strategy._widget.render_game(GameId.CONNECT_FOUR, payload)

        # Update Human vs Agent tab status
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent

        player_display = "Red" if state.current_player == "player_0" else "Yellow"
        turn_text = f"{player_display}'s turn"

        score_text = f"Move {state.move_count}"

        result = None
        if state.is_game_over:
            if state.winner == "draw":
                result = "Draw"
            elif state.winner:
                winner_color = "Red" if state.winner == "player_0" else "Yellow"
                result = f"{winner_color} wins!"

        human_vs_agent_tab.update_game_status(
            current_turn=turn_text,
            score=score_text,
            result=result,
        )

    def _on_game_started(self) -> None:
        """Handle game start event."""
        _LOG.info("Connect Four game started")
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent
        human_vs_agent_tab._start_btn.setEnabled(True)
        human_vs_agent_tab._reset_btn.setEnabled(True)

    def _on_game_over(self, winner: str) -> None:
        """Handle game end event.

        Args:
            winner: "player_0", "player_1", or "draw"
        """
        _LOG.info(f"Connect Four game over: winner={winner}")


__all__ = ["ConnectFourEnvLoader"]
