"""Chess environment loader for Human vs Agent mode.

This loader handles:
- Chess game initialization with interactive board
- Stockfish AI opponent setup
- Game lifecycle management
- Signal connections between components
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget

from gym_gui.controllers.chess_controller import ChessGameController
from gym_gui.ui.widgets.human_vs_agent_board import InteractiveChessBoard
from gym_gui.ui.handlers.features.human_vs_agent import HumanVsAgentHandler

_LOG = logging.getLogger(__name__)


class ChessEnvLoader:
    """Loader for Human vs Agent chess games.

    This class encapsulates all chess-specific loading and lifecycle management,
    keeping MainWindow clean of environment-specific code.

    Args:
        render_tabs: The render tabs widget for displaying the chess board.
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
        self._chess_board: Optional[InteractiveChessBoard] = None
        self._chess_controller: Optional[ChessGameController] = None
        self._human_vs_agent_handler: Optional[HumanVsAgentHandler] = None
        self._chess_tab_index: int = -1

    @property
    def chess_controller(self) -> Optional[ChessGameController]:
        """The active chess game controller, if any."""
        return self._chess_controller

    @property
    def is_loaded(self) -> bool:
        """Whether a chess game is currently loaded."""
        return self._chess_controller is not None

    def load(self, seed: int, parent: object = None) -> str:
        """Load and initialize the Chess game with interactive board.

        Args:
            seed: Random seed for game initialization.
            parent: Parent object for ChessGameController (usually MainWindow).

        Returns:
            AI opponent display name.
        """
        # Clean up existing game
        self.cleanup()

        # Get human vs agent tab reference
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent

        # Create chess board widget
        self._chess_board = InteractiveChessBoard(self._render_tabs)

        # Create chess controller
        self._chess_controller = ChessGameController(parent)

        # Create Human vs Agent handler and bind components
        self._human_vs_agent_handler = HumanVsAgentHandler(self._status_bar)
        self._human_vs_agent_handler.bind_game_components(
            chess_board=self._chess_board,
            chess_controller=self._chess_controller,
            human_vs_agent_tab=human_vs_agent_tab,
        )

        # Connect controller signals to handler
        self._chess_controller.state_changed.connect(
            self._human_vs_agent_handler.on_chess_state_changed
        )
        self._chess_controller.game_started.connect(
            self._human_vs_agent_handler.on_chess_game_started
        )
        self._chess_controller.game_over.connect(
            self._human_vs_agent_handler.on_chess_game_over
        )
        self._chess_controller.status_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._chess_controller.error_occurred.connect(
            lambda msg: self._status_bar.showMessage(f"Chess error: {msg}", 5000)
        )

        # Connect board moves to handler
        self._chess_board.move_made.connect(
            self._human_vs_agent_handler.on_chess_move_made
        )

        # Add chess board tab with proper Human vs Agent naming
        self._chess_tab_index = self._render_tabs.addTab(
            self._chess_board, "Human vs Agent - Chess"
        )
        self._render_tabs.setCurrentIndex(self._chess_tab_index)

        # Debug logging
        _LOG.info(
            f"Chess board added: tab_index={self._chess_tab_index}, "
            f"board_visible={self._chess_board.isVisible()}, "
            f"board_size={self._chess_board.size()}"
        )

        # Get human player selection
        human_agent = human_vs_agent_tab._human_player_combo.currentData()
        human_color = "white" if human_agent == "player_0" else "black"

        # Get AI opponent configuration and set up provider via handler
        ai_config = human_vs_agent_tab.get_ai_config()
        ai_name = self._human_vs_agent_handler.setup_ai_provider(
            ai_config, self._chess_controller
        )

        # Start the game
        self._chess_controller.start_game(human_color=human_color, seed=seed)

        # Update control panel state
        human_vs_agent_tab.set_environment_loaded("chess_v6", seed)
        human_vs_agent_tab.set_policy_loaded(ai_name)
        human_vs_agent_tab._reset_btn.setEnabled(True)

        self._status_bar.showMessage(
            f"Chess loaded (seed={seed}). You play as {human_color} vs {ai_name}. Click board to move.",
            5000
        )

        return ai_name

    def cleanup(self) -> None:
        """Clean up chess game resources."""
        # Clean up controller
        if self._chess_controller is not None:
            self._chess_controller.close()
            self._chess_controller = None

        # Clean up handler
        if self._human_vs_agent_handler is not None:
            self._human_vs_agent_handler.cleanup()
            self._human_vs_agent_handler = None

        # Remove tab
        if self._chess_tab_index >= 0:
            self._render_tabs.removeTab(self._chess_tab_index)
            self._chess_tab_index = -1

        self._chess_board = None

    def on_ai_config_changed(self, opponent_type: str, difficulty: str) -> Optional[str]:
        """Handle AI opponent configuration change.

        Args:
            opponent_type: Type of AI opponent ("random", "stockfish", "custom").
            difficulty: Difficulty level for engines like Stockfish.

        Returns:
            AI display name if updated, None if no game active.
        """
        if self._human_vs_agent_handler is None:
            return None

        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent
        ai_name = self._human_vs_agent_handler.on_ai_config_changed(
            opponent_type,
            difficulty,
            self._chess_controller,
            human_vs_agent_tab.get_ai_config,
        )

        if ai_name:
            human_vs_agent_tab.set_policy_loaded(ai_name)
            self._status_bar.showMessage(f"AI opponent changed to {ai_name}", 3000)

        return ai_name

    def on_start_requested(self, human_agent: str, seed: int) -> None:
        """Handle start game request.

        Args:
            human_agent: Which agent the human plays ("player_0" or "player_1").
            seed: Random seed.
        """
        if self._chess_controller is not None:
            human_color = "white" if human_agent == "player_0" else "black"
            self._chess_controller.start_game(human_color=human_color, seed=seed)
            self._status_bar.showMessage(f"Chess game started. You play as {human_color}.", 3000)

    def on_reset_requested(self, seed: int) -> None:
        """Handle reset game request.

        Args:
            seed: New random seed for reset.
        """
        if self._chess_controller is not None:
            self._chess_controller.reset_game(seed=seed)
            self._status_bar.showMessage(f"Chess game reset with seed={seed}", 3000)
        else:
            self._status_bar.showMessage("No active game to reset", 3000)


__all__ = ["ChessEnvLoader"]
