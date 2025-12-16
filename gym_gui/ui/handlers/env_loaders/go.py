"""Go environment loader for Human vs Agent mode.

This loader handles:
- Go game initialization with interactive board
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

from gym_gui.controllers.go_controller import GoGameController
from gym_gui.core.adapters.go_adapter import GoState
from gym_gui.core.enums import GameId
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

_LOG = logging.getLogger(__name__)


class GoEnvLoader:
    """Loader for Human vs Agent Go games.

    This class encapsulates all Go-specific loading and lifecycle management,
    keeping MainWindow clean of environment-specific code.

    Uses the existing BoardGameRendererStrategy for rendering, connecting its signals
    to the game controller for move handling.

    Args:
        render_tabs: The render tabs widget for displaying the board.
        control_panel: The control panel for accessing game configuration.
        status_bar: The status bar for showing feedback messages.
    """

    # Default Go settings
    DEFAULT_BOARD_SIZE = 19
    DEFAULT_KOMI = 7.5

    def __init__(
        self,
        render_tabs: "RenderTabs",
        control_panel: "ControlPanelWidget",
        status_bar: "QStatusBar",
        board_size: int = DEFAULT_BOARD_SIZE,
        komi: float = DEFAULT_KOMI,
    ) -> None:
        self._render_tabs = render_tabs
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._board_size = board_size
        self._komi = komi

        # Game components (created on load)
        self._controller: Optional[GoGameController] = None
        self._renderer_strategy: Optional[BoardGameRendererStrategy] = None
        self._tab_index: int = -1
        self._signal_connected: bool = False

    @property
    def controller(self) -> Optional[GoGameController]:
        """The active Go game controller, if any."""
        return self._controller

    @property
    def is_loaded(self) -> bool:
        """Whether a Go game is currently loaded."""
        return self._controller is not None

    def load(self, seed: int, parent: Optional[QtCore.QObject] = None) -> str:
        """Load and initialize the Go game with interactive board.

        Args:
            seed: Random seed for game initialization.
            parent: Parent QObject for GoGameController (usually MainWindow).

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
        self._controller = GoGameController(
            parent,
            board_size=self._board_size,
            komi=self._komi,
        )

        # Connect controller signals
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.game_started.connect(self._on_game_started)
        self._controller.game_over.connect(self._on_game_over)
        self._controller.status_message.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._controller.error_occurred.connect(
            lambda msg: self._status_bar.showMessage(f"Go error: {msg}", 5000)
        )

        # Connect renderer signals to controller for human moves
        self._renderer_strategy.go_intersection_clicked.connect(
            self._on_intersection_clicked
        )
        self._renderer_strategy.go_pass_requested.connect(
            self._on_pass_requested
        )
        self._signal_connected = True

        # Add board tab with proper Human vs Agent naming
        self._tab_index = self._render_tabs.addTab(
            self._renderer_strategy.widget, "Human vs Agent - Go"
        )
        self._render_tabs.setCurrentIndex(self._tab_index)

        _LOG.info(
            f"Go board added: tab_index={self._tab_index}, "
            f"board_size={self._board_size}, "
            f"widget_visible={self._renderer_strategy.widget.isVisible()}"
        )

        # Get human player selection
        human_agent = human_vs_agent_tab._human_player_combo.currentData()
        human_player = "black_0" if human_agent == "player_0" else "white_0"

        # AI is Random for Go (no strong engine like Stockfish available by default)
        ai_name = "Random AI"

        # Start the game
        self._controller.start_game(human_player=human_player, seed=seed)

        # Update control panel state
        human_vs_agent_tab.set_environment_loaded("go_v5", seed)
        human_vs_agent_tab.set_policy_loaded(ai_name)
        human_vs_agent_tab._reset_btn.setEnabled(True)

        human_color = "Black" if human_player == "black_0" else "White"
        self._status_bar.showMessage(
            f"Go loaded ({self._board_size}x{self._board_size}, komi={self._komi}, seed={seed}). "
            f"You play as {human_color} vs {ai_name}. Click intersection to place stone.",
            5000
        )

        return ai_name

    def cleanup(self) -> None:
        """Clean up Go game resources."""
        # Disconnect signals
        if self._signal_connected and self._renderer_strategy is not None:
            try:
                self._renderer_strategy.go_intersection_clicked.disconnect(
                    self._on_intersection_clicked
                )
                self._renderer_strategy.go_pass_requested.disconnect(
                    self._on_pass_requested
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
            human_agent: Which agent the human plays ("player_0"/"black_0" or "player_1"/"white_0").
            seed: Random seed.
        """
        if self._controller is not None:
            # Normalize agent name to Go convention
            human_player = "black_0" if human_agent in ("player_0", "black_0") else "white_0"
            human_color = "Black" if human_player == "black_0" else "White"
            self._controller.start_game(human_player=human_player, seed=seed)
            self._status_bar.showMessage(f"Go game started. You play as {human_color}.", 3000)

    def on_reset_requested(self, seed: int) -> None:
        """Handle reset game request.

        Args:
            seed: New random seed for reset.
        """
        if self._controller is not None:
            self._controller.reset_game(seed=seed)
            self._status_bar.showMessage(f"Go game reset with seed={seed}", 3000)
        else:
            self._status_bar.showMessage("No active game to reset", 3000)

    # -------------------------------------------------------------------------
    # Internal Event Handlers
    # -------------------------------------------------------------------------

    def _on_intersection_clicked(self, row: int, col: int) -> None:
        """Handle intersection click from the board renderer.

        Args:
            row: Row index clicked
            col: Column index clicked
        """
        if self._controller is not None and self._controller.is_human_turn():
            self._controller.submit_human_move(row, col)

    def _on_pass_requested(self) -> None:
        """Handle pass request from the board renderer."""
        if self._controller is not None and self._controller.is_human_turn():
            self._controller.submit_pass()

    def _on_state_changed(self, state: GoState) -> None:
        """Handle game state update from controller.

        Args:
            state: New Go game state.
        """
        if self._renderer_strategy is None:
            return

        # Update board via payload
        payload = state.to_dict()
        self._renderer_strategy._widget.render_game(GameId.GO, payload)

        # Update Human vs Agent tab status
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent

        player_display = "Black" if state.current_player == "black_0" else "White"
        turn_text = f"{player_display}'s turn"
        if state.last_was_pass:
            turn_text += " (opponent passed)"

        score_text = f"Move {state.move_count}"
        if state.black_captures > 0 or state.white_captures > 0:
            score_text += f" | Captures: B={state.black_captures} W={state.white_captures}"

        result = None
        if state.is_game_over:
            if state.winner == "draw":
                result = "Draw"
            elif state.winner:
                winner_color = "Black" if state.winner == "black_0" else "White"
                result = f"{winner_color} wins!"

        human_vs_agent_tab.update_game_status(
            current_turn=turn_text,
            score=score_text,
            result=result,
        )

    def _on_game_started(self) -> None:
        """Handle game start event."""
        _LOG.info("Go game started")
        human_vs_agent_tab = self._control_panel.multi_agent_tab.human_vs_agent
        human_vs_agent_tab._start_btn.setEnabled(True)
        human_vs_agent_tab._reset_btn.setEnabled(True)

    def _on_game_over(self, winner: str) -> None:
        """Handle game end event.

        Args:
            winner: "black_0", "white_0", or "draw"
        """
        _LOG.info(f"Go game over: winner={winner}")


__all__ = ["GoEnvLoader"]
