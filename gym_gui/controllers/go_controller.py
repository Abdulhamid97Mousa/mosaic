"""Go game controller for Human vs Agent gameplay.

This controller manages the turn-based flow between human and AI players
in PettingZoo Go, coordinating the adapter with the board renderer.
"""

from __future__ import annotations

import logging
import random
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from qtpy import QtCore
from qtpy.QtCore import Signal as pyqtSignal

from gym_gui.core.adapters.go_adapter import GoAdapter, GoState

_LOG = logging.getLogger(__name__)


class PlayerType(Enum):
    """Type of player controlling an agent."""
    HUMAN = auto()
    AI = auto()


class GoGameController(QtCore.QObject):
    """Controller for Go Human vs Agent gameplay.

    This controller manages:
    - Turn-based flow between human and AI players
    - Move validation and execution (including pass)
    - Game state synchronization with the UI
    - AI move scheduling (non-blocking)

    Signals:
        state_changed(GoState): Emitted when game state changes
        game_started(): Emitted when a new game starts
        game_over(str): Emitted when game ends (winner: "black_0"/"white_0"/"draw")
        move_made(int, int): Emitted when any move is made (row, col) or (-1, -1) for pass
        error_occurred(str): Emitted on errors
        awaiting_human(bool): Emitted when waiting for human input
        ai_thinking(bool): Emitted when AI is computing
        status_message(str): Emitted for status updates

    Usage:
        controller = GoGameController(board_size=19)
        controller.state_changed.connect(board.update_from_state)
        controller.start_game(human_player="black_0")

        # From UI intersection click:
        controller.submit_human_move(9, 9)  # center of 19x19 board
        # Or pass:
        controller.submit_pass()
    """

    # Signals
    state_changed = pyqtSignal(object)  # GoState
    game_started = pyqtSignal()
    game_over = pyqtSignal(str)  # winner
    move_made = pyqtSignal(int, int)  # row, col (-1, -1 for pass)
    error_occurred = pyqtSignal(str)
    awaiting_human = pyqtSignal(bool)
    ai_thinking = pyqtSignal(bool)
    status_message = pyqtSignal(str)

    # AI delay to make moves feel natural (ms)
    AI_MOVE_DELAY_MS = 500

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
        board_size: int = 19,
        komi: float = 7.5,
        ai_action_provider: Optional[Callable[[GoState], Optional[int]]] = None,
    ) -> None:
        """Initialize the Go game controller.

        Args:
            parent: Parent QObject for memory management
            board_size: Go board size (9, 13, or 19)
            komi: Komi compensation for white
            ai_action_provider: Optional function that returns an AI move given state.
                               If None, uses random legal moves.
        """
        super().__init__(parent)

        self._board_size = board_size
        self._komi = komi
        self._adapter: Optional[GoAdapter] = None
        self._current_state: Optional[GoState] = None
        self._human_player: str = "black_0"  # "black_0" or "white_0"
        self._game_active: bool = False
        self._ai_action_provider = ai_action_provider

        # Timer for AI moves (non-blocking)
        self._ai_timer = QtCore.QTimer(self)
        self._ai_timer.setSingleShot(True)
        self._ai_timer.timeout.connect(self._execute_ai_move)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_game(
        self,
        human_player: str = "black_0",
        seed: int = 42,
    ) -> None:
        """Start a new Go game.

        Args:
            human_player: Which player the human plays ("black_0" or "white_0")
                         black_0 goes first
            seed: Random seed for environment reset
        """
        _LOG.info(f"Starting Go game: human={human_player}, seed={seed}, board={self._board_size}x{self._board_size}")

        # Create and load adapter
        self._adapter = GoAdapter(board_size=self._board_size, komi=self._komi)
        try:
            initial_state = self._adapter.load(seed=seed)
        except Exception as e:
            _LOG.error(f"Failed to load Go environment: {e}")
            self.error_occurred.emit(f"Failed to load Go: {e}")
            return

        self._human_player = human_player
        self._game_active = True
        self._current_state = initial_state

        self.game_started.emit()
        self.state_changed.emit(initial_state)

        human_color = "Black" if human_player == "black_0" else "White"
        self.status_message.emit(f"Go game started ({self._board_size}x{self._board_size}). You play as {human_color}.")

        # If human is white, AI (black) goes first
        if human_player == "white_0":
            self._schedule_ai_move()
        else:
            self.awaiting_human.emit(True)

    def reset_game(self, seed: int = 42) -> None:
        """Reset the game with a new seed.

        Args:
            seed: New random seed
        """
        if self._adapter is None:
            self.error_occurred.emit("No game to reset")
            return

        self.start_game(human_player=self._human_player, seed=seed)

    def submit_human_move(self, row: int, col: int) -> bool:
        """Submit a stone placement from the human player.

        Args:
            row: Row index (0 to board_size-1)
            col: Column index (0 to board_size-1)

        Returns:
            True if move was valid and executed
        """
        if not self._game_active or self._adapter is None:
            self.error_occurred.emit("No active game")
            return False

        if self._current_state is None:
            self.error_occurred.emit("No game state available")
            return False

        # Check if it's human's turn
        if self._current_state.current_player != self._human_player:
            self.status_message.emit("Not your turn!")
            return False

        # Validate move
        if not self._adapter.is_move_legal_at(row, col):
            self.status_message.emit(f"Invalid move at ({row}, {col})!")
            return False

        action = self._adapter.coords_to_action(row, col)
        return self._execute_move(action)

    def submit_pass(self) -> bool:
        """Submit a pass from the human player.

        Returns:
            True if pass was valid and executed
        """
        if not self._game_active or self._adapter is None:
            self.error_occurred.emit("No active game")
            return False

        if self._current_state is None:
            self.error_occurred.emit("No game state available")
            return False

        # Check if it's human's turn
        if self._current_state.current_player != self._human_player:
            self.status_message.emit("Not your turn!")
            return False

        return self._execute_move(self._adapter.pass_action)

    def get_state(self) -> Optional[GoState]:
        """Get the current game state."""
        return self._current_state

    def get_legal_positions(self) -> List[Tuple[int, int]]:
        """Get legal board positions.

        Returns:
            List of (row, col) tuples where stones can be placed
        """
        if self._adapter is None:
            return []
        return self._adapter.get_legal_positions()

    def is_human_turn(self) -> bool:
        """Check if it's the human player's turn."""
        if self._current_state is None:
            return False
        return self._current_state.current_player == self._human_player

    def is_game_active(self) -> bool:
        """Check if a game is currently active."""
        return self._game_active

    def close(self) -> None:
        """Clean up resources."""
        self._ai_timer.stop()
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        self._game_active = False

    def set_ai_action_provider(
        self, provider: Optional[Callable[[GoState], Optional[int]]]
    ) -> None:
        """Set or update the AI action provider.

        Args:
            provider: Function that takes GoState and returns action index
        """
        self._ai_action_provider = provider

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _execute_move(self, action: int) -> bool:
        """Execute a move and update game state.

        Args:
            action: Action index (0 to board_size^2 for placement, board_size^2 for pass)

        Returns:
            True if move was executed successfully
        """
        if self._adapter is None:
            return False

        is_pass = action == self._adapter.pass_action

        try:
            new_state = self._adapter.make_move(action)
        except ValueError as e:
            self.error_occurred.emit(str(e))
            return False

        self._current_state = new_state

        if is_pass:
            self.move_made.emit(-1, -1)  # Signal pass
        else:
            row = action // self._adapter.board_size
            col = action % self._adapter.board_size
            self.move_made.emit(row, col)

        self.state_changed.emit(new_state)

        # Check for game over
        if new_state.is_game_over:
            self._handle_game_over(new_state)
            return True

        # Schedule next turn
        if new_state.current_player != self._human_player:
            self._schedule_ai_move()
        else:
            self.awaiting_human.emit(True)

        return True

    def _schedule_ai_move(self) -> None:
        """Schedule an AI move with a small delay for natural feel."""
        self.awaiting_human.emit(False)
        self.ai_thinking.emit(True)
        self.status_message.emit("AI is thinking...")
        self._ai_timer.start(self.AI_MOVE_DELAY_MS)

    def _execute_ai_move(self) -> None:
        """Execute the AI's move (called by timer)."""
        if not self._game_active or self._adapter is None or self._current_state is None:
            self.ai_thinking.emit(False)
            return

        # Get AI move
        ai_action = self._get_ai_action()

        self.ai_thinking.emit(False)

        if ai_action is None:
            # Fallback: random legal move
            legal_moves = self._current_state.legal_moves
            if legal_moves:
                ai_action = random.choice(legal_moves)
            else:
                _LOG.warning("No legal moves available for AI")
                return

        if ai_action == self._adapter.pass_action:
            _LOG.debug("AI passes")
            self.status_message.emit("AI passes")
        else:
            row = ai_action // self._adapter.board_size
            col = ai_action % self._adapter.board_size
            _LOG.debug(f"AI plays: ({row}, {col})")
            self.status_message.emit(f"AI plays at ({row + 1}, {col + 1})")

        self._execute_move(ai_action)

    def _get_ai_action(self) -> Optional[int]:
        """Get an action from the AI provider or use random."""
        if self._current_state is None:
            return None

        if self._ai_action_provider is not None:
            try:
                return self._ai_action_provider(self._current_state)
            except Exception as e:
                _LOG.warning(f"AI action provider failed: {e}")

        # Fallback: random move (prefer placements over pass)
        if self._current_state.legal_moves:
            # Filter out pass if there are other moves
            non_pass_moves = [
                m for m in self._current_state.legal_moves
                if m < self._adapter.pass_action
            ]
            if non_pass_moves:
                return random.choice(non_pass_moves)
            return random.choice(self._current_state.legal_moves)

        return None

    def _handle_game_over(self, state: GoState) -> None:
        """Handle end of game.

        Args:
            state: Final game state
        """
        self._game_active = False

        winner = state.winner
        if winner == "draw":
            result_msg = "Game Over: Draw!"
        elif winner == self._human_player:
            human_color = "Black" if self._human_player == "black_0" else "White"
            result_msg = f"Game Over: You win! ({human_color})"
        else:
            ai_color = "White" if self._human_player == "black_0" else "Black"
            result_msg = f"Game Over: AI wins ({ai_color})"

        _LOG.info(result_msg)
        self.status_message.emit(result_msg)
        self.game_over.emit(winner or "draw")


__all__ = ["GoGameController", "PlayerType"]
