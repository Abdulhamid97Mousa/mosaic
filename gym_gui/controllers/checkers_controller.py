"""Checkers game controller for Human vs Agent gameplay.

This controller manages the turn-based flow between human and AI players
in OpenSpiel Checkers, coordinating the adapter with the board renderer.
"""

from __future__ import annotations

import logging
import random
from enum import Enum, auto
from typing import Callable, List, Optional

from qtpy import QtCore
from qtpy.QtCore import Signal as pyqtSignal

from gym_gui.core.adapters.checkers_adapter import CheckersAdapter, CheckersState

_LOG = logging.getLogger(__name__)


class PlayerType(Enum):
    """Type of player controlling an agent."""
    HUMAN = auto()
    AI = auto()


class CheckersGameController(QtCore.QObject):
    """Controller for Checkers Human vs Agent gameplay.

    This controller manages:
    - Turn-based flow between human and AI players
    - Move validation and execution
    - Game state synchronization with the UI
    - AI move scheduling (non-blocking)

    Signals:
        state_changed(CheckersState): Emitted when game state changes
        game_started(): Emitted when a new game starts
        game_over(str): Emitted when game ends (winner: "player_0"/"player_1"/"draw")
        move_made(int): Emitted when any move is made (action index)
        error_occurred(str): Emitted on errors
        awaiting_human(bool): Emitted when waiting for human input
        ai_thinking(bool): Emitted when AI is computing
        status_message(str): Emitted for status updates

    Usage:
        controller = CheckersGameController()
        controller.state_changed.connect(board.update_from_state)
        controller.start_game(human_player="player_0")

        # From UI cell click (needs move selection logic):
        controller.submit_human_move(action_index)
    """

    # Signals
    state_changed = pyqtSignal(object)  # CheckersState
    game_started = pyqtSignal()
    game_over = pyqtSignal(str)  # winner
    move_made = pyqtSignal(int)  # action index
    error_occurred = pyqtSignal(str)
    awaiting_human = pyqtSignal(bool)
    ai_thinking = pyqtSignal(bool)
    status_message = pyqtSignal(str)

    # AI delay to make moves feel natural (ms)
    AI_MOVE_DELAY_MS = 500

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
        ai_action_provider: Optional[Callable[[CheckersState], Optional[int]]] = None,
    ) -> None:
        """Initialize the Checkers game controller.

        Args:
            parent: Parent QObject for memory management
            ai_action_provider: Optional function that returns an AI move given state.
                               If None, uses random legal moves.
        """
        super().__init__(parent)

        self._adapter: Optional[CheckersAdapter] = None
        self._current_state: Optional[CheckersState] = None
        self._human_player: str = "player_0"  # "player_0" or "player_1"
        self._game_active: bool = False
        self._ai_action_provider = ai_action_provider

        # For handling piece selection (checkers requires from->to)
        self._selected_piece: Optional[tuple[int, int]] = None
        self._valid_destinations: List[int] = []

        # Timer for AI moves (non-blocking)
        self._ai_timer = QtCore.QTimer(self)
        self._ai_timer.setSingleShot(True)
        self._ai_timer.timeout.connect(self._execute_ai_move)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_game(
        self,
        human_player: str = "player_0",
        seed: int = 42,
    ) -> None:
        """Start a new Checkers game.

        Args:
            human_player: Which player the human plays ("player_0" or "player_1")
                         player_0 (Black) goes first
            seed: Random seed for environment reset
        """
        _LOG.info(f"Starting Checkers game: human={human_player}, seed={seed}")

        # Create and load adapter
        self._adapter = CheckersAdapter()
        try:
            initial_state = self._adapter.load(seed=seed)
        except Exception as e:
            _LOG.error(f"Failed to load Checkers environment: {e}")
            self.error_occurred.emit(f"Failed to load Checkers: {e}")
            return

        self._human_player = human_player
        self._game_active = True
        self._current_state = initial_state
        self._selected_piece = None
        self._valid_destinations = []

        self.game_started.emit()
        self.state_changed.emit(initial_state)

        human_color = "Black" if human_player == "player_0" else "White"
        self.status_message.emit(f"Checkers game started. You play as {human_color}.")

        # If human is player_1 (white), AI (black) goes first
        if human_player == "player_1":
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

    def submit_human_move(self, action: int) -> bool:
        """Submit a move from the human player.

        Args:
            action: Action index from legal_moves

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
        if not self._adapter.is_move_legal(action):
            self.status_message.emit(f"Invalid move!")
            return False

        return self._execute_move(action)

    def handle_cell_click(self, row: int, col: int) -> None:
        """Handle a cell click from the board renderer.

        In checkers, we need two clicks: select piece, then select destination.
        This method handles the selection logic.

        Args:
            row: Row index clicked (0-7)
            col: Column index clicked (0-7)
        """
        if not self._game_active or self._adapter is None or self._current_state is None:
            return

        if self._current_state.current_player != self._human_player:
            self.status_message.emit("Not your turn!")
            return

        # Get the board value at clicked position
        piece = self._current_state.board[row][col]

        # Determine which pieces belong to the human player
        if self._human_player == "player_0":
            human_pieces = {1, 2}  # Black pieces
        else:
            human_pieces = {3, 4}  # White pieces

        if self._selected_piece is None:
            # First click - try to select a piece
            if piece in human_pieces:
                self._selected_piece = (row, col)
                # Find valid moves from this position
                self._find_valid_moves_from(row, col)
                if self._valid_destinations:
                    self.status_message.emit(f"Piece selected at ({row+1}, {col+1}). Click destination.")
                else:
                    self._selected_piece = None
                    self.status_message.emit("No valid moves from this piece.")
            else:
                self.status_message.emit("Click on one of your pieces to select it.")
        else:
            # Second click - try to move to destination
            if (row, col) == self._selected_piece:
                # Clicked same piece - deselect
                self._selected_piece = None
                self._valid_destinations = []
                self.status_message.emit("Piece deselected.")
            elif piece in human_pieces:
                # Clicked another of our pieces - select it instead
                self._selected_piece = (row, col)
                self._find_valid_moves_from(row, col)
                if self._valid_destinations:
                    self.status_message.emit(f"Piece selected at ({row+1}, {col+1}). Click destination.")
                else:
                    self._selected_piece = None
                    self.status_message.emit("No valid moves from this piece.")
            else:
                # Try to find a move to this destination
                action = self._find_action_to(row, col)
                if action is not None:
                    self._selected_piece = None
                    self._valid_destinations = []
                    self.submit_human_move(action)
                else:
                    self.status_message.emit("Invalid destination. Try again.")

    def _find_valid_moves_from(self, from_row: int, from_col: int) -> None:
        """Find all valid moves starting from a given position.

        Args:
            from_row: Starting row
            from_col: Starting column
        """
        self._valid_destinations = []

        if self._current_state is None:
            return

        # OpenSpiel checkers encodes moves as action indices
        # We need to decode them to find moves from this position
        # For now, store all legal moves and let the user try
        # A more sophisticated approach would decode the action space
        self._valid_destinations = list(self._current_state.legal_moves)

    def _find_action_to(self, to_row: int, to_col: int) -> Optional[int]:
        """Find an action that moves to the given destination.

        Args:
            to_row: Destination row
            to_col: Destination column

        Returns:
            Action index if found, None otherwise
        """
        if self._current_state is None or self._selected_piece is None:
            return None

        # In OpenSpiel, actions are encoded integers
        # We'll use a simple heuristic: try each legal move
        # and see if it involves the selected piece and destination
        # This is a simplification - a full implementation would
        # decode the action space properly

        # For now, just return the first legal move as a demonstration
        # The user can cycle through moves
        if self._valid_destinations:
            return self._valid_destinations[0]

        return None

    def get_state(self) -> Optional[CheckersState]:
        """Get the current game state."""
        return self._current_state

    def get_legal_moves(self) -> List[int]:
        """Get legal moves.

        Returns:
            List of action indices
        """
        if self._adapter is None:
            return []
        return self._adapter.get_legal_moves()

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
        self, provider: Optional[Callable[[CheckersState], Optional[int]]]
    ) -> None:
        """Set or update the AI action provider.

        Args:
            provider: Function that takes CheckersState and returns action index
        """
        self._ai_action_provider = provider

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _execute_move(self, action: int) -> bool:
        """Execute a move and update game state.

        Args:
            action: Action index

        Returns:
            True if move was executed successfully
        """
        if self._adapter is None:
            return False

        try:
            new_state = self._adapter.make_move(action)
        except ValueError as e:
            self.error_occurred.emit(str(e))
            return False

        self._current_state = new_state

        self.move_made.emit(action)
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

        _LOG.debug(f"AI plays action: {ai_action}")
        self.status_message.emit(f"AI plays move {ai_action}")

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

        # Fallback: random move
        if self._current_state.legal_moves:
            return random.choice(self._current_state.legal_moves)

        return None

    def _handle_game_over(self, state: CheckersState) -> None:
        """Handle end of game.

        Args:
            state: Final game state
        """
        self._game_active = False

        winner = state.winner
        if winner == "draw":
            result_msg = "Game Over: Draw!"
        elif winner == self._human_player:
            human_color = "Black" if self._human_player == "player_0" else "White"
            result_msg = f"Game Over: You win! ({human_color})"
        else:
            ai_color = "White" if self._human_player == "player_0" else "Black"
            result_msg = f"Game Over: AI wins ({ai_color})"

        _LOG.info(result_msg)
        self.status_message.emit(result_msg)
        self.game_over.emit(winner or "draw")


__all__ = ["CheckersGameController", "PlayerType"]
