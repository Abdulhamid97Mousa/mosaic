"""Chess game controller for Human vs Agent gameplay.

This controller manages the turn-based flow between human and AI players
in PettingZoo Chess, coordinating the ChessAdapter with the InteractiveChessBoard.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Callable, Optional

from qtpy import QtCore
from qtpy.QtCore import Signal as pyqtSignal

from gym_gui.core.adapters.chess_adapter import ChessAdapter, ChessState

_LOG = logging.getLogger(__name__)


class PlayerType(Enum):
    """Type of player controlling an agent."""
    HUMAN = auto()
    AI = auto()


class ChessGameController(QtCore.QObject):
    """Controller for Chess Human vs Agent gameplay.

    This controller manages:
    - Turn-based flow between human and AI players
    - Move validation and execution
    - Game state synchronization with the UI
    - AI move scheduling (non-blocking)

    Signals:
        state_changed(ChessState): Emitted when game state changes
        game_started(): Emitted when a new game starts
        game_over(str): Emitted when game ends (winner: "white"/"black"/"draw")
        move_made(str): Emitted when any move is made (UCI notation)
        error_occurred(str): Emitted on errors
        awaiting_human(bool): Emitted when waiting for human input
        ai_thinking(bool): Emitted when AI is computing

    Usage:
        controller = ChessGameController()
        controller.state_changed.connect(board.update_from_state)
        controller.start_game(human_color="white")

        # From UI move input:
        controller.submit_human_move("e2", "e4")
    """

    # Signals
    state_changed = pyqtSignal(object)  # ChessState
    game_started = pyqtSignal()
    game_over = pyqtSignal(str)  # winner
    move_made = pyqtSignal(str)  # uci_move
    error_occurred = pyqtSignal(str)
    awaiting_human = pyqtSignal(bool)
    ai_thinking = pyqtSignal(bool)
    status_message = pyqtSignal(str)

    # AI delay to make moves feel natural (ms)
    AI_MOVE_DELAY_MS = 500

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
        ai_action_provider: Optional[Callable[[ChessState], Optional[str]]] = None,
    ) -> None:
        """Initialize the chess game controller.

        Args:
            parent: Parent QObject for memory management
            ai_action_provider: Optional function that returns an AI move given state.
                               If None, uses random legal moves.
        """
        super().__init__(parent)

        self._adapter: Optional[ChessAdapter] = None
        self._current_state: Optional[ChessState] = None
        self._human_color: str = "white"  # "white" or "black"
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
        human_color: str = "white",
        seed: int = 42,
    ) -> None:
        """Start a new chess game.

        Args:
            human_color: Which color the human plays ("white" or "black")
            seed: Random seed for environment reset
        """
        _LOG.info(f"Starting chess game: human={human_color}, seed={seed}")

        # Create and load adapter
        self._adapter = ChessAdapter()
        try:
            initial_state = self._adapter.load(seed=seed)
        except Exception as e:
            _LOG.error(f"Failed to load chess environment: {e}")
            self.error_occurred.emit(f"Failed to load chess: {e}")
            return

        self._human_color = human_color
        self._game_active = True
        self._current_state = initial_state

        self.game_started.emit()
        self.state_changed.emit(initial_state)
        self.status_message.emit(f"Chess game started. You play as {human_color}.")

        # If human is black, AI goes first
        if human_color == "black":
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

        self.start_game(human_color=self._human_color, seed=seed)

    def submit_human_move(self, from_square: str, to_square: str) -> bool:
        """Submit a move from the human player.

        Args:
            from_square: Source square (e.g., "e2")
            to_square: Destination square (e.g., "e4")

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
        if self._current_state.current_player != self._human_color:
            self.status_message.emit("Not your turn!")
            return False

        # Build UCI move
        uci_move = f"{from_square}{to_square}"

        # Check for pawn promotion
        if self._is_promotion_move(from_square, to_square):
            # Default to queen promotion (UI could prompt for choice)
            uci_move += "q"

        # Validate and execute move
        if not self._adapter.is_move_legal(uci_move):
            self.status_message.emit(f"Illegal move: {uci_move}")
            return False

        return self._execute_move(uci_move)

    def get_state(self) -> Optional[ChessState]:
        """Get the current game state."""
        return self._current_state

    def get_legal_moves_from(self, square: str) -> list[str]:
        """Get legal destination squares for a piece.

        Args:
            square: Source square (e.g., "e2")

        Returns:
            List of destination squares
        """
        if self._adapter is None:
            return []
        return self._adapter.get_legal_moves_from_square(square)

    def is_human_turn(self) -> bool:
        """Check if it's the human player's turn."""
        if self._current_state is None:
            return False
        return self._current_state.current_player == self._human_color

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
        self, provider: Optional[Callable[[ChessState], Optional[str]]]
    ) -> None:
        """Set or update the AI action provider.

        Args:
            provider: Function that takes ChessState and returns UCI move string
        """
        self._ai_action_provider = provider

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _execute_move(self, uci_move: str) -> bool:
        """Execute a move and update game state.

        Args:
            uci_move: Move in UCI notation

        Returns:
            True if move was executed successfully
        """
        if self._adapter is None:
            return False

        try:
            new_state = self._adapter.make_move(uci_move)
        except ValueError as e:
            self.error_occurred.emit(str(e))
            return False

        self._current_state = new_state
        self.move_made.emit(uci_move)
        self.state_changed.emit(new_state)

        # Check for game over
        if new_state.is_game_over:
            self._handle_game_over(new_state)
            return True

        # Schedule next turn
        if new_state.current_player != self._human_color:
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
        ai_move = self._get_ai_action()

        self.ai_thinking.emit(False)

        if ai_move is None:
            # Fallback: random legal move
            legal_moves = self._current_state.legal_moves
            if legal_moves:
                import random
                ai_move = random.choice(legal_moves)
            else:
                _LOG.warning("No legal moves available for AI")
                return

        _LOG.debug(f"AI plays: {ai_move}")
        self.status_message.emit(f"AI plays: {ai_move}")
        self._execute_move(ai_move)

    def _get_ai_action(self) -> Optional[str]:
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
            import random
            return random.choice(self._current_state.legal_moves)

        return None

    def _handle_game_over(self, state: ChessState) -> None:
        """Handle end of game.

        Args:
            state: Final game state
        """
        self._game_active = False

        winner = state.winner
        if winner == "draw":
            result_msg = "Game Over: Draw!"
        elif winner == self._human_color:
            result_msg = f"Game Over: You win! ({self._human_color})"
        else:
            result_msg = f"Game Over: AI wins ({winner})"

        if state.is_checkmate:
            result_msg += " - Checkmate!"
        elif state.is_stalemate:
            result_msg += " - Stalemate!"

        _LOG.info(result_msg)
        self.status_message.emit(result_msg)
        self.game_over.emit(winner or "draw")

    def _is_promotion_move(self, from_square: str, to_square: str) -> bool:
        """Check if a move is a pawn promotion.

        Args:
            from_square: Source square
            to_square: Destination square

        Returns:
            True if this is a pawn promotion move
        """
        if self._current_state is None or self._adapter is None:
            return False

        # Check if piece at from_square is a pawn
        from_row = int(from_square[1])
        to_row = int(to_square[1])

        # White pawn promoting: from row 7 to row 8
        # Black pawn promoting: from row 2 to row 1
        if from_row == 7 and to_row == 8:
            # Check if it's a white pawn
            piece = self._get_piece_at(from_square)
            return piece == "P"
        elif from_row == 2 and to_row == 1:
            # Check if it's a black pawn
            piece = self._get_piece_at(from_square)
            return piece == "p"

        return False

    def _get_piece_at(self, square: str) -> Optional[str]:
        """Get the piece at a given square from FEN.

        Args:
            square: Square in algebraic notation

        Returns:
            Piece character or None
        """
        if self._current_state is None:
            return None

        fen = self._current_state.fen
        position = fen.split()[0]

        col = ord(square[0]) - ord("a")
        row = int(square[1]) - 1

        # Parse FEN position
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


__all__ = ["ChessGameController", "PlayerType"]
