"""Chess-specific adapter for PettingZoo chess_v6 environment.

This adapter provides a state-based interface for Chess, enabling:
- FEN string for Qt-based board rendering
- Legal move validation
- UCI move execution
- Turn management

The adapter uses the Parallel API wrapper internally for consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gym_gui.logging_config.log_constants import (
    LOG_UI_MULTI_AGENT_ENV_LOADED,
    LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class ChessState:
    """Structured chess game state for UI rendering."""

    fen: str
    current_player: str  # "white" or "black"
    current_agent: str  # "player_0" or "player_1"
    legal_moves: List[str]  # UCI notation: ["e2e4", "d2d4", ...]
    last_move: Optional[str] = None
    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    is_game_over: bool = False
    winner: Optional[str] = None  # "white", "black", or "draw"
    move_count: int = 0
    captured_pieces: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal emission."""
        return {
            "fen": self.fen,
            "current_player": self.current_player,
            "current_agent": self.current_agent,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
            "captured_pieces": self.captured_pieces,
        }


class ChessAdapter:
    """Adapter for PettingZoo Chess environment with state-based interface.

    This adapter wraps the PettingZoo chess_v6 environment and provides:
    - State-based interface (FEN + legal moves) for Qt rendering
    - Move execution via UCI notation
    - Automatic AEC→Parallel conversion

    Example usage:
        adapter = ChessAdapter()
        adapter.load(seed=42)

        state = adapter.get_state()
        # state.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # state.legal_moves = ["e2e4", "d2d4", ...]

        new_state = adapter.make_move("e2e4")
        # new_state.fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    """

    # Agent mapping
    WHITE_AGENT = "player_0"
    BLACK_AGENT = "player_1"

    def __init__(self) -> None:
        self._aec_env: Any = None
        self._board: Any = None  # chess.Board reference
        self._last_move: Optional[str] = None
        self._move_count: int = 0
        self._captured_white: List[str] = []  # Pieces captured by white
        self._captured_black: List[str] = []  # Pieces captured by black
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if environment is loaded."""
        return self._is_loaded

    def load(self, seed: int = 42, render_mode: str = "rgb_array") -> ChessState:
        """Load and reset the chess environment.

        Args:
            seed: Random seed for environment reset
            render_mode: Render mode ("rgb_array" for pixel output)

        Returns:
            Initial chess state

        Note:
            Chess uses AEC API directly (not parallelizable) since it's
            strictly turn-based. We work with the underlying chess.Board
            for state and move execution.
        """
        try:
            from pettingzoo.classic import chess_v6

            # Create AEC environment
            self._aec_env = chess_v6.env(render_mode=render_mode)

            # Reset environment
            self._aec_env.reset(seed=seed)

            # Get reference to underlying chess board
            self._board = self._aec_env.unwrapped.board

            # Reset tracking
            self._last_move = None
            self._move_count = 0
            self._captured_white = []
            self._captured_black = []
            self._is_loaded = True

            _LOG.info(
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.message} | "
                f"env_id=chess_v6 seed={seed}"
            )

            return self.get_state()

        except Exception as e:
            _LOG.error(
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.message} | "
                f"env_id=chess_v6 error={e}"
            )
            raise

    def get_state(self) -> ChessState:
        """Get current chess game state.

        Returns:
            ChessState with FEN, legal moves, game status, etc.
        """
        if not self._is_loaded or self._board is None:
            raise RuntimeError("Chess environment not loaded. Call load() first.")

        # Determine current player
        is_white_turn = self._board.turn  # True = white, False = black
        current_player = "white" if is_white_turn else "black"
        current_agent = self.WHITE_AGENT if is_white_turn else self.BLACK_AGENT

        # Get legal moves in UCI notation
        legal_moves = [str(move) for move in self._board.legal_moves]

        # Check game status
        is_checkmate = self._board.is_checkmate()
        is_stalemate = self._board.is_stalemate()
        is_game_over = self._board.is_game_over()

        # Determine winner
        winner = None
        if is_checkmate:
            # The player who just moved wins (not the current player)
            winner = "black" if is_white_turn else "white"
        elif is_stalemate or (is_game_over and not is_checkmate):
            winner = "draw"

        return ChessState(
            fen=self._board.fen(),
            current_player=current_player,
            current_agent=current_agent,
            legal_moves=legal_moves,
            last_move=self._last_move,
            is_check=self._board.is_check(),
            is_checkmate=is_checkmate,
            is_stalemate=is_stalemate,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
            captured_pieces={
                "white": self._captured_white.copy(),
                "black": self._captured_black.copy(),
            },
        )

    def make_move(self, uci_move: str) -> ChessState:
        """Execute a move in UCI notation.

        Args:
            uci_move: Move in UCI notation (e.g., "e2e4", "e7e8q" for promotion)

        Returns:
            New chess state after the move

        Raises:
            ValueError: If move is invalid
            RuntimeError: If environment not loaded
        """
        if not self._is_loaded or self._board is None:
            raise RuntimeError("Chess environment not loaded. Call load() first.")

        import chess

        # Parse and validate move
        try:
            move = chess.Move.from_uci(uci_move)
        except ValueError as e:
            raise ValueError(f"Invalid UCI move format: {uci_move}") from e

        if move not in self._board.legal_moves:
            raise ValueError(
                f"Illegal move: {uci_move}. "
                f"Legal moves: {[str(m) for m in self._board.legal_moves]}"
            )

        # Track captured piece before move
        captured = self._board.piece_at(move.to_square)
        if captured:
            piece_symbol = captured.symbol()
            if self._board.turn:  # White is moving, so white captured
                self._captured_white.append(piece_symbol.lower())
            else:  # Black is moving
                self._captured_black.append(piece_symbol.upper())

        # Convert move to PettingZoo action index
        action = self._move_to_action(move)

        # Step environment using AEC API
        try:
            self._aec_env.step(action)
        except Exception as e:
            _LOG.warning(f"PettingZoo step error (expected at game end): {e}")

        # Update tracking
        self._last_move = uci_move
        self._move_count += 1

        return self.get_state()

    def _move_to_action(self, move: Any) -> int:
        """Convert chess.Move to PettingZoo action index.

        PettingZoo chess uses a specific action encoding that depends on
        the current player's perspective. We use action_to_move in reverse
        by searching through the action mask.

        Args:
            move: chess.Move object

        Returns:
            PettingZoo action index

        Raises:
            ValueError: If move cannot be found in legal actions
        """
        from pettingzoo.classic.chess import chess_utils
        import numpy as np

        # Get current observation with action mask
        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]

        # Determine current player (0=white, 1=black)
        player = 0 if self._board.turn else 1

        # Find the action that corresponds to this move
        legal_indices = np.where(action_mask == 1)[0]
        move_uci = str(move)

        for action_idx in legal_indices:
            decoded_move = chess_utils.action_to_move(self._board, action_idx, player)
            if str(decoded_move) == move_uci:
                return int(action_idx)

        raise ValueError(
            f"Could not find action index for move: {move}. "
            f"Legal actions: {len(legal_indices)}"
        )

    def is_move_legal(self, uci_move: str) -> bool:
        """Check if a move is legal.

        Args:
            uci_move: Move in UCI notation

        Returns:
            True if move is legal
        """
        if not self._is_loaded or self._board is None:
            return False

        import chess

        try:
            move = chess.Move.from_uci(uci_move)
            return move in self._board.legal_moves
        except ValueError:
            return False

    def get_legal_moves_from_square(self, square: str) -> List[str]:
        """Get all legal moves from a specific square.

        Args:
            square: Square in algebraic notation (e.g., "e2")

        Returns:
            List of destination squares in algebraic notation
        """
        if not self._is_loaded or self._board is None:
            return []

        import chess

        try:
            from_square = chess.parse_square(square)
        except ValueError:
            return []

        destinations = []
        for move in self._board.legal_moves:
            if move.from_square == from_square:
                to_square = chess.square_name(move.to_square)
                destinations.append(to_square)

        return destinations

    def render(self) -> Optional[np.ndarray]:
        """Get pixel render of current board state.

        Returns:
            RGB numpy array (H, W, 3) or None if render fails
        """
        if not self._is_loaded or self._aec_env is None:
            return None

        try:
            return self._aec_env.render()
        except Exception as e:
            _LOG.warning(f"Render failed: {e}")
            return None

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass

        self._aec_env = None
        self._board = None
        self._is_loaded = False

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.close()


__all__ = ["ChessAdapter", "ChessState"]
