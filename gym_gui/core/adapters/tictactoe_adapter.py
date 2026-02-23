"""Tic-Tac-Toe-specific adapter for PettingZoo tictactoe_v3 environment.

This adapter provides a state-based interface for Tic-Tac-Toe, enabling:
- Board state for Qt-based rendering (3x3 grid)
- Legal cell validation via action masks
- Cell-based move execution
- Turn management

The adapter uses the AEC API directly for turn-based gameplay.

Note: PettingZoo Tic-Tac-Toe uses column-major indexing:
    0 | 3 | 6
    ---------
    1 | 4 | 7
    ---------
    2 | 5 | 8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gym_gui.logging_config.log_constants import (
    LOG_UI_MULTI_AGENT_ENV_LOADED,
    LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class TicTacToeState:
    """Structured Tic-Tac-Toe game state for UI rendering."""

    board: List[List[int]]  # 3x3 grid, 0=empty, 1=player_1 (X), 2=player_2 (O)
    current_player: str  # "player_1" or "player_2"
    current_agent: str  # Same as current_player for Tic-Tac-Toe
    legal_actions: List[int]  # Cell indices where a mark can be placed (0-8)
    last_action: Optional[int] = None  # Last action taken (0-8)
    last_row: Optional[int] = None  # Row of last move
    last_col: Optional[int] = None  # Column of last move
    is_game_over: bool = False
    winner: Optional[str] = None  # "player_1", "player_2", or "draw"
    move_count: int = 0
    winning_positions: Optional[List[Tuple[int, int]]] = None  # 3 winning positions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal emission."""
        return {
            "game_type": "tictactoe",
            "board": self.board,
            "current_player": self.current_player,
            "current_agent": self.current_agent,
            "legal_actions": self.legal_actions,
            "last_action": self.last_action,
            "last_row": self.last_row,
            "last_col": self.last_col,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
            "winning_positions": self.winning_positions,
        }


class TicTacToeAdapter:
    """Adapter for PettingZoo Tic-Tac-Toe environment with state-based interface.

    This adapter wraps the PettingZoo tictactoe_v3 environment and provides:
    - State-based interface (board + legal actions) for Qt rendering
    - Move execution via cell index (0-8) or row/col
    - Automatic turn management

    PettingZoo uses column-major indexing:
        action = col * 3 + row

    So action 0 is (row=0, col=0), action 3 is (row=0, col=1), etc.

    Example usage:
        adapter = TicTacToeAdapter()
        adapter.load(seed=42)

        state = adapter.get_state()
        # state.board = [[0]*3 for _ in range(3)]
        # state.legal_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        new_state = adapter.make_move(4)  # Place mark in center
    """

    # Agent mapping (PettingZoo uses player_1/player_2 for tictactoe)
    PLAYER_1 = "player_1"  # X (goes first)
    PLAYER_2 = "player_2"  # O

    # Board dimensions
    SIZE = 3

    def __init__(self) -> None:
        self._aec_env: Any = None
        self._board: List[List[int]] = [[0] * self.SIZE for _ in range(self.SIZE)]
        self._last_action: Optional[int] = None
        self._last_row: Optional[int] = None
        self._last_col: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = self.PLAYER_1
        self._is_loaded: bool = False
        self._winning_positions: Optional[List[Tuple[int, int]]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if environment is loaded."""
        return self._is_loaded

    def load(self, seed: int = 42, render_mode: str = "rgb_array") -> TicTacToeState:
        """Load and reset the Tic-Tac-Toe environment.

        Args:
            seed: Random seed for environment reset
            render_mode: Render mode ("rgb_array" for pixel output)

        Returns:
            Initial Tic-Tac-Toe state
        """
        try:
            from pettingzoo.classic import tictactoe_v3

            # Create AEC environment
            self._aec_env = tictactoe_v3.env(render_mode=render_mode)

            # Reset environment
            self._aec_env.reset(seed=seed)

            # Reset tracking
            self._board = [[0] * self.SIZE for _ in range(self.SIZE)]
            self._last_action = None
            self._last_row = None
            self._last_col = None
            self._move_count = 0
            self._current_player = self.PLAYER_1
            self._winning_positions = None
            self._is_loaded = True

            _LOG.info(
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.message} | "
                f"env_id=tictactoe_v3 seed={seed}"
            )

            return self.get_state()

        except Exception as e:
            _LOG.error(
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.message} | "
                f"env_id=tictactoe_v3 error={e}"
            )
            raise

    def get_state(self) -> TicTacToeState:
        """Get current Tic-Tac-Toe game state.

        Returns:
            TicTacToeState with board, legal actions, game status, etc.
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Tic-Tac-Toe environment not loaded. Call load() first.")

        # Get legal actions from action mask
        obs, _, terminated, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        legal_actions = [i for i in range(9) if action_mask[i] == 1]

        # Determine game over and winner
        is_game_over = terminated or len(legal_actions) == 0
        winner = None
        if is_game_over:
            winner = self._determine_winner()

        return TicTacToeState(
            board=[row[:] for row in self._board],  # Deep copy
            current_player=self._current_player,
            current_agent=self._current_player,
            legal_actions=legal_actions,
            last_action=self._last_action,
            last_row=self._last_row,
            last_col=self._last_col,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
            winning_positions=self._winning_positions,
        )

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert PettingZoo action index to (row, col) coordinates.

        PettingZoo uses column-major: action = col * 3 + row

        Args:
            action: Action index (0-8)

        Returns:
            (row, col) tuple
        """
        col = action // 3
        row = action % 3
        return (row, col)

    def coords_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to PettingZoo action index.

        Args:
            row: Row index (0-2)
            col: Column index (0-2)

        Returns:
            Action index (0-8)
        """
        return col * 3 + row

    def make_move(self, action: int) -> TicTacToeState:
        """Execute a move by placing a mark in the specified cell.

        Args:
            action: Cell index (0-8) using PettingZoo column-major indexing

        Returns:
            New Tic-Tac-Toe state after the move

        Raises:
            ValueError: If action is invalid or cell is occupied
            RuntimeError: If environment not loaded
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Tic-Tac-Toe environment not loaded. Call load() first.")

        # Validate action
        if action < 0 or action >= 9:
            raise ValueError(f"Invalid action: {action}. Must be 0-8.")

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]

        if action_mask[action] == 0:
            raise ValueError(f"Cell {action} is occupied or illegal.")

        # Place mark
        row, col = self.action_to_coords(action)
        self._place_mark(row, col)
        self._last_action = action
        self._last_row = row
        self._last_col = col
        self._move_count += 1

        # Execute step in environment
        self._aec_env.step(action)

        # Update current player
        self._current_player = self._aec_env.agent_selection

        # Check for win
        self._check_for_win()

        return self.get_state()

    def make_move_by_coords(self, row: int, col: int) -> TicTacToeState:
        """Execute a move by placing a mark at (row, col).

        Args:
            row: Row index (0-2)
            col: Column index (0-2)

        Returns:
            New Tic-Tac-Toe state after the move
        """
        action = self.coords_to_action(row, col)
        return self.make_move(action)

    def _place_mark(self, row: int, col: int) -> None:
        """Place a mark at the specified position.

        Args:
            row: Row index
            col: Column index
        """
        player_value = 1 if self._current_player == self.PLAYER_1 else 2
        self._board[row][col] = player_value

    def _determine_winner(self) -> Optional[str]:
        """Check if there's a winner by looking for 3 in a row.

        Returns:
            Winner agent name, "draw", or None
        """
        # Check rows
        for row in range(self.SIZE):
            if self._board[row][0] != 0 and \
               self._board[row][0] == self._board[row][1] == self._board[row][2]:
                self._winning_positions = [(row, 0), (row, 1), (row, 2)]
                return self.PLAYER_1 if self._board[row][0] == 1 else self.PLAYER_2

        # Check columns
        for col in range(self.SIZE):
            if self._board[0][col] != 0 and \
               self._board[0][col] == self._board[1][col] == self._board[2][col]:
                self._winning_positions = [(0, col), (1, col), (2, col)]
                return self.PLAYER_1 if self._board[0][col] == 1 else self.PLAYER_2

        # Check diagonals
        if self._board[0][0] != 0 and \
           self._board[0][0] == self._board[1][1] == self._board[2][2]:
            self._winning_positions = [(0, 0), (1, 1), (2, 2)]
            return self.PLAYER_1 if self._board[0][0] == 1 else self.PLAYER_2

        if self._board[0][2] != 0 and \
           self._board[0][2] == self._board[1][1] == self._board[2][0]:
            self._winning_positions = [(0, 2), (1, 1), (2, 0)]
            return self.PLAYER_1 if self._board[0][2] == 1 else self.PLAYER_2

        # Check for draw (board full)
        if all(self._board[r][c] != 0 for r in range(self.SIZE) for c in range(self.SIZE)):
            return "draw"

        return None

    def _check_for_win(self) -> None:
        """Check for win and store winning positions."""
        self._determine_winner()

    def is_action_legal(self, action: int) -> bool:
        """Check if placing a mark in the cell is legal.

        Args:
            action: Cell index (0-8)

        Returns:
            True if the move is legal
        """
        if not self._is_loaded or self._aec_env is None:
            return False

        if action < 0 or action >= 9:
            return False

        obs, _, _, _, _ = self._aec_env.last()
        return bool(obs["action_mask"][action] == 1)

    def is_cell_legal(self, row: int, col: int) -> bool:
        """Check if placing a mark at (row, col) is legal.

        Args:
            row: Row index (0-2)
            col: Column index (0-2)

        Returns:
            True if the move is legal
        """
        if row < 0 or row >= self.SIZE or col < 0 or col >= self.SIZE:
            return False
        action = self.coords_to_action(row, col)
        return self.is_action_legal(action)

    def get_legal_actions(self) -> List[int]:
        """Get all legal actions.

        Returns:
            List of cell indices where a mark can be placed
        """
        if not self._is_loaded or self._aec_env is None:
            return []

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        return [i for i in range(9) if action_mask[i] == 1]

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
        self._is_loaded = False

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.close()


__all__ = ["TicTacToeAdapter", "TicTacToeState"]
