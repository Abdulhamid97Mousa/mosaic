"""Connect Four-specific adapter for PettingZoo connect_four_v3 environment.

This adapter provides a state-based interface for Connect Four, enabling:
- Board state for Qt-based rendering (6x7 grid)
- Legal column validation via action masks
- Column-based move execution
- Turn management

The adapter uses the AEC API directly for turn-based gameplay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from gym_gui.logging_config.log_constants import (
    LOG_UI_MULTI_AGENT_ENV_LOADED,
    LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class ConnectFourState:
    """Structured Connect Four game state for UI rendering."""

    board: List[List[int]]  # 6x7 grid, 0=empty, 1=player_0 (red), 2=player_1 (yellow)
    current_player: str  # "player_0" or "player_1"
    current_agent: str  # Same as current_player for Connect Four
    legal_columns: List[int]  # Columns where a piece can be dropped (0-6)
    last_column: Optional[int] = None
    last_row: Optional[int] = None  # Row where last piece landed
    is_game_over: bool = False
    winner: Optional[str] = None  # "player_0", "player_1", or "draw"
    move_count: int = 0
    winning_positions: Optional[List[tuple[int, int]]] = None  # 4 winning positions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal emission."""
        return {
            "game_type": "connect_four",
            "board": self.board,
            "current_player": self.current_player,
            "current_agent": self.current_agent,
            "legal_columns": self.legal_columns,
            "last_column": self.last_column,
            "last_row": self.last_row,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
            "winning_positions": self.winning_positions,
        }


class ConnectFourAdapter:
    """Adapter for PettingZoo Connect Four environment with state-based interface.

    This adapter wraps the PettingZoo connect_four_v3 environment and provides:
    - State-based interface (board + legal columns) for Qt rendering
    - Move execution via column index (0-6)
    - Automatic turn management

    Example usage:
        adapter = ConnectFourAdapter()
        adapter.load(seed=42)

        state = adapter.get_state()
        # state.board = [[0]*7 for _ in range(6)]
        # state.legal_columns = [0, 1, 2, 3, 4, 5, 6]

        new_state = adapter.make_move(3)  # Drop piece in column 3
    """

    # Agent mapping
    PLAYER_0 = "player_0"  # Red (goes first)
    PLAYER_1 = "player_1"  # Yellow

    # Board dimensions
    ROWS = 6
    COLS = 7

    def __init__(self) -> None:
        self._aec_env: Any = None
        self._board: List[List[int]] = [[0] * self.COLS for _ in range(self.ROWS)]
        self._last_column: Optional[int] = None
        self._last_row: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = self.PLAYER_0
        self._is_loaded: bool = False
        self._winning_positions: Optional[List[tuple[int, int]]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if environment is loaded."""
        return self._is_loaded

    def load(self, seed: int = 42, render_mode: str = "rgb_array") -> ConnectFourState:
        """Load and reset the Connect Four environment.

        Args:
            seed: Random seed for environment reset
            render_mode: Render mode ("rgb_array" for pixel output)

        Returns:
            Initial Connect Four state
        """
        try:
            from pettingzoo.classic import connect_four_v3

            # Create AEC environment
            self._aec_env = connect_four_v3.env(render_mode=render_mode)

            # Reset environment
            self._aec_env.reset(seed=seed)

            # Reset tracking
            self._board = [[0] * self.COLS for _ in range(self.ROWS)]
            self._last_column = None
            self._last_row = None
            self._move_count = 0
            self._current_player = self.PLAYER_0
            self._winning_positions = None
            self._is_loaded = True

            _LOG.info(
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.message} | "
                f"env_id=connect_four_v3 seed={seed}"
            )

            return self.get_state()

        except Exception as e:
            _LOG.error(
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.message} | "
                f"env_id=connect_four_v3 error={e}"
            )
            raise

    def get_state(self) -> ConnectFourState:
        """Get current Connect Four game state.

        Returns:
            ConnectFourState with board, legal columns, game status, etc.
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Connect Four environment not loaded. Call load() first.")

        # Get legal columns from action mask
        obs, _, terminated, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        legal_columns = [i for i in range(self.COLS) if action_mask[i] == 1]

        # Determine game over and winner
        is_game_over = terminated or len(legal_columns) == 0
        winner = None
        if is_game_over:
            winner = self._determine_winner()

        return ConnectFourState(
            board=[row[:] for row in self._board],  # Deep copy
            current_player=self._current_player,
            current_agent=self._current_player,
            legal_columns=legal_columns,
            last_column=self._last_column,
            last_row=self._last_row,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
            winning_positions=self._winning_positions,
        )

    def make_move(self, column: int) -> ConnectFourState:
        """Execute a move by dropping a piece in the specified column.

        Args:
            column: Column index (0-6)

        Returns:
            New Connect Four state after the move

        Raises:
            ValueError: If column is invalid or full
            RuntimeError: If environment not loaded
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Connect Four environment not loaded. Call load() first.")

        # Validate column
        if column < 0 or column >= self.COLS:
            raise ValueError(f"Invalid column: {column}. Must be 0-6.")

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]

        if action_mask[column] == 0:
            raise ValueError(f"Column {column} is full or illegal.")

        # Drop piece in column
        row = self._drop_piece(column)
        self._last_column = column
        self._last_row = row
        self._move_count += 1

        # Execute step in environment
        self._aec_env.step(column)

        # Update current player
        self._current_player = self._aec_env.agent_selection

        # Check for win
        self._check_for_win()

        return self.get_state()

    def _drop_piece(self, column: int) -> int:
        """Drop a piece in the specified column.

        Args:
            column: Column index

        Returns:
            Row where the piece landed
        """
        player_value = 1 if self._current_player == self.PLAYER_0 else 2

        # Find the lowest empty row in the column
        for row in range(self.ROWS - 1, -1, -1):
            if self._board[row][column] == 0:
                self._board[row][column] = player_value
                return row

        raise ValueError(f"Column {column} is full")

    def _determine_winner(self) -> Optional[str]:
        """Check if there's a winner by looking for 4 in a row.

        Returns:
            Winner agent name, "draw", or None
        """
        # Check for 4 in a row (horizontal, vertical, diagonal)
        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece = self._board[row][col]
                if piece == 0:
                    continue

                # Check horizontal
                if col <= self.COLS - 4:
                    if all(self._board[row][col + i] == piece for i in range(4)):
                        self._winning_positions = [(row, col + i) for i in range(4)]
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1

                # Check vertical
                if row <= self.ROWS - 4:
                    if all(self._board[row + i][col] == piece for i in range(4)):
                        self._winning_positions = [(row + i, col) for i in range(4)]
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1

                # Check diagonal (down-right)
                if row <= self.ROWS - 4 and col <= self.COLS - 4:
                    if all(self._board[row + i][col + i] == piece for i in range(4)):
                        self._winning_positions = [(row + i, col + i) for i in range(4)]
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1

                # Check diagonal (up-right)
                if row >= 3 and col <= self.COLS - 4:
                    if all(self._board[row - i][col + i] == piece for i in range(4)):
                        self._winning_positions = [(row - i, col + i) for i in range(4)]
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1

        # Check for draw (board full)
        if all(self._board[0][col] != 0 for col in range(self.COLS)):
            return "draw"

        return None

    def _check_for_win(self) -> None:
        """Check for win and store winning positions."""
        self._determine_winner()

    def is_column_legal(self, column: int) -> bool:
        """Check if dropping a piece in the column is legal.

        Args:
            column: Column index (0-6)

        Returns:
            True if the move is legal
        """
        if not self._is_loaded or self._aec_env is None:
            return False

        if column < 0 or column >= self.COLS:
            return False

        obs, _, _, _, _ = self._aec_env.last()
        return bool(obs["action_mask"][column] == 1)

    def get_legal_columns(self) -> List[int]:
        """Get all legal columns.

        Returns:
            List of column indices where a piece can be dropped
        """
        if not self._is_loaded or self._aec_env is None:
            return []

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        return [i for i in range(self.COLS) if action_mask[i] == 1]

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


__all__ = ["ConnectFourAdapter", "ConnectFourState"]
