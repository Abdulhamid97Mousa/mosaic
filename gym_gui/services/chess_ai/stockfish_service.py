"""Stockfish chess engine service for Human vs Agent gameplay.

This service wraps the Stockfish chess engine to provide AI moves
for the ChessGameController. It uses the python-stockfish library
to communicate via UCI protocol.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from stockfish import Stockfish as StockfishType
    from gym_gui.core.adapters.chess_adapter import ChessState

_LOG = logging.getLogger(__name__)

# Default path for Stockfish binary
_DEFAULT_STOCKFISH_PATHS = [
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish",  # Try PATH
]


@dataclass
class StockfishConfig:
    """Configuration for Stockfish engine.

    Attributes:
        skill_level: Engine strength (0=weakest to 20=strongest)
        depth: Search depth (higher = stronger but slower)
        time_limit_ms: Time limit per move in milliseconds
        threads: Number of CPU threads to use
        hash_mb: Hash table size in MB
    """

    skill_level: int = 10  # Medium difficulty by default
    depth: int = 15
    time_limit_ms: int = 1000
    threads: int = 1
    hash_mb: int = 16


class StockfishService:
    """Service for Stockfish chess engine integration.

    This service provides:
    - Automatic Stockfish binary detection
    - Configurable difficulty levels
    - Move generation from any chess position (FEN)
    - Thread-safe operation

    Usage:
        service = StockfishService()
        if service.is_available():
            service.start()
            move = service.get_best_move(chess_state)
            service.stop()

    Integration with ChessGameController:
        controller = ChessGameController()
        stockfish = StockfishService()
        stockfish.start()
        controller.set_ai_action_provider(stockfish.get_best_move)
    """

    def __init__(self, config: Optional[StockfishConfig] = None) -> None:
        """Initialize the Stockfish service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or StockfishConfig()
        self._engine: Optional["StockfishType"] = None
        self._stockfish_path: Optional[str] = None

    @staticmethod
    def find_stockfish_binary() -> Optional[str]:
        """Find the Stockfish binary on the system.

        Returns:
            Path to Stockfish binary, or None if not found.
        """
        for path in _DEFAULT_STOCKFISH_PATHS:
            found = shutil.which(path) or (path if shutil.which(path) else None)
            if found:
                return found
            # Check if path exists directly
            import os

            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def is_available(self) -> bool:
        """Check if Stockfish is available on this system.

        Returns:
            True if Stockfish binary was found.
        """
        if self._stockfish_path is None:
            self._stockfish_path = self.find_stockfish_binary()
        return self._stockfish_path is not None

    def start(self) -> bool:
        """Start the Stockfish engine.

        Returns:
            True if engine started successfully.
        """
        if self._engine is not None:
            _LOG.warning("Stockfish already running")
            return True

        if not self.is_available():
            _LOG.error("Stockfish binary not found")
            return False

        try:
            from stockfish import Stockfish

            self._engine = Stockfish(
                path=self._stockfish_path,
                depth=self._config.depth,
                parameters={
                    "Threads": self._config.threads,
                    "Hash": self._config.hash_mb,
                    "Skill Level": self._config.skill_level,
                },
            )
            _LOG.info(
                f"Stockfish started: skill={self._config.skill_level}, "
                f"depth={self._config.depth}"
            )
            return True
        except Exception as e:
            _LOG.error(f"Failed to start Stockfish: {e}")
            self._engine = None
            return False

    def stop(self) -> None:
        """Stop the Stockfish engine and release resources."""
        if self._engine is not None:
            try:
                # The stockfish library doesn't have an explicit close method
                # but we can clear our reference to allow garbage collection
                del self._engine
            except Exception as e:
                _LOG.warning(f"Error stopping Stockfish: {e}")
            finally:
                self._engine = None
                _LOG.info("Stockfish stopped")

    def is_running(self) -> bool:
        """Check if the engine is currently running.

        Returns:
            True if engine is active.
        """
        return self._engine is not None

    def set_skill_level(self, level: int) -> None:
        """Set the engine's skill level.

        Args:
            level: Skill level from 0 (weakest) to 20 (strongest)
        """
        level = max(0, min(20, level))
        self._config.skill_level = level
        if self._engine is not None:
            self._engine.set_skill_level(level)
            _LOG.debug(f"Stockfish skill level set to {level}")

    def set_depth(self, depth: int) -> None:
        """Set the search depth.

        Args:
            depth: Search depth (1-30, higher = stronger but slower)
        """
        depth = max(1, min(30, depth))
        self._config.depth = depth
        if self._engine is not None:
            self._engine.set_depth(depth)
            _LOG.debug(f"Stockfish depth set to {depth}")

    def get_best_move(self, state: "ChessState") -> Optional[str]:
        """Get the best move for the current position.

        This method is compatible with ChessGameController.set_ai_action_provider().

        Args:
            state: Current chess game state containing FEN and legal moves.

        Returns:
            Best move in UCI notation (e.g., "e2e4"), or None if no move found.
        """
        if self._engine is None:
            _LOG.warning("Stockfish not running, call start() first")
            return None

        if state.is_game_over:
            _LOG.debug("Game is over, no move to make")
            return None

        try:
            # Set the position from FEN
            self._engine.set_fen_position(state.fen)

            # Get best move with time limit
            best_move = self._engine.get_best_move_time(self._config.time_limit_ms)

            if best_move:
                _LOG.debug(f"Stockfish suggests: {best_move}")
                # Verify the move is legal
                if best_move in state.legal_moves:
                    return best_move
                # Handle promotion moves (Stockfish returns e.g., "e7e8q")
                # which should match our legal_moves format
                _LOG.warning(
                    f"Stockfish move {best_move} not in legal moves: {state.legal_moves[:5]}..."
                )
                # Try without promotion suffix if it's there
                base_move = best_move[:4]
                matching = [m for m in state.legal_moves if m.startswith(base_move)]
                if matching:
                    return matching[0]

            return None

        except Exception as e:
            _LOG.error(f"Stockfish error getting move: {e}")
            return None

    def get_evaluation(self, state: "ChessState") -> Optional[dict]:
        """Get the position evaluation.

        Args:
            state: Current chess game state.

        Returns:
            Evaluation dict with 'type' ('cp' or 'mate') and 'value',
            or None if evaluation failed.
        """
        if self._engine is None:
            return None

        try:
            self._engine.set_fen_position(state.fen)
            return self._engine.get_evaluation()
        except Exception as e:
            _LOG.error(f"Stockfish evaluation error: {e}")
            return None

    def get_config(self) -> StockfishConfig:
        """Get the current configuration.

        Returns:
            Copy of current configuration.
        """
        return StockfishConfig(
            skill_level=self._config.skill_level,
            depth=self._config.depth,
            time_limit_ms=self._config.time_limit_ms,
            threads=self._config.threads,
            hash_mb=self._config.hash_mb,
        )

    def __enter__(self) -> "StockfishService":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# Difficulty presets for user-friendly configuration
DIFFICULTY_PRESETS = {
    "beginner": StockfishConfig(skill_level=1, depth=5, time_limit_ms=500),
    "easy": StockfishConfig(skill_level=5, depth=8, time_limit_ms=500),
    "medium": StockfishConfig(skill_level=10, depth=12, time_limit_ms=1000),
    "hard": StockfishConfig(skill_level=15, depth=18, time_limit_ms=1500),
    "expert": StockfishConfig(skill_level=20, depth=20, time_limit_ms=2000),
}


def create_stockfish_provider(
    difficulty: str = "medium",
) -> tuple[StockfishService, callable]:
    """Create a Stockfish service and action provider function.

    This is a convenience function that creates and starts a Stockfish service
    configured for the given difficulty level.

    Args:
        difficulty: One of "beginner", "easy", "medium", "hard", "expert"

    Returns:
        Tuple of (StockfishService instance, action_provider callable)

    Example:
        service, provider = create_stockfish_provider("hard")
        controller.set_ai_action_provider(provider)
        # ... play game ...
        service.stop()  # Don't forget to stop when done!
    """
    config = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["medium"])
    service = StockfishService(config)

    if not service.start():
        raise RuntimeError("Failed to start Stockfish engine")

    return service, service.get_best_move
