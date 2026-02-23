"""GNU Go engine service for Human vs Agent gameplay.

This service wraps the GNU Go engine to provide AI moves
for the GoGameController. It uses the GTP protocol for communication.

GNU Go is a classical Go AI (no neural network required):
- Strength: Amateur dan level
- Installation: sudo apt install gnugo
- Simpler setup than KataGo (no model files needed)
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from gym_gui.services.go_ai.gtp_engine import (
    GTPEngine,
    action_to_vertex,
    vertex_to_action,
)

if TYPE_CHECKING:
    from gym_gui.core.adapters.go_adapter import GoState

_LOG = logging.getLogger(__name__)

# Default paths for GNU Go binary
_DEFAULT_GNUGO_PATHS = [
    "/usr/games/gnugo",
    "/usr/bin/gnugo",
    "/usr/local/bin/gnugo",
    "gnugo",  # Try PATH
]


@dataclass
class GnuGoConfig:
    """Configuration for GNU Go engine.

    Attributes:
        level: Strength level (0=weakest to 10=strongest)
        chinese_rules: Use Chinese rules instead of Japanese
        capture_all_dead: Don't stop at captures, continue to fill
        monte_carlo: Use Monte Carlo mode (experimental)
    """

    level: int = 10  # Default to strongest
    chinese_rules: bool = False
    capture_all_dead: bool = False
    monte_carlo: bool = False


class GnuGoService:
    """Service for GNU Go engine integration via GTP protocol.

    This service provides:
    - Automatic GNU Go binary detection
    - Configurable difficulty via level (0-10)
    - Move generation from any Go position
    - Board synchronization with GoState
    - No neural network required (simpler setup than KataGo)

    Usage:
        service = GnuGoService()
        if service.is_available():
            service.start()
            move = service.get_best_move(go_state)
            service.stop()

    Integration with GoGameController:
        controller = GoGameController()
        gnugo = GnuGoService()
        gnugo.start()
        controller.set_ai_action_provider(gnugo.get_best_move)
    """

    def __init__(self, config: Optional[GnuGoConfig] = None) -> None:
        """Initialize the GNU Go service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or GnuGoConfig()
        self._engine: Optional[GTPEngine] = None
        self._gnugo_path: Optional[str] = None
        self._current_board_size: int = 19

    @staticmethod
    def find_gnugo_binary() -> Optional[str]:
        """Find the GNU Go binary on the system.

        Returns:
            Path to GNU Go binary, or None if not found.
        """
        for path in _DEFAULT_GNUGO_PATHS:
            found = shutil.which(path)
            if found:
                return found
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def is_available(self) -> bool:
        """Check if GNU Go is available on this system.

        Returns:
            True if GNU Go binary was found.
        """
        if self._gnugo_path is None:
            self._gnugo_path = self.find_gnugo_binary()
        return self._gnugo_path is not None

    def start(self) -> bool:
        """Start the GNU Go engine.

        Returns:
            True if engine started successfully.
        """
        if self._engine is not None:
            _LOG.warning("GNU Go already running")
            return True

        if not self.is_available():
            _LOG.error("GNU Go binary not found")
            return False

        try:
            # Build GNU Go GTP command
            cmd = [self._gnugo_path, "--mode", "gtp"]

            # Add level
            cmd.extend(["--level", str(self._config.level)])

            # Add optional flags
            if self._config.chinese_rules:
                cmd.append("--chinese-rules")
            if self._config.capture_all_dead:
                cmd.append("--capture-all-dead")
            if self._config.monte_carlo:
                cmd.append("--monte-carlo")

            self._engine = GTPEngine(*cmd)

            if not self._engine.start():
                self._engine = None
                return False

            # Verify engine is responding
            try:
                name = self._engine.name()
                version = self._engine.version()
                _LOG.info(f"GNU Go started: {name} {version}, level={self._config.level}")
            except Exception as e:
                _LOG.warning(f"GNU Go responding but name/version failed: {e}")

            return True

        except Exception as e:
            _LOG.error(f"Failed to start GNU Go: {e}")
            self._engine = None
            return False

    def stop(self) -> None:
        """Stop the GNU Go engine and release resources."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception as e:
                _LOG.warning(f"Error stopping GNU Go: {e}")
            finally:
                self._engine = None
                _LOG.info("GNU Go stopped")

    def is_running(self) -> bool:
        """Check if the engine is currently running.

        Returns:
            True if engine is active.
        """
        return self._engine is not None and self._engine.is_running()

    def set_level(self, level: int) -> None:
        """Set the engine's strength level.

        Note: This requires restarting the engine to take effect.

        Args:
            level: Strength level from 0 (weakest) to 10 (strongest)
        """
        level = max(0, min(10, level))
        self._config.level = level
        _LOG.debug(f"GNU Go level set to {level} (restart required)")

    def _sync_board(self, state: "GoState") -> bool:
        """Synchronize engine board state with GoState.

        Args:
            state: Current Go game state.

        Returns:
            True if synchronization succeeded.
        """
        if self._engine is None:
            return False

        try:
            # Set board size if changed
            if state.board_size != self._current_board_size:
                if not self._engine.boardsize(state.board_size):
                    return False
                self._current_board_size = state.board_size

            # Clear and replay position
            if not self._engine.clear_board():
                return False

            # Set komi
            self._engine.komi(state.komi)

            # Replay stones on board
            for row in range(state.board_size):
                for col in range(state.board_size):
                    stone = state.board[row][col]
                    if stone != 0:
                        vertex = action_to_vertex(
                            row * state.board_size + col, state.board_size
                        )
                        color = "black" if stone == 1 else "white"
                        self._engine.play(color, vertex)

            return True

        except Exception as e:
            _LOG.error(f"Failed to sync board: {e}")
            return False

    def get_best_move(self, state: "GoState") -> Optional[int]:
        """Get the best move for the current position.

        This method is compatible with GoGameController.set_ai_action_provider().

        Args:
            state: Current Go game state containing board and legal moves.

        Returns:
            Best move action index, or None if no move found.
        """
        if self._engine is None:
            _LOG.warning("GNU Go not running, call start() first")
            return None

        if state.is_game_over:
            _LOG.debug("Game is over, no move to make")
            return None

        if not state.legal_moves:
            _LOG.debug("No legal moves available")
            return None

        try:
            # Sync board position
            if not self._sync_board(state):
                _LOG.warning("Failed to sync board with GNU Go")
                return None

            # Determine color to play
            color = "black" if "black" in state.current_player else "white"

            # Generate move
            vertex = self._engine.genmove(color)

            if vertex is None:
                _LOG.warning("GNU Go returned no move")
                return None

            # Convert vertex to action
            action = vertex_to_action(vertex, state.board_size)
            _LOG.debug(f"GNU Go suggests: {vertex} (action={action})")

            # Verify the move is legal
            if action in state.legal_moves:
                return action

            # If suggested move is illegal, try pass
            _LOG.warning(
                f"GNU Go move {vertex} (action={action}) not in legal moves"
            )
            pass_action = state.board_size ** 2
            if pass_action in state.legal_moves:
                return pass_action

            # Fall back to first legal move
            return state.legal_moves[0] if state.legal_moves else None

        except Exception as e:
            _LOG.error(f"GNU Go error getting move: {e}")
            return None

    def get_config(self) -> GnuGoConfig:
        """Get the current configuration.

        Returns:
            Copy of current configuration.
        """
        return GnuGoConfig(
            level=self._config.level,
            chinese_rules=self._config.chinese_rules,
            capture_all_dead=self._config.capture_all_dead,
            monte_carlo=self._config.monte_carlo,
        )

    def __enter__(self) -> "GnuGoService":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# Difficulty presets for user-friendly configuration
GNUGO_DIFFICULTY_PRESETS = {
    "beginner": GnuGoConfig(level=1),
    "easy": GnuGoConfig(level=4),
    "medium": GnuGoConfig(level=7),
    "hard": GnuGoConfig(level=9),
    "expert": GnuGoConfig(level=10),
}

GNUGO_DIFFICULTY_DESCRIPTIONS = {
    "beginner": "Very weak play. Perfect for learning Go basics.",
    "easy": "Casual play. Makes obvious mistakes.",
    "medium": "Moderate challenge. Good for beginners improving.",
    "hard": "Strong classical play. Good for intermediate players.",
    "expert": "Maximum GNU Go strength. Amateur dan level.",
}


def create_gnugo_provider(
    difficulty: str = "medium",
) -> tuple[GnuGoService, Callable]:
    """Create a GNU Go service and action provider function.

    Args:
        difficulty: One of "beginner", "easy", "medium", "hard", "expert"

    Returns:
        Tuple of (GnuGoService instance, action_provider callable)

    Raises:
        RuntimeError: If GNU Go is not available or fails to start.
    """
    config = GNUGO_DIFFICULTY_PRESETS.get(difficulty, GNUGO_DIFFICULTY_PRESETS["medium"])
    service = GnuGoService(config)

    if not service.is_available():
        raise RuntimeError("GNU Go not available (binary not found)")

    if not service.start():
        raise RuntimeError("Failed to start GNU Go engine")

    return service, service.get_best_move


__all__ = [
    "GnuGoConfig",
    "GnuGoService",
    "GNUGO_DIFFICULTY_PRESETS",
    "GNUGO_DIFFICULTY_DESCRIPTIONS",
    "create_gnugo_provider",
]
