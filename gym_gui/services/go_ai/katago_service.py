"""KataGo Go engine service for Human vs Agent gameplay.

This service wraps the KataGo Go engine to provide AI moves
for the GoGameController. It uses the GTP protocol for communication.

KataGo is a superhuman-strength Go AI that requires:
- KataGo binary (sudo apt install katago)
- Neural network model file (.bin.gz or .txt.gz)
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from gym_gui.config.paths import VAR_BIN_DIR, VAR_MODELS_KATAGO_DIR
from gym_gui.services.go_ai.gtp_engine import (
    GTPEngine,
    action_to_vertex,
    vertex_to_action,
)

if TYPE_CHECKING:
    from gym_gui.core.adapters.go_adapter import GoState

_LOG = logging.getLogger(__name__)

# Default paths for KataGo binary
_DEFAULT_KATAGO_PATHS = [
    str(VAR_BIN_DIR / "katago"),  # Project-local binary (preferred)
    "/usr/games/katago",
    "/usr/bin/katago",
    "/usr/local/bin/katago",
    "/opt/katago/katago",
    "katago",  # Try PATH
]

# Default paths for KataGo model files
_DEFAULT_MODEL_PATHS = [
    VAR_MODELS_KATAGO_DIR,  # Project-local: var/models/katago/
]


@dataclass
class KataGoConfig:
    """Configuration for KataGo engine.

    Attributes:
        playouts: Number of playouts/visits per move (strength control)
        max_visits: Maximum visits (alternative strength control)
        time_limit_sec: Time limit per move in seconds
        threads: Number of CPU threads for search
        model_path: Optional path to neural network model file
    """

    playouts: int = 200
    max_visits: int = 400
    time_limit_sec: float = 5.0
    threads: int = 1
    model_path: Optional[str] = None


class KataGoService:
    """Service for KataGo Go engine integration via GTP protocol.

    This service provides:
    - Automatic KataGo binary and model detection
    - Configurable difficulty via playouts/visits
    - Move generation from any Go position
    - Board synchronization with GoState

    Usage:
        service = KataGoService()
        if service.is_available():
            service.start()
            move = service.get_best_move(go_state)
            service.stop()

    Integration with GoGameController:
        controller = GoGameController()
        katago = KataGoService()
        katago.start()
        controller.set_ai_action_provider(katago.get_best_move)
    """

    def __init__(self, config: Optional[KataGoConfig] = None) -> None:
        """Initialize the KataGo service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or KataGoConfig()
        self._engine: Optional[GTPEngine] = None
        self._katago_path: Optional[str] = None
        self._model_path: Optional[str] = None
        self._current_board_size: int = 19
        self._move_history: List[tuple] = []  # [(color, vertex), ...]

    @staticmethod
    def find_katago_binary() -> Optional[str]:
        """Find the KataGo binary on the system.

        Returns:
            Path to KataGo binary, or None if not found.
        """
        for path in _DEFAULT_KATAGO_PATHS:
            found = shutil.which(path)
            if found:
                return found
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    @staticmethod
    def find_katago_model() -> Optional[str]:
        """Find a KataGo neural network model file.

        Returns:
            Path to model file, or None if not found.
        """
        for model_dir in _DEFAULT_MODEL_PATHS:
            model_dir_str = str(model_dir)
            expanded = os.path.expanduser(model_dir_str)
            if os.path.isdir(expanded):
                # Look for .bin.gz or .txt.gz model files
                for f in os.listdir(expanded):
                    if f.endswith(".bin.gz") or f.endswith(".txt.gz"):
                        return os.path.join(expanded, f)
        return None

    def is_available(self) -> bool:
        """Check if KataGo is available on this system.

        Returns:
            True if KataGo binary and model were found.
        """
        if self._katago_path is None:
            self._katago_path = self.find_katago_binary()
        if self._model_path is None:
            self._model_path = self._config.model_path or self.find_katago_model()

        if self._katago_path is None:
            _LOG.debug("KataGo binary not found")
            return False
        if self._model_path is None:
            _LOG.debug("KataGo model not found")
            return False
        return True

    def start(self) -> bool:
        """Start the KataGo engine.

        Returns:
            True if engine started successfully.
        """
        if self._engine is not None:
            _LOG.warning("KataGo already running")
            return True

        if not self.is_available():
            _LOG.error("KataGo binary or model not found")
            return False

        try:
            # Build KataGo GTP command
            # katago gtp -model <model> -config <config>
            # We use default config and set options via GTP commands
            assert self._katago_path is not None
            assert self._model_path is not None
            self._engine = GTPEngine(
                self._katago_path,
                "gtp",
                "-model",
                self._model_path,
            )

            if not self._engine.start():
                self._engine = None
                return False

            # Verify engine is responding
            try:
                name = self._engine.name()
                version = self._engine.version()
                _LOG.info(f"KataGo started: {name} {version}")
            except Exception as e:
                _LOG.warning(f"KataGo responding but name/version failed: {e}")

            return True

        except Exception as e:
            _LOG.error(f"Failed to start KataGo: {e}")
            self._engine = None
            return False

    def stop(self) -> None:
        """Stop the KataGo engine and release resources."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception as e:
                _LOG.warning(f"Error stopping KataGo: {e}")
            finally:
                self._engine = None
                self._move_history.clear()
                _LOG.info("KataGo stopped")

    def is_running(self) -> bool:
        """Check if the engine is currently running.

        Returns:
            True if engine is active.
        """
        return self._engine is not None and self._engine.is_running()

    def set_playouts(self, playouts: int) -> None:
        """Set the number of playouts per move.

        Args:
            playouts: Number of playouts (higher = stronger but slower)
        """
        playouts = max(1, playouts)
        self._config.playouts = playouts
        _LOG.debug(f"KataGo playouts set to {playouts}")

    def _sync_board(self, state: "GoState") -> bool:
        """Synchronize engine board state with GoState.

        This replays the current board position to the engine.

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
            # We need to place all existing stones
            for row in range(state.board_size):
                for col in range(state.board_size):
                    stone = state.board[row][col]
                    if stone != 0:
                        vertex = action_to_vertex(
                            row * state.board_size + col, state.board_size
                        )
                        color = "black" if stone == 1 else "white"
                        # Use "play" to set up position without generating response
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
            _LOG.warning("KataGo not running, call start() first")
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
                _LOG.warning("Failed to sync board with KataGo")
                return None

            # Determine color to play
            color = "black" if "black" in state.current_player else "white"

            # Generate move
            vertex = self._engine.genmove(color)

            if vertex is None:
                _LOG.warning("KataGo returned no move")
                return None

            # Convert vertex to action
            action = vertex_to_action(vertex, state.board_size)
            _LOG.debug(f"KataGo suggests: {vertex} (action={action})")

            # Verify the move is legal
            if action in state.legal_moves:
                return action

            # If suggested move is illegal, log warning and return pass if available
            _LOG.warning(
                f"KataGo move {vertex} (action={action}) not in legal moves"
            )
            pass_action = state.board_size ** 2
            if pass_action in state.legal_moves:
                return pass_action

            # Fall back to first legal move
            return state.legal_moves[0] if state.legal_moves else None

        except Exception as e:
            _LOG.error(f"KataGo error getting move: {e}")
            return None

    def get_config(self) -> KataGoConfig:
        """Get the current configuration.

        Returns:
            Copy of current configuration.
        """
        return KataGoConfig(
            playouts=self._config.playouts,
            max_visits=self._config.max_visits,
            time_limit_sec=self._config.time_limit_sec,
            threads=self._config.threads,
            model_path=self._config.model_path,
        )

    def __enter__(self) -> "KataGoService":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# Difficulty presets for user-friendly configuration
KATAGO_DIFFICULTY_PRESETS = {
    "beginner": KataGoConfig(playouts=10, max_visits=20, time_limit_sec=1.0),
    "easy": KataGoConfig(playouts=50, max_visits=100, time_limit_sec=2.0),
    "medium": KataGoConfig(playouts=200, max_visits=400, time_limit_sec=5.0),
    "hard": KataGoConfig(playouts=800, max_visits=1600, time_limit_sec=10.0),
    "expert": KataGoConfig(playouts=3200, max_visits=6400, time_limit_sec=30.0),
}

KATAGO_DIFFICULTY_DESCRIPTIONS = {
    "beginner": "Very weak play. Perfect for learning Go basics.",
    "easy": "Casual play. Makes mistakes but plays sensible Go.",
    "medium": "Balanced challenge. Good for intermediate players (10-15 kyu).",
    "hard": "Strong play. Good for dan-level players.",
    "expert": "Maximum strength. Professional-level play.",
}


def create_katago_provider(
    difficulty: str = "medium",
) -> tuple[KataGoService, Callable]:
    """Create a KataGo service and action provider function.

    Args:
        difficulty: One of "beginner", "easy", "medium", "hard", "expert"

    Returns:
        Tuple of (KataGoService instance, action_provider callable)

    Raises:
        RuntimeError: If KataGo is not available or fails to start.
    """
    config = KATAGO_DIFFICULTY_PRESETS.get(difficulty, KATAGO_DIFFICULTY_PRESETS["medium"])
    service = KataGoService(config)

    if not service.is_available():
        raise RuntimeError("KataGo not available (binary or model not found)")

    if not service.start():
        raise RuntimeError("Failed to start KataGo engine")

    return service, service.get_best_move


__all__ = [
    "KataGoConfig",
    "KataGoService",
    "KATAGO_DIFFICULTY_PRESETS",
    "KATAGO_DIFFICULTY_DESCRIPTIONS",
    "create_katago_provider",
]
