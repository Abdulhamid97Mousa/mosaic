"""Runtime for Stockfish Worker.

This module provides the interactive runtime for the Stockfish chess engine worker.
It integrates with the MOSAIC operator system for multi-agent chess games.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import StockfishWorkerConfig

# Import Stockfish library
try:
    from stockfish import Stockfish
    _HAS_STOCKFISH = True
except ImportError:
    _HAS_STOCKFISH = False
    Stockfish = None

# Import standardized telemetry
try:
    from gym_gui.core.worker import TelemetryEmitter
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None

LOGGER = logging.getLogger(__name__)

# Default paths for Stockfish binary
_DEFAULT_STOCKFISH_PATHS = [
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish",  # Try PATH
]


@dataclass
class StockfishState:
    """State tracking for a chess game."""
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    legal_moves: list = None  # type: ignore[assignment]
    is_game_over: bool = False
    result: Optional[str] = None

    def __post_init__(self) -> None:
        if self.legal_moves is None:
            self.legal_moves = []


class StockfishWorkerRuntime:
    """Interactive runtime for Stockfish chess engine.

    This runtime operates in interactive mode, receiving observations via stdin
    and emitting actions via stdout (JSONL protocol). It integrates with the
    MOSAIC operator system for multi-agent chess games.

    Protocol:
        Input (stdin): JSON objects with observation data
            {"type": "observation", "fen": "...", "legal_moves": [...]}
            {"type": "reset"}
            {"type": "shutdown"}

        Output (stdout): JSONL with actions and telemetry
            {"type": "action", "action": "e2e4"}
            {"type": "step", "step_index": 0, "action": "e2e4", ...}

    Example:
        >>> config = StockfishWorkerConfig(run_id="test", difficulty="medium")
        >>> runtime = StockfishWorkerRuntime(config)
        >>> runtime.run_interactive()
    """

    def __init__(self, config: StockfishWorkerConfig) -> None:
        """Initialize Stockfish runtime.

        Args:
            config: Worker configuration

        Raises:
            ImportError: If python-stockfish is not installed
            RuntimeError: If Stockfish binary not found
        """
        if not _HAS_STOCKFISH:
            raise ImportError(
                "python-stockfish is required. Install with: pip install stockfish"
            )

        self.config = config
        self._engine: Optional[Stockfish] = None
        self._stockfish_path: Optional[str] = None
        self._step_index = 0
        self._episode_index = 0

        # Find Stockfish binary
        self._stockfish_path = self._find_stockfish_binary()
        if not self._stockfish_path:
            raise RuntimeError(
                "Stockfish binary not found. Install with: sudo apt install stockfish"
            )

        # Create lifecycle telemetry emitter
        if _HAS_GYM_GUI and config.run_id:
            self._emitter = TelemetryEmitter(run_id=config.run_id)
        else:
            self._emitter = None

        LOGGER.info(
            f"StockfishWorkerRuntime initialized: "
            f"skill={config.skill_level}, depth={config.depth}, "
            f"time={config.time_limit_ms}ms"
        )

    def _find_stockfish_binary(self) -> Optional[str]:
        """Find the Stockfish binary on the system."""
        import shutil
        import os

        # Check custom path first
        if self.config.stockfish_path:
            if os.path.isfile(self.config.stockfish_path):
                return self.config.stockfish_path

        # Search default paths
        for path in _DEFAULT_STOCKFISH_PATHS:
            found = shutil.which(path)
            if found:
                return found
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    def start(self) -> bool:
        """Start the Stockfish engine.

        Returns:
            True if engine started successfully.
        """
        if self._engine is not None:
            LOGGER.warning("Stockfish already running")
            return True

        try:
            self._engine = Stockfish(
                path=self._stockfish_path,
                depth=self.config.depth,
                parameters={
                    "Threads": self.config.threads,
                    "Hash": self.config.hash_mb,
                    "Skill Level": self.config.skill_level,
                },
            )
            LOGGER.info(f"Stockfish started: {self._stockfish_path}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to start Stockfish: {e}")
            self._engine = None
            return False

    def stop(self) -> None:
        """Stop the Stockfish engine."""
        if self._engine is not None:
            try:
                del self._engine
            except Exception as e:
                LOGGER.warning(f"Error stopping Stockfish: {e}")
            finally:
                self._engine = None
                LOGGER.info("Stockfish stopped")

    def get_best_move(self, fen: str, legal_moves: list) -> Optional[str]:
        """Get the best move for a position.

        Args:
            fen: Position in FEN notation
            legal_moves: List of legal moves in UCI notation

        Returns:
            Best move in UCI notation, or None if no move found
        """
        if self._engine is None:
            LOGGER.warning("Stockfish not running")
            return None

        try:
            self._engine.set_fen_position(fen)
            best_move = self._engine.get_best_move_time(self.config.time_limit_ms)

            if best_move:
                # Verify the move is legal
                if best_move in legal_moves:
                    return best_move
                # Handle promotion (Stockfish returns e.g., "e7e8q")
                base_move = best_move[:4]
                matching = [m for m in legal_moves if m.startswith(base_move)]
                if matching:
                    return matching[0]

            return None
        except Exception as e:
            LOGGER.error(f"Error getting move: {e}")
            return None

    def get_evaluation(self, fen: str) -> Optional[Dict[str, Any]]:
        """Get position evaluation.

        Args:
            fen: Position in FEN notation

        Returns:
            Evaluation dict with 'type' and 'value', or None
        """
        if self._engine is None:
            return None

        try:
            self._engine.set_fen_position(fen)
            return self._engine.get_evaluation()
        except Exception as e:
            LOGGER.error(f"Error getting evaluation: {e}")
            return None

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit data to stdout as JSONL."""
        print(json.dumps(data), flush=True)

    def run_interactive(self) -> Dict[str, Any]:
        """Run in interactive mode, processing stdin commands.

        This is the main entry point for operator integration.

        Returns:
            Summary dict with execution results
        """
        # Emit run_started
        if self._emitter:
            self._emitter.run_started({
                "env_name": self.config.env_name,
                "task": self.config.task,
                "difficulty": self.config.difficulty,
                "skill_level": self.config.skill_level,
            })

        # Start engine
        if not self.start():
            error_summary = {"status": "failed", "error": "Failed to start Stockfish"}
            if self._emitter:
                self._emitter.run_failed(error_summary)
            return error_summary

        try:
            LOGGER.info("Entering interactive loop...")

            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    LOGGER.warning(f"Invalid JSON: {e}")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "shutdown":
                    LOGGER.info("Received shutdown command")
                    break

                elif msg_type == "reset":
                    self._step_index = 0
                    self._episode_index += 1
                    LOGGER.debug(f"Reset: episode {self._episode_index}")
                    self._emit({"type": "reset_ack", "episode_index": self._episode_index})

                elif msg_type == "observation":
                    # Process observation and return action
                    fen = msg.get("fen", "")
                    legal_moves = msg.get("legal_moves", [])
                    is_game_over = msg.get("is_game_over", False)

                    if is_game_over:
                        LOGGER.debug("Game over, no action needed")
                        continue

                    if not fen or not legal_moves:
                        LOGGER.warning("Invalid observation: missing fen or legal_moves")
                        continue

                    # Get best move
                    start_time = time.time()
                    action = self.get_best_move(fen, legal_moves)
                    think_time_ms = int((time.time() - start_time) * 1000)

                    if action is None:
                        # Fallback to first legal move
                        action = legal_moves[0] if legal_moves else None
                        LOGGER.warning(f"No best move found, using fallback: {action}")

                    # Get evaluation
                    evaluation = self.get_evaluation(fen)

                    # Emit action
                    action_msg = {
                        "type": "action",
                        "action": action,
                        "think_time_ms": think_time_ms,
                    }
                    self._emit(action_msg)

                    # Emit step telemetry
                    step_data = {
                        "type": "step",
                        "step_index": self._step_index,
                        "episode_index": self._episode_index,
                        "action": action,
                        "observation": fen,
                        "think_time_ms": think_time_ms,
                    }
                    if evaluation:
                        step_data["evaluation"] = evaluation

                    self._emit(step_data)

                    # Emit heartbeat periodically
                    if self._emitter and self._step_index % 10 == 0:
                        self._emitter.heartbeat({
                            "step": self._step_index,
                            "episode": self._episode_index,
                        })

                    self._step_index += 1

                else:
                    LOGGER.warning(f"Unknown message type: {msg_type}")

            # Clean shutdown
            summary = {
                "status": "completed",
                "episodes": self._episode_index,
                "total_steps": self._step_index,
                "config": self.config.to_dict(),
            }

            if self._emitter:
                self._emitter.run_completed(summary)

            return summary

        except Exception as e:
            LOGGER.error(f"Interactive loop error: {e}", exc_info=True)
            error_summary = {"status": "failed", "error": str(e)}
            if self._emitter:
                self._emitter.run_failed(error_summary)
            raise

        finally:
            self.stop()

    def run(self) -> Dict[str, Any]:
        """Run the worker (alias for run_interactive).

        Returns:
            Summary dict with execution results
        """
        return self.run_interactive()

    def __enter__(self) -> "StockfishWorkerRuntime":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


__all__ = ["StockfishWorkerRuntime", "StockfishState"]
