"""Runtime for Dragon (Komodo Dragon) chess engine worker.

This module provides the interactive runtime for the Dragon worker,
handling UCI protocol communication and move generation.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import chess
import chess.engine

from .config import DragonWorkerConfig, _find_dragon_binary


logger = logging.getLogger(__name__)


class DragonState(Enum):
    """States for the Dragon worker state machine."""

    IDLE = "idle"
    WAITING_FOR_POSITION = "waiting_for_position"
    THINKING = "thinking"
    MOVE_READY = "move_ready"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class DragonWorkerRuntime:
    """Interactive runtime for Dragon chess engine.

    This runtime operates in an interactive mode, receiving chess positions
    via stdin (as JSONL) and emitting moves via stdout.

    Protocol:
        Input (JSONL):
            {"type": "position", "fen": "...", "legal_moves": [...]}
            {"type": "shutdown"}

        Output (JSONL):
            {"type": "move", "uci": "e2e4", "fen_after": "..."}
            {"type": "ready"}
            {"type": "error", "message": "..."}

    Attributes:
        config: Dragon worker configuration
        state: Current runtime state
    """

    config: DragonWorkerConfig
    state: DragonState = field(default=DragonState.IDLE)
    _engine: Optional[chess.engine.SimpleEngine] = field(default=None, repr=False)
    _dragon_path: Optional[str] = field(default=None, repr=False)
    _board: chess.Board = field(default_factory=chess.Board, repr=False)
    _moves_made: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize the runtime and locate Dragon binary."""
        # Find Dragon binary
        if self.config.dragon_path:
            self._dragon_path = self.config.dragon_path
        else:
            self._dragon_path = _find_dragon_binary()

        if not self._dragon_path:
            raise RuntimeError(
                "Dragon binary not found. Please either:\n"
                "1. Download from https://komodochess.com/installation.htm\n"
                "2. Place binary in ./dragon/ directory\n"
                "3. Specify path via config.dragon_path"
            )

        # Verify binary exists
        if not Path(self._dragon_path).exists():
            raise RuntimeError(
                f"Dragon binary not found at: {self._dragon_path}"
            )

    def __enter__(self) -> "DragonWorkerRuntime":
        """Start the Dragon engine."""
        self._start_engine()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the Dragon engine."""
        self._stop_engine()

    def _start_engine(self) -> None:
        """Start the Dragon UCI engine."""
        if self._engine is not None:
            return

        logger.info(f"Starting Dragon engine from: {self._dragon_path}")
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._dragon_path)  # type: ignore[arg-type]

            # Configure skill level
            if self.config.skill_level is not None:
                self._engine.configure({"Skill": self.config.skill_level})
                logger.info(
                    f"Dragon configured: skill_level={self.config.skill_level}, "
                    f"estimated_elo={self.config.estimated_elo}"
                )

            self.state = DragonState.WAITING_FOR_POSITION
        except Exception as e:
            self.state = DragonState.ERROR
            raise RuntimeError(f"Failed to start Dragon engine: {e}") from e

    def _stop_engine(self) -> None:
        """Stop the Dragon UCI engine."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception as e:
                logger.warning(f"Error stopping Dragon engine: {e}")
            finally:
                self._engine = None
                self.state = DragonState.TERMINATED

    def get_best_move(
        self,
        fen: str,
        legal_moves: Optional[list[str]] = None,  # type: ignore[assignment]
    ) -> tuple[str, str]:
        """Get the best move for a given position.

        Args:
            fen: FEN string representing the board position
            legal_moves: Optional list of legal moves (for validation)

        Returns:
            Tuple of (uci_move, fen_after_move)

        Raises:
            RuntimeError: If engine is not started or move generation fails
        """
        if self._engine is None:
            raise RuntimeError("Dragon engine not started")

        self.state = DragonState.THINKING

        # Set up board from FEN
        if self.config.remove_history:
            self._board = chess.Board(fen)
        else:
            self._board.set_fen(fen)

        # Calculate time limit in seconds
        time_limit = self.config.time_limit_ms / 1000.0

        # Build search limits
        limit = chess.engine.Limit(time=time_limit)
        if self.config.depth is not None:
            limit = chess.engine.Limit(
                time=time_limit,
                depth=self.config.depth,
            )

        try:
            result = self._engine.play(self._board, limit)
            move = result.move

            if move is None:
                raise RuntimeError("Dragon returned no move")

            # Validate against legal moves if provided
            if legal_moves and move.uci() not in legal_moves:
                logger.warning(
                    f"Dragon suggested illegal move {move.uci()}, "
                    f"legal moves: {legal_moves}"
                )

            # Apply move to get resulting FEN
            self._board.push(move)
            fen_after = self._board.fen()

            self._moves_made += 1
            self.state = DragonState.MOVE_READY

            return move.uci(), fen_after

        except Exception as e:
            self.state = DragonState.ERROR
            raise RuntimeError(f"Dragon move generation failed: {e}") from e

    def run_interactive(self) -> dict[str, Any]:
        """Run the worker in interactive mode.

        Reads JSONL from stdin, writes JSONL to stdout.

        Returns:
            Summary dictionary with run statistics
        """
        logger.info("Starting Dragon interactive mode")

        with self:
            # Signal ready
            self._emit({"type": "ready", "config": self.config.to_dict()})

            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    self._emit({"type": "error", "message": f"Invalid JSON: {e}"})
                    continue

                response = self._handle_request(request)
                if response:
                    self._emit(response)

                if request.get("type") == "shutdown":
                    break

        return {
            "status": "completed",
            "moves_made": self._moves_made,
            "config": self.config.to_dict(),
        }

    def _handle_request(self, request: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Handle a single request.

        Args:
            request: Parsed JSON request

        Returns:
            Response dictionary or None
        """
        req_type = request.get("type", "")

        if req_type == "position":
            return self._handle_position(request)
        elif req_type == "shutdown":
            return {"type": "shutdown_ack"}
        elif req_type == "status":
            return {
                "type": "status",
                "state": self.state.value,
                "moves_made": self._moves_made,
            }
        else:
            return {"type": "error", "message": f"Unknown request type: {req_type}"}

    def _handle_position(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a position request.

        Args:
            request: Position request with 'fen' and optional 'legal_moves'

        Returns:
            Move response or error
        """
        fen = request.get("fen")
        if not fen:
            return {"type": "error", "message": "Missing 'fen' in position request"}

        legal_moves = request.get("legal_moves")

        try:
            uci_move, fen_after = self.get_best_move(fen, legal_moves)
            return {
                "type": "move",
                "uci": uci_move,
                "fen_after": fen_after,
                "move_number": self._moves_made,
            }
        except Exception as e:
            logger.exception("Error generating move")
            return {"type": "error", "message": str(e)}

    def _emit(self, data: dict[str, Any]) -> None:
        """Emit a JSONL response to stdout."""
        print(json.dumps(data), flush=True)
