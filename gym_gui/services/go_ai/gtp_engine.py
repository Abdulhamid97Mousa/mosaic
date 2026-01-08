"""Go Text Protocol (GTP) engine handler for subprocess communication.

This module provides a low-level GTP client that communicates with Go engines
(KataGo, GNU Go, etc.) via subprocess stdin/stdout.

GTP Protocol Reference:
- Commands are sent as plain text lines
- Responses start with "= " (success) or "? " (error)
- Vertex format: Column letter (A-T, skipping I) + row number (1-19 from bottom)
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import List, Optional, Tuple

_LOG = logging.getLogger(__name__)


class GTPEngine:
    """Low-level GTP protocol handler via subprocess.

    This class manages communication with GTP-compatible Go engines
    through subprocess stdin/stdout pipes.

    Usage:
        engine = GTPEngine("/usr/bin/gnugo", "--mode", "gtp")
        if engine.start():
            engine.boardsize(19)
            engine.clear_board()
            move = engine.genmove("black")
            engine.quit()
    """

    def __init__(self, *command: str) -> None:
        """Initialize the GTP engine.

        Args:
            *command: Command and arguments to launch the engine
                     (e.g., "/usr/bin/gnugo", "--mode", "gtp")
        """
        self._command = list(command)
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start the GTP engine subprocess.

        Returns:
            True if engine started successfully.
        """
        if self._process is not None:
            _LOG.warning("GTP engine already running")
            return True

        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env={**os.environ, "LC_ALL": "C"},  # Ensure consistent output
            )
            _LOG.info(f"GTP engine started: {self._command[0]}")
            return True
        except FileNotFoundError:
            _LOG.error(f"GTP engine not found: {self._command[0]}")
            return False
        except Exception as e:
            _LOG.error(f"Failed to start GTP engine: {e}")
            return False

    def stop(self) -> None:
        """Stop the GTP engine subprocess."""
        if self._process is None:
            return

        try:
            # Try graceful quit first
            self.send_command("quit")
        except Exception:
            pass

        try:
            self._process.terminate()
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        except Exception as e:
            _LOG.warning(f"Error stopping GTP engine: {e}")
        finally:
            self._process = None
            _LOG.info("GTP engine stopped")

    def is_running(self) -> bool:
        """Check if the engine is running.

        Returns:
            True if engine subprocess is active.
        """
        return self._process is not None and self._process.poll() is None

    def send_command(self, command: str) -> str:
        """Send a GTP command and return the response.

        Args:
            command: GTP command (e.g., "boardsize 19", "genmove black")

        Returns:
            Response string (without "= " or "? " prefix).

        Raises:
            RuntimeError: If engine is not running or communication fails.
            GTPError: If engine returns an error response.
        """
        if not self.is_running():
            raise RuntimeError("GTP engine not running")

        with self._lock:
            try:
                assert self._process is not None
                assert self._process.stdin is not None
                assert self._process.stdout is not None

                # Send command
                self._process.stdin.write(command + "\n")
                self._process.stdin.flush()

                # Read response (may be multiple lines)
                response_lines = []
                while True:
                    line = self._process.stdout.readline()
                    if not line:
                        raise RuntimeError("GTP engine terminated unexpectedly")

                    line = line.rstrip("\n\r")

                    # Empty line marks end of response
                    if line == "":
                        break

                    response_lines.append(line)

                if not response_lines:
                    return ""

                # Parse response
                first_line = response_lines[0]
                if first_line.startswith("= "):
                    # Success response
                    response_lines[0] = first_line[2:]
                    return "\n".join(response_lines)
                elif first_line.startswith("="):
                    # Success with no content
                    return first_line[1:].strip()
                elif first_line.startswith("? "):
                    # Error response
                    error_msg = first_line[2:]
                    raise GTPError(error_msg)
                else:
                    # Unexpected format
                    return "\n".join(response_lines)

            except GTPError:
                raise
            except Exception as e:
                _LOG.error(f"GTP communication error: {e}")
                raise RuntimeError(f"GTP communication failed: {e}")

    # -------------------------------------------------------------------------
    # Standard GTP Commands
    # -------------------------------------------------------------------------

    def protocol_version(self) -> str:
        """Get GTP protocol version."""
        return self.send_command("protocol_version")

    def name(self) -> str:
        """Get engine name."""
        return self.send_command("name")

    def version(self) -> str:
        """Get engine version."""
        return self.send_command("version")

    def boardsize(self, size: int) -> bool:
        """Set board size.

        Args:
            size: Board size (9, 13, or 19)

        Returns:
            True if command succeeded.
        """
        try:
            self.send_command(f"boardsize {size}")
            return True
        except GTPError as e:
            _LOG.warning(f"boardsize failed: {e}")
            return False

    def clear_board(self) -> bool:
        """Clear the board for a new game.

        Returns:
            True if command succeeded.
        """
        try:
            self.send_command("clear_board")
            return True
        except GTPError as e:
            _LOG.warning(f"clear_board failed: {e}")
            return False

    def komi(self, value: float) -> bool:
        """Set komi value.

        Args:
            value: Komi compensation for white (e.g., 7.5)

        Returns:
            True if command succeeded.
        """
        try:
            self.send_command(f"komi {value}")
            return True
        except GTPError as e:
            _LOG.warning(f"komi failed: {e}")
            return False

    def play(self, color: str, vertex: str) -> bool:
        """Play a move.

        Args:
            color: "black" or "white" (or "b"/"w")
            vertex: Board position (e.g., "D4") or "pass"

        Returns:
            True if move was accepted.
        """
        try:
            self.send_command(f"play {color} {vertex}")
            return True
        except GTPError as e:
            _LOG.warning(f"play failed: {e}")
            return False

    def genmove(self, color: str) -> Optional[str]:
        """Generate a move for the given color.

        Args:
            color: "black" or "white" (or "b"/"w")

        Returns:
            Vertex string (e.g., "D4") or "pass", or None on error.
        """
        try:
            response = self.send_command(f"genmove {color}")
            return response.strip() if response else None
        except GTPError as e:
            _LOG.warning(f"genmove failed: {e}")
            return None

    def quit(self) -> None:
        """Send quit command to engine."""
        try:
            self.send_command("quit")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "GTPEngine":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class GTPError(Exception):
    """Exception raised when GTP engine returns an error."""

    pass


# -----------------------------------------------------------------------------
# Coordinate Conversion Utilities
# -----------------------------------------------------------------------------

# GTP uses letters A-T (skipping I) for columns
_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"  # Note: no 'I'


def action_to_vertex(action: int, board_size: int) -> str:
    """Convert action index to GTP vertex.

    Args:
        action: Action index (0 to board_size^2 - 1 for placements,
                board_size^2 for pass)
        board_size: Board size (9, 13, or 19)

    Returns:
        GTP vertex string (e.g., "D4") or "pass"
    """
    if action == board_size ** 2:
        return "pass"

    row = action // board_size
    col = action % board_size

    # GTP rows are 1-indexed from bottom
    gtp_row = board_size - row
    # GTP columns skip 'I'
    gtp_col = _GTP_COLUMNS[col]

    return f"{gtp_col}{gtp_row}"


def vertex_to_action(vertex: str, board_size: int) -> int:
    """Convert GTP vertex to action index.

    Args:
        vertex: GTP vertex string (e.g., "D4") or "pass"/"PASS"
        board_size: Board size (9, 13, or 19)

    Returns:
        Action index (0 to board_size^2 - 1 for placements, board_size^2 for pass)

    Raises:
        ValueError: If vertex format is invalid.
    """
    vertex = vertex.strip().upper()

    if vertex in ("PASS", "RESIGN"):
        return board_size ** 2

    if len(vertex) < 2:
        raise ValueError(f"Invalid vertex: {vertex}")

    col_char = vertex[0]
    try:
        row_num = int(vertex[1:])
    except ValueError:
        raise ValueError(f"Invalid vertex row: {vertex}")

    # Find column index (skip 'I')
    if col_char not in _GTP_COLUMNS:
        raise ValueError(f"Invalid vertex column: {col_char}")
    col = _GTP_COLUMNS.index(col_char)

    # Convert GTP row (1-indexed from bottom) to 0-indexed from top
    row = board_size - row_num

    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Vertex out of bounds: {vertex}")

    return row * board_size + col


def coords_to_vertex(row: int, col: int, board_size: int) -> str:
    """Convert (row, col) coordinates to GTP vertex.

    Args:
        row: Row index (0-indexed from top)
        col: Column index (0-indexed from left)
        board_size: Board size

    Returns:
        GTP vertex string (e.g., "D4")
    """
    gtp_row = board_size - row
    gtp_col = _GTP_COLUMNS[col]
    return f"{gtp_col}{gtp_row}"


def vertex_to_coords(vertex: str, board_size: int) -> Tuple[int, int]:
    """Convert GTP vertex to (row, col) coordinates.

    Args:
        vertex: GTP vertex string (e.g., "D4")
        board_size: Board size

    Returns:
        Tuple of (row, col) 0-indexed from top-left.

    Raises:
        ValueError: If vertex format is invalid.
    """
    action = vertex_to_action(vertex, board_size)
    if action == board_size ** 2:
        raise ValueError("Pass has no coordinates")
    row = action // board_size
    col = action % board_size
    return row, col


def board_to_gtp_moves(
    board: List[List[int]], board_size: int
) -> Tuple[List[str], List[str]]:
    """Extract black and white moves from board state.

    Args:
        board: NxN grid with 0=empty, 1=black, 2=white
        board_size: Board size

    Returns:
        Tuple of (black_moves, white_moves) as GTP vertex strings.
    """
    black_moves = []
    white_moves = []

    for row in range(board_size):
        for col in range(board_size):
            stone = board[row][col]
            if stone == 1:
                black_moves.append(coords_to_vertex(row, col, board_size))
            elif stone == 2:
                white_moves.append(coords_to_vertex(row, col, board_size))

    return black_moves, white_moves


__all__ = [
    "GTPEngine",
    "GTPError",
    "action_to_vertex",
    "vertex_to_action",
    "coords_to_vertex",
    "vertex_to_coords",
    "board_to_gtp_moves",
]
