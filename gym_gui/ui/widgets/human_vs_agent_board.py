"""Interactive chess board widget for Human vs Agent mode.

This widget wraps the _ChessBoardRenderer from board_game.py and provides
a standalone interface for the Human vs Agent gameplay mode.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from qtpy import QtCore, QtGui, QtWidgets

_LOG = logging.getLogger(__name__)


# Chess piece Unicode symbols
_CHESS_PIECES = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656",
    "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265A", "q": "\u265B", "r": "\u265C",
    "b": "\u265D", "n": "\u265E", "p": "\u265F",
}

# Colors
_LIGHT_SQUARE = QtGui.QColor(240, 217, 181)
_DARK_SQUARE = QtGui.QColor(181, 136, 99)
_SELECTED = QtGui.QColor(255, 255, 0, 150)
_LEGAL_MOVE = QtGui.QColor(0, 255, 0, 100)
_LAST_MOVE = QtGui.QColor(255, 255, 0, 80)
_CHECK = QtGui.QColor(255, 0, 0, 100)
_HOVER = QtGui.QColor(100, 100, 255, 50)


class InteractiveChessBoard(QtWidgets.QWidget):
    """Interactive chess board widget for Human vs Agent mode.

    This widget displays a chess board and handles mouse input for moves.
    It's designed to work with ChessGameController.

    Signals:
        move_made(str, str): Emitted when user makes a move (from_square, to_square)
    """

    move_made = QtCore.Signal(str, str)  # from_square, to_square

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Board state
        self._fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self._board: List[List[Optional[str]]] = [[None] * 8 for _ in range(8)]
        self._current_player: str = "white"

        # Interaction state
        self._selected_square: Optional[str] = None
        self._legal_moves: Set[str] = set()
        self._legal_destinations: Set[str] = set()
        self._last_move_from: Optional[str] = None
        self._last_move_to: Optional[str] = None
        self._hover_square: Optional[str] = None
        self._is_check: bool = False
        self._king_square: Optional[str] = None
        self._enabled: bool = True

        # Visual settings
        self._square_size: int = 70
        self._margin: int = 25

        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

        # Initialize board
        self._parse_fen(self._fen)

    def _update_minimum_size(self) -> None:
        size = 8 * self._square_size + 2 * self._margin
        self.setMinimumSize(size, size)

    # -------------------------------------------------------------------------
    # Public API for ChessGameController
    # -------------------------------------------------------------------------

    def set_position(self, fen: str) -> None:
        """Set the board position from FEN string."""
        self._fen = fen
        self._parse_fen(fen)
        self._selected_square = None
        self._legal_destinations.clear()
        self.update()

    def set_legal_moves(self, moves: List[str]) -> None:
        """Set the list of legal moves in UCI format."""
        self._legal_moves = set(moves)
        self._update_legal_destinations()
        self.update()

    def set_current_player(self, player: str) -> None:
        """Set the current player ('white' or 'black')."""
        self._current_player = player
        self.update()

    def set_last_move(self, from_sq: Optional[str], to_sq: Optional[str]) -> None:
        """Set the last move for highlighting."""
        self._last_move_from = from_sq
        self._last_move_to = to_sq
        self.update()

    def set_check(self, is_check: bool, king_square: Optional[str]) -> None:
        """Set whether the current player is in check."""
        self._is_check = is_check
        self._king_square = king_square
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable user input."""
        self._enabled = enabled
        if not enabled:
            self._selected_square = None
            self._legal_destinations.clear()
        self.update()

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _parse_fen(self, fen: str) -> None:
        """Parse FEN string into board array."""
        parts = fen.split()
        position = parts[0]
        self._current_player = "white" if len(parts) < 2 or parts[1] == "w" else "black"

        self._board = [[None] * 8 for _ in range(8)]
        row, col = 7, 0

        for char in position:
            if char == "/":
                row -= 1
                col = 0
            elif char.isdigit():
                col += int(char)
            else:
                if 0 <= row < 8 and 0 <= col < 8:
                    self._board[row][col] = char
                col += 1

    def _update_legal_destinations(self) -> None:
        """Update legal destinations for selected piece."""
        self._legal_destinations.clear()
        if self._selected_square is None:
            return
        for move in self._legal_moves:
            if move.startswith(self._selected_square):
                dest = move[2:4]
                self._legal_destinations.add(dest)

    def _square_to_coords(self, square: str) -> tuple[int, int]:
        col = ord(square[0]) - ord("a")
        row = int(square[1]) - 1
        return (row, col)

    def _coords_to_square(self, row: int, col: int) -> str:
        return f"{chr(ord('a') + col)}{row + 1}"

    def _pixel_to_square(self, pos: QtCore.QPointF) -> Optional[str]:
        x, y = int(pos.x()) - self._margin, int(pos.y()) - self._margin
        col = x // self._square_size
        row = 7 - (y // self._square_size)
        if 0 <= col < 8 and 0 <= row < 8:
            return self._coords_to_square(row, col)
        return None

    def _square_to_pixel(self, square: str) -> QtCore.QPoint:
        row, col = self._square_to_coords(square)
        x = self._margin + col * self._square_size
        y = self._margin + (7 - row) * self._square_size
        return QtCore.QPoint(x, y)

    def _has_piece_at(self, square: str) -> bool:
        row, col = self._square_to_coords(square)
        return self._board[row][col] is not None

    def _is_own_piece(self, square: str) -> bool:
        row, col = self._square_to_coords(square)
        piece = self._board[row][col]
        if piece is None:
            return False
        is_white_piece = piece.isupper()
        return (self._current_player == "white") == is_white_piece

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        square = self._pixel_to_square(event.position())
        if square is None:
            return super().mousePressEvent(event)

        if self._selected_square is None:
            # No piece selected - try to select one
            if self._has_piece_at(square) and self._is_own_piece(square):
                self._selected_square = square
                self._update_legal_destinations()
                self.update()
        else:
            # A piece is selected
            if square == self._selected_square:
                # Clicked same square - deselect
                self._selected_square = None
                self._legal_destinations.clear()
                self.update()
            elif square in self._legal_destinations:
                # Valid move - emit signal
                from_sq = self._selected_square
                self._selected_square = None
                self._legal_destinations.clear()
                self.move_made.emit(from_sq, square)
                self.update()
            elif self._has_piece_at(square) and self._is_own_piece(square):
                # Clicked another own piece - select it
                self._selected_square = square
                self._update_legal_destinations()
                self.update()
            else:
                # Clicked invalid square - deselect
                self._selected_square = None
                self._legal_destinations.clear()
                self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        square = self._pixel_to_square(event.position())
        if square != self._hover_square:
            self._hover_square = square
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_square = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        size = min(event.size().width(), event.size().height())
        board_space = size - 2 * self._margin
        self._square_size = max(20, board_space // 8)
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_squares(painter)
        self._draw_highlights(painter)
        self._draw_pieces(painter)
        self._draw_coordinates(painter)

        painter.end()

    def _draw_squares(self, painter: QtGui.QPainter) -> None:
        for row in range(8):
            for col in range(8):
                x = self._margin + col * self._square_size
                y = self._margin + (7 - row) * self._square_size
                is_light = (row + col) % 2 == 0
                color = _LIGHT_SQUARE if is_light else _DARK_SQUARE
                painter.fillRect(x, y, self._square_size, self._square_size, color)

    def _draw_highlights(self, painter: QtGui.QPainter) -> None:
        # Last move
        if self._last_move_from:
            pos = self._square_to_pixel(self._last_move_from)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _LAST_MOVE
            )
        if self._last_move_to:
            pos = self._square_to_pixel(self._last_move_to)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _LAST_MOVE
            )

        # Check
        if self._is_check and self._king_square:
            pos = self._square_to_pixel(self._king_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHECK
            )

        # Selected
        if self._selected_square:
            pos = self._square_to_pixel(self._selected_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _SELECTED
            )

        # Legal moves
        for dest in self._legal_destinations:
            pos = self._square_to_pixel(dest)
            center_x = pos.x() + self._square_size // 2
            center_y = pos.y() + self._square_size // 2

            if self._has_piece_at(dest):
                # Capture indicator - ring around the piece
                painter.setPen(QtGui.QPen(_LEGAL_MOVE, 4))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                radius = self._square_size // 2 - 4
            else:
                # Empty square - dot in center
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(_LEGAL_MOVE)
                radius = self._square_size // 6

            painter.drawEllipse(QtCore.QPoint(center_x, center_y), radius, radius)

        # Hover
        if self._hover_square and self._hover_square != self._selected_square:
            pos = self._square_to_pixel(self._hover_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _HOVER
            )

    def _draw_pieces(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", int(self._square_size * 0.7))
        painter.setFont(font)

        for row in range(8):
            for col in range(8):
                piece = self._board[row][col]
                if piece is None:
                    continue

                square = self._coords_to_square(row, col)
                pos = self._square_to_pixel(square)
                symbol = _CHESS_PIECES.get(piece, "?")

                # Draw piece shadow for better visibility
                painter.setPen(QtGui.QColor(0, 0, 0, 100))
                rect = QtCore.QRect(
                    pos.x() + 2, pos.y() + 2, self._square_size, self._square_size
                )
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, symbol)

                # Draw piece
                if piece.isupper():
                    painter.setPen(QtGui.QColor(255, 255, 255))
                else:
                    painter.setPen(QtGui.QColor(0, 0, 0))

                rect = QtCore.QRect(
                    pos.x(), pos.y(), self._square_size, self._square_size
                )
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, symbol)

    def _draw_coordinates(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(100, 100, 100))

        # File letters (a-h)
        for col in range(8):
            letter = chr(ord("a") + col)
            x = self._margin + col * self._square_size + self._square_size // 2 - 4
            y = self._margin + 8 * self._square_size + 15
            painter.drawText(x, y, letter)

        # Rank numbers (1-8)
        for row in range(8):
            number = str(row + 1)
            x = self._margin - 15
            y = self._margin + (7 - row) * self._square_size + self._square_size // 2 + 4
            painter.drawText(x, y, number)


__all__ = ["InteractiveChessBoard"]
