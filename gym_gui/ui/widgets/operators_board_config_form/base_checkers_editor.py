"""Base checkers/draughts board configuration editor.

This module provides shared logic for all checkers/draughts variants:
- American Checkers (8x8)
- Russian Checkers (8x8)
- International Draughts (10x10)

All variants share the same 4 piece types but differ in board size and rules.
"""

import logging
from typing import List, Tuple, Optional, Dict

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal

from .base import (
    BoardConfigDialog,
    EditableBoardWidget,
    PieceTrayWidget,
    BoardState,
    GamePiece,
)

_LOGGER = logging.getLogger(__name__)

# Checkers piece definitions (same for all variants)
# Using Unicode symbols for checkers pieces
CHECKERS_PIECES: Dict[str, GamePiece] = {
    "b": GamePiece("man", "black", "⛀", "b"),       # Black man (regular piece)
    "B": GamePiece("king", "black", "⛁", "B"),      # Black king (crowned)
    "w": GamePiece("man", "white", "⛂", "w"),       # White man (regular piece)
    "W": GamePiece("king", "white", "⛃", "W"),      # White king (crowned)
}

# Alternative symbols if the above don't render well
CHECKERS_PIECES_ALT: Dict[str, GamePiece] = {
    "b": GamePiece("man", "black", "●", "b"),       # Black man
    "B": GamePiece("king", "black", "◉", "B"),      # Black king
    "w": GamePiece("man", "white", "○", "w"),       # White man
    "W": GamePiece("king", "white", "◎", "W"),      # White king
}


def _piece_to_notation(piece: GamePiece) -> str:
    """Convert GamePiece to checkers notation character."""
    for notation, gp in CHECKERS_PIECES.items():
        if gp.piece_type == piece.piece_type and gp.color == piece.color:
            return notation
    return ""


def _notation_to_piece(notation: str) -> Optional[GamePiece]:
    """Convert checkers notation character to GamePiece."""
    return CHECKERS_PIECES.get(notation)


class CheckersBoardState(BoardState):
    """Checkers board state for any board size.

    Supports 8x8 (American/Russian) and 10x10 (International) boards.
    Pieces only exist on dark squares (where row + col is odd).

    Notation format:
    - String of characters representing each dark square from top-left to bottom-right
    - 'b' = black man, 'B' = black king
    - 'w' = white man, 'W' = white king
    - '.' = empty

    Example 8x8 starting position (32 dark squares):
    "bbbbbbbbbbbb........wwwwwwwwwwww"
    """

    def __init__(self, size: int = 8, notation: Optional[str] = None):
        """Initialize checkers board state.

        Args:
            size: Board size (8 or 10)
            notation: Optional initial notation string
        """
        self._size = size
        self._num_dark_squares = (size * size) // 2
        # Internal representation: 2D grid with piece values
        # 0=empty, 1=black man, 2=black king, 3=white man, 4=white king
        self._board: List[List[int]] = [[0] * size for _ in range(size)]

        if notation:
            self.from_notation(notation)
        else:
            self._init_standard_position()

    def _init_standard_position(self) -> None:
        """Initialize standard starting position."""
        # Number of rows with pieces at each end
        piece_rows = 3 if self._size == 8 else 4

        for row in range(self._size):
            for col in range(self._size):
                # Only dark squares (where row + col is odd)
                if (row + col) % 2 == 1:
                    if row < piece_rows:
                        self._board[row][col] = 1  # Black man
                    elif row >= self._size - piece_rows:
                        self._board[row][col] = 3  # White man
                    else:
                        self._board[row][col] = 0  # Empty

    def _is_dark_square(self, row: int, col: int) -> bool:
        """Check if a square is a dark (playable) square."""
        return (row + col) % 2 == 1

    def get_piece(self, row: int, col: int) -> Optional[GamePiece]:
        """Get piece at position."""
        if not (0 <= row < self._size and 0 <= col < self._size):
            return None

        value = self._board[row][col]
        if value == 0:
            return None
        elif value == 1:
            return CHECKERS_PIECES["b"]
        elif value == 2:
            return CHECKERS_PIECES["B"]
        elif value == 3:
            return CHECKERS_PIECES["w"]
        elif value == 4:
            return CHECKERS_PIECES["W"]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[GamePiece]) -> None:
        """Set piece at position (only on dark squares)."""
        if not (0 <= row < self._size and 0 <= col < self._size):
            return

        # Only allow pieces on dark squares
        if not self._is_dark_square(row, col):
            _LOGGER.debug(f"Cannot place piece on light square ({row}, {col})")
            return

        if piece is None:
            self._board[row][col] = 0
        else:
            notation = _piece_to_notation(piece)
            if notation == "b":
                self._board[row][col] = 1
            elif notation == "B":
                self._board[row][col] = 2
            elif notation == "w":
                self._board[row][col] = 3
            elif notation == "W":
                self._board[row][col] = 4

    def to_notation(self) -> str:
        """Convert board to notation string.

        Returns string representing dark squares from top-left to bottom-right,
        row by row.
        """
        chars = []
        for row in range(self._size):
            for col in range(self._size):
                if self._is_dark_square(row, col):
                    value = self._board[row][col]
                    if value == 0:
                        chars.append(".")
                    elif value == 1:
                        chars.append("b")
                    elif value == 2:
                        chars.append("B")
                    elif value == 3:
                        chars.append("w")
                    elif value == 4:
                        chars.append("W")
        return "".join(chars)

    def from_notation(self, notation: str) -> None:
        """Load board from notation string."""
        notation = notation.strip()

        # Clear board first
        self._board = [[0] * self._size for _ in range(self._size)]

        if len(notation) != self._num_dark_squares:
            _LOGGER.warning(
                f"Invalid notation length: {len(notation)}, expected {self._num_dark_squares}"
            )
            # Try to use what we have

        idx = 0
        for row in range(self._size):
            for col in range(self._size):
                if self._is_dark_square(row, col) and idx < len(notation):
                    ch = notation[idx]
                    if ch == "b":
                        self._board[row][col] = 1
                    elif ch == "B":
                        self._board[row][col] = 2
                    elif ch == "w":
                        self._board[row][col] = 3
                    elif ch == "W":
                        self._board[row][col] = 4
                    else:
                        self._board[row][col] = 0
                    idx += 1

    def get_dimensions(self) -> Tuple[int, int]:
        """Get board dimensions."""
        return (self._size, self._size)

    def clear(self) -> None:
        """Clear all pieces from the board."""
        self._board = [[0] * self._size for _ in range(self._size)]

    def copy(self) -> "CheckersBoardState":
        """Create a deep copy."""
        new_state = CheckersBoardState(self._size)
        new_state._board = [row[:] for row in self._board]
        return new_state

    @property
    def size(self) -> int:
        """Get board size."""
        return self._size


class EditableCheckersBoard(EditableBoardWidget):
    """Editable checkers board with drag-and-drop piece manipulation.

    Features:
    - Drag pieces between dark squares
    - Drag pieces off board to remove them
    - Click to place piece from tray (dark squares only)
    - Coordinate labels (1-8 or 1-10)
    - Visual distinction between light and dark squares
    """

    # Board colors
    LIGHT_SQUARE = QtGui.QColor(240, 217, 181)  # Beige (non-playable)
    DARK_SQUARE = QtGui.QColor(139, 90, 43)     # Brown (playable)
    HIGHLIGHT_COLOR = QtGui.QColor(255, 255, 0, 100)
    DROP_HIGHLIGHT = QtGui.QColor(100, 200, 100, 150)
    INVALID_DROP = QtGui.QColor(200, 100, 100, 100)
    COORD_COLOR = QtGui.QColor(80, 80, 80)

    def __init__(self, size: int = 8, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._size = size
        self._state = CheckersBoardState(size)
        self.setMinimumSize(400, 400)

    def set_state(self, state: BoardState) -> None:
        """Set the board state."""
        if isinstance(state, CheckersBoardState):
            self._state = state
            self._size = state.size
        else:
            # Try to convert from notation
            self._state = CheckersBoardState(self._size, state.to_notation())
        self.update()

    def get_state(self) -> BoardState:
        """Get the current board state."""
        return self._state

    def _get_square_size(self) -> int:
        """Calculate square size based on widget dimensions."""
        margin = 25
        available = min(self.width() - margin, self.height() - margin)
        return max(35, available // self._size)

    def _get_board_offset(self) -> Tuple[int, int]:
        """Get offset to center the board."""
        sq_size = self._get_square_size()
        board_size = sq_size * self._size
        offset_x = (self.width() - board_size) // 2
        offset_y = (self.height() - board_size) // 2
        return (max(20, offset_x), max(20, offset_y))

    def _pos_to_square(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        """Convert widget position to board coordinates."""
        sq_size = self._get_square_size()
        offset_x, offset_y = self._get_board_offset()
        col = (pos.x() - offset_x) // sq_size
        row = (pos.y() - offset_y) // sq_size
        return (row, col)

    def _square_to_rect(self, row: int, col: int) -> QtCore.QRect:
        """Get rectangle for a board square."""
        sq_size = self._get_square_size()
        offset_x, offset_y = self._get_board_offset()
        return QtCore.QRect(
            offset_x + col * sq_size,
            offset_y + row * sq_size,
            sq_size,
            sq_size
        )

    def _is_valid_square(self, row: int, col: int) -> bool:
        """Check if coordinates are within board bounds."""
        return 0 <= row < self._size and 0 <= col < self._size

    def _is_dark_square(self, row: int, col: int) -> bool:
        """Check if a square is dark (playable)."""
        return (row + col) % 2 == 1

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the checkers board."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        sq_size = self._get_square_size()
        offset_x, offset_y = self._get_board_offset()

        # Draw board squares
        for row in range(self._size):
            for col in range(self._size):
                rect = self._square_to_rect(row, col)

                # Square color - dark squares are playable
                is_dark = self._is_dark_square(row, col)
                color = self.DARK_SQUARE if is_dark else self.LIGHT_SQUARE
                painter.fillRect(rect, color)

                # Highlight for drag target (only dark squares)
                if (row, col) in self._highlighted_squares:
                    if is_dark:
                        painter.fillRect(rect, self.DROP_HIGHLIGHT)
                    else:
                        painter.fillRect(rect, self.INVALID_DROP)

                # Draw piece (skip if being dragged from this square)
                if self._dragging_piece and self._dragging_piece[1:] == (row, col):
                    continue

                piece = self._state.get_piece(row, col)
                if piece:
                    self._draw_piece(painter, rect, piece)

        # Draw board border
        board_rect = QtCore.QRect(
            offset_x, offset_y,
            sq_size * self._size, sq_size * self._size
        )
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 2))
        painter.drawRect(board_rect)

        # Draw coordinates
        painter.setPen(self.COORD_COLOR)
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for i in range(self._size):
            # Column numbers below board
            col_num = str(i + 1)
            x = offset_x + i * sq_size + sq_size // 2 - 4
            y = offset_y + self._size * sq_size + 14
            painter.drawText(x, y, col_num)

            # Row numbers left of board
            row_num = str(i + 1)
            x = offset_x - 14
            y = offset_y + i * sq_size + sq_size // 2 + 4
            painter.drawText(x, y, row_num)

        # Draw piece being dragged
        if self._dragging_piece and self._drag_pos:
            piece, _, _ = self._dragging_piece
            drag_rect = QtCore.QRect(
                self._drag_pos.x() - sq_size // 2,
                self._drag_pos.y() - sq_size // 2,
                sq_size,
                sq_size
            )
            painter.setOpacity(0.8)
            self._draw_piece(painter, drag_rect, piece)
            painter.setOpacity(1.0)

        # Draw cursor indicator for incoming piece from tray
        if self._incoming_piece and not self._dragging_piece:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            row, col = self._pos_to_square(pos)
            if self._is_valid_square(row, col) and self._is_dark_square(row, col):
                rect = self._square_to_rect(row, col)
                painter.setOpacity(0.5)
                self._draw_piece(painter, rect, self._incoming_piece)
                painter.setOpacity(1.0)

    def _draw_piece(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        piece: GamePiece
    ) -> None:
        """Draw a checkers piece in the given rectangle."""
        # Calculate piece size (slightly smaller than square)
        margin = int(rect.width() * 0.1)
        piece_rect = rect.adjusted(margin, margin, -margin, -margin)

        # Determine colors based on piece color
        if piece.color == "black":
            fill_color = QtGui.QColor(40, 40, 40)
            border_color = QtGui.QColor(20, 20, 20)
            highlight_color = QtGui.QColor(80, 80, 80)
        else:  # white
            fill_color = QtGui.QColor(240, 240, 240)
            border_color = QtGui.QColor(180, 180, 180)
            highlight_color = QtGui.QColor(255, 255, 255)

        # Draw piece as circle
        painter.setBrush(QtGui.QBrush(fill_color))
        painter.setPen(QtGui.QPen(border_color, 2))
        painter.drawEllipse(piece_rect)

        # Draw highlight (3D effect)
        highlight_rect = piece_rect.adjusted(
            int(piece_rect.width() * 0.2),
            int(piece_rect.height() * 0.1),
            -int(piece_rect.width() * 0.4),
            -int(piece_rect.height() * 0.5)
        )
        painter.setBrush(QtGui.QBrush(highlight_color))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setOpacity(0.3)
        painter.drawEllipse(highlight_rect)
        painter.setOpacity(1.0)

        # Draw crown for king pieces
        if piece.piece_type == "king":
            self._draw_crown(painter, piece_rect, piece.color)

    def _draw_crown(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        color: str
    ) -> None:
        """Draw a crown symbol on king pieces."""
        # Crown color contrasts with piece color
        crown_color = QtGui.QColor(255, 215, 0)  # Gold

        # Calculate crown position (center of piece)
        center_x = rect.center().x()
        center_y = rect.center().y()
        crown_size = int(rect.width() * 0.4)

        # Draw crown using "♔" or "K" symbol
        painter.setPen(crown_color)
        font = painter.font()
        font.setPointSize(int(crown_size * 0.8))
        font.setBold(True)
        painter.setFont(font)

        crown_rect = QtCore.QRect(
            center_x - crown_size // 2,
            center_y - crown_size // 2,
            crown_size,
            crown_size
        )
        painter.drawText(crown_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "♔")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press - start drag or place piece from tray."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        row, col = self._pos_to_square(event.pos())

        # If we have an incoming piece from tray, place it (only on dark squares)
        if self._incoming_piece:
            if self._is_valid_square(row, col) and self._is_dark_square(row, col):
                self._state.set_piece(row, col, self._incoming_piece)
                self._incoming_piece = None
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                self.update()
                self.board_changed.emit(self._state.to_notation())
            return

        # Otherwise, start dragging if there's a piece on a dark square
        if not self._is_valid_square(row, col) or not self._is_dark_square(row, col):
            return

        piece = self._state.get_piece(row, col)
        if piece:
            self._dragging_piece = (piece, row, col)
            self._drag_pos = event.pos()
            self._state.set_piece(row, col, None)
            self.piece_picked_up.emit(piece, row, col)
            # Highlight only dark squares as valid drop targets
            self._highlighted_squares = [
                (r, c) for r in range(self._size) for c in range(self._size)
                if self._is_dark_square(r, c)
            ]
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move - update drag position."""
        if self._dragging_piece:
            self._drag_pos = event.pos()
            self.update()
        elif self._incoming_piece:
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release - drop piece."""
        if not self._dragging_piece:
            return

        piece, source_row, source_col = self._dragging_piece
        row, col = self._pos_to_square(event.pos())

        # Check if dropped on valid dark square
        if self._is_valid_square(row, col) and self._is_dark_square(row, col):
            self._state.set_piece(row, col, piece)
            self.piece_dropped.emit(row, col)
        else:
            # Dropped outside board or on light square - send to tray
            self.piece_removed.emit(piece)

        # Clear drag state
        self._dragging_piece = None
        self._drag_pos = None
        self._highlighted_squares = []
        self.update()
        self.board_changed.emit(self._state.to_notation())

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handle mouse leaving widget."""
        if self._incoming_piece:
            self.update()
        super().leaveEvent(event)


class CheckersPieceTray(PieceTrayWidget):
    """Piece tray for checkers showing available pieces by color.

    Displays clickable buttons for each piece type (man and king for each color).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._piece_counts: Dict[str, int] = {}
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._selected_piece: Optional[str] = None
        self._setup_ui()
        self.reset()

    def _setup_ui(self) -> None:
        """Create the tray UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Click a piece, then click on a dark square to place it.\n"
            "Drag pieces on the board to move them.\n"
            "Drag off the board to remove."
        )
        instructions.setStyleSheet("color: #666; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Black pieces
        black_group = QtWidgets.QGroupBox("Black Pieces")
        black_layout = QtWidgets.QHBoxLayout(black_group)
        black_layout.setSpacing(8)

        for notation in ["b", "B"]:
            btn = self._create_piece_button(notation)
            black_layout.addWidget(btn)
            self._buttons[notation] = btn

        layout.addWidget(black_group)

        # White pieces
        white_group = QtWidgets.QGroupBox("White Pieces")
        white_layout = QtWidgets.QHBoxLayout(white_group)
        white_layout.setSpacing(8)

        for notation in ["w", "W"]:
            btn = self._create_piece_button(notation)
            white_layout.addWidget(btn)
            self._buttons[notation] = btn

        layout.addWidget(white_group)

        # Clear selection button
        clear_btn = QtWidgets.QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._clear_selection)
        layout.addWidget(clear_btn)

        layout.addStretch()

    def _create_piece_button(self, notation: str) -> QtWidgets.QPushButton:
        """Create a button for a piece type."""
        piece = CHECKERS_PIECES[notation]

        # Use descriptive text instead of just symbol
        if piece.piece_type == "man":
            label = "Man"
        else:
            label = "King"

        btn = QtWidgets.QPushButton(label)
        btn.setFixedSize(70, 70)
        btn.setFont(QtGui.QFont("", 12))
        btn.setToolTip(f"{piece.color.title()} {piece.piece_type.title()}")
        btn.setCheckable(True)

        # Style based on piece color
        if piece.color == "black":
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 2px solid #202020;
                    border-radius: 35px;
                }
                QPushButton:checked {
                    border: 3px solid #4CAF50;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
            """)
        else:  # white
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    color: black;
                    border: 2px solid #b0b0b0;
                    border-radius: 35px;
                }
                QPushButton:checked {
                    border: 3px solid #4CAF50;
                }
                QPushButton:hover {
                    background-color: #ffffff;
                }
            """)

        btn.clicked.connect(lambda checked, n=notation: self._on_piece_clicked(n))
        return btn

    def _on_piece_clicked(self, notation: str) -> None:
        """Handle piece button click."""
        # Uncheck other buttons
        for n, btn in self._buttons.items():
            if n != notation:
                btn.setChecked(False)

        if self._buttons[notation].isChecked():
            self._selected_piece = notation
            piece = CHECKERS_PIECES[notation]
            self.piece_selected.emit(piece)
        else:
            self._selected_piece = None
            self.piece_selected.emit(None)

    def _clear_selection(self) -> None:
        """Clear the current piece selection."""
        for btn in self._buttons.values():
            btn.setChecked(False)
        self._selected_piece = None
        self.piece_selected.emit(None)

    def get_available_pieces(self) -> List[GamePiece]:
        """Get all available pieces."""
        return list(CHECKERS_PIECES.values())

    def add_piece(self, piece: GamePiece) -> None:
        """Add piece to tray (when removed from board)."""
        notation = _piece_to_notation(piece)
        if notation:
            self._piece_counts[notation] = self._piece_counts.get(notation, 0) + 1

    def remove_piece(self, piece: GamePiece) -> None:
        """Remove piece from tray (when placed on board)."""
        notation = _piece_to_notation(piece)
        if notation and self._piece_counts.get(notation, 0) > 0:
            self._piece_counts[notation] -= 1

    def reset(self) -> None:
        """Reset tray to have unlimited pieces available."""
        self._piece_counts = {n: 99 for n in CHECKERS_PIECES}
        self._clear_selection()


class BaseCheckersConfigDialog(BoardConfigDialog):
    """Base configuration dialog for checkers/draughts variants.

    Subclasses specify board size, variant name, and presets.
    """

    def __init__(
        self,
        initial_state: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        board_size: int = 8,
        variant_name: str = "Checkers"
    ):
        self._board_size = board_size
        self._variant_name = variant_name
        self._num_dark_squares = (board_size * board_size) // 2
        super().__init__(initial_state, parent)

    def _get_title(self) -> str:
        return f"Configure {self._variant_name} Position"

    def _get_notation_name(self) -> str:
        return "Position"

    def _get_notation_placeholder(self) -> str:
        # Standard starting position
        piece_rows = 3 if self._board_size == 8 else 4
        pieces_per_side = piece_rows * (self._board_size // 2)
        empty_squares = self._num_dark_squares - (2 * pieces_per_side)
        return "b" * pieces_per_side + "." * empty_squares + "w" * pieces_per_side

    def _create_board_widget(self) -> EditableBoardWidget:
        return EditableCheckersBoard(self._board_size, self)

    def _create_piece_tray(self) -> PieceTrayWidget:
        return CheckersPieceTray(self)

    def _get_presets(self) -> List[Tuple[str, str]]:
        """Return preset positions - override in subclasses for variant-specific presets."""
        piece_rows = 3 if self._board_size == 8 else 4
        pieces_per_side = piece_rows * (self._board_size // 2)
        empty_squares = self._num_dark_squares - (2 * pieces_per_side)

        standard = "b" * pieces_per_side + "." * empty_squares + "w" * pieces_per_side
        empty = "." * self._num_dark_squares

        return [
            ("Standard", standard),
            ("Empty", empty),
        ]

    def _validate_notation(self, notation: str) -> bool:
        """Validate the notation string."""
        notation = notation.strip()

        # Check length
        if len(notation) != self._num_dark_squares:
            return False

        # Check characters
        valid_chars = set("bBwW.")
        return all(ch in valid_chars for ch in notation)

    def _get_validation_error(self, notation: str) -> str:
        """Return error message for invalid notation."""
        notation = notation.strip()

        if len(notation) != self._num_dark_squares:
            return (
                f"Invalid length: {len(notation)} characters.\n"
                f"Expected {self._num_dark_squares} characters for {self._board_size}x{self._board_size} board.\n"
                f"(One character per dark square)"
            )

        invalid_chars = set(notation) - set("bBwW.")
        if invalid_chars:
            return (
                f"Invalid characters: {invalid_chars}\n"
                f"Valid characters: b (black man), B (black king), "
                f"w (white man), W (white king), . (empty)"
            )

        return "Unknown validation error"

    def _create_state_from_notation(self, notation: str) -> BoardState:
        return CheckersBoardState(self._board_size, notation)


__all__ = [
    "CHECKERS_PIECES",
    "CheckersBoardState",
    "EditableCheckersBoard",
    "CheckersPieceTray",
    "BaseCheckersConfigDialog",
]
