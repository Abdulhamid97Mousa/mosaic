"""Chess-specific board configuration dialog.

This module provides the chess implementation of the board configuration
system, including an editable chess board with drag-and-drop piece
manipulation and FEN notation support.

Uses the python-chess library for board representation and validation.
"""

import logging
from typing import List, Tuple, Optional, Dict

import chess
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

# Standard FEN positions
STANDARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EMPTY_FEN = "8/8/8/8/8/8/8/8 w - - 0 1"
ENDGAME_KR_K = "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1"
ENDGAME_KQ_K = "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"

# Chess piece definitions with Unicode symbols
CHESS_PIECES: Dict[str, GamePiece] = {
    "K": GamePiece("king", "white", "\u2654", "K"),      # ♔
    "Q": GamePiece("queen", "white", "\u2655", "Q"),     # ♕
    "R": GamePiece("rook", "white", "\u2656", "R"),      # ♖
    "B": GamePiece("bishop", "white", "\u2657", "B"),    # ♗
    "N": GamePiece("knight", "white", "\u2658", "N"),    # ♘
    "P": GamePiece("pawn", "white", "\u2659", "P"),      # ♙
    "k": GamePiece("king", "black", "\u265A", "k"),      # ♚
    "q": GamePiece("queen", "black", "\u265B", "q"),     # ♛
    "r": GamePiece("rook", "black", "\u265C", "r"),      # ♜
    "b": GamePiece("bishop", "black", "\u265D", "b"),    # ♝
    "n": GamePiece("knight", "black", "\u265E", "n"),    # ♞
    "p": GamePiece("pawn", "black", "\u265F", "p"),      # ♟
}


def _piece_to_notation(piece: GamePiece) -> str:
    """Convert GamePiece to chess notation character."""
    for notation, gp in CHESS_PIECES.items():
        if gp.piece_type == piece.piece_type and gp.color == piece.color:
            return notation
    return ""


def _notation_to_piece(notation: str) -> Optional[GamePiece]:
    """Convert chess notation character to GamePiece."""
    return CHESS_PIECES.get(notation)


class ChessBoardState(BoardState):
    """Chess board state using python-chess library.

    Wraps a chess.Board instance and provides the BoardState interface
    for manipulation and FEN conversion.
    """

    def __init__(self, fen: str = STANDARD_FEN):
        """Initialize with a FEN position.

        Args:
            fen: FEN notation string (defaults to standard starting position)
        """
        self._board = chess.Board(fen)

    def get_piece(self, row: int, col: int) -> Optional[GamePiece]:
        """Get piece at position (0,0 = top-left = a8)."""
        # Convert from display coordinates to chess coordinates
        # Display: row 0 = rank 8, col 0 = file a
        square = chess.square(col, 7 - row)
        piece = self._board.piece_at(square)
        if piece:
            return CHESS_PIECES.get(piece.symbol())
        return None

    def set_piece(self, row: int, col: int, piece: Optional[GamePiece]) -> None:
        """Set piece at position (0,0 = top-left = a8)."""
        square = chess.square(col, 7 - row)
        if piece is None:
            self._board.remove_piece_at(square)
        else:
            notation = _piece_to_notation(piece)
            if notation:
                self._board.set_piece_at(square, chess.Piece.from_symbol(notation))

    def to_notation(self) -> str:
        """Return FEN string."""
        return self._board.fen()

    def from_notation(self, notation: str) -> None:
        """Load position from FEN string."""
        self._board.set_fen(notation)

    def get_dimensions(self) -> Tuple[int, int]:
        """Chess is always 8x8."""
        return (8, 8)

    def clear(self) -> None:
        """Clear all pieces from the board."""
        self._board.clear()

    def copy(self) -> "ChessBoardState":
        """Create a deep copy."""
        return ChessBoardState(self._board.fen())

    @property
    def board(self) -> chess.Board:
        """Access underlying chess.Board for advanced operations."""
        return self._board


class EditableChessBoard(EditableBoardWidget):
    """Editable chess board with drag-and-drop piece manipulation.

    Features:
    - Drag pieces between squares
    - Drag pieces off board to remove them
    - Click to place piece from tray
    - Coordinate labels (a-h, 1-8)
    - Square highlighting during drag
    """

    # Board colors matching InteractiveChessBoard
    LIGHT_SQUARE = QtGui.QColor(240, 217, 181)
    DARK_SQUARE = QtGui.QColor(181, 136, 99)
    HIGHLIGHT_COLOR = QtGui.QColor(255, 255, 0, 100)
    DROP_HIGHLIGHT = QtGui.QColor(100, 200, 100, 150)
    COORD_COLOR = QtGui.QColor(80, 80, 80)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state = ChessBoardState()
        self.setMinimumSize(400, 400)

    def set_state(self, state: BoardState) -> None:
        """Set the board state."""
        if isinstance(state, ChessBoardState):
            self._state = state
        else:
            # Convert generic BoardState to ChessBoardState
            self._state = ChessBoardState(state.to_notation())
        self.update()

    def get_state(self) -> BoardState:
        """Get the current board state."""
        return self._state

    def _get_square_size(self) -> int:
        """Calculate square size based on widget dimensions."""
        # Leave margin for coordinates
        margin = 20
        available = min(self.width() - margin, self.height() - margin)
        return max(40, available // 8)

    def _get_board_offset(self) -> Tuple[int, int]:
        """Get offset to center the board."""
        sq_size = self._get_square_size()
        board_size = sq_size * 8
        offset_x = (self.width() - board_size) // 2
        offset_y = (self.height() - board_size) // 2
        return (max(15, offset_x), max(15, offset_y))

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
        return 0 <= row < 8 and 0 <= col < 8

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the chess board."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        sq_size = self._get_square_size()
        offset_x, offset_y = self._get_board_offset()

        # Draw board squares
        for row in range(8):
            for col in range(8):
                rect = self._square_to_rect(row, col)

                # Square color
                is_light = (row + col) % 2 == 0
                color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                painter.fillRect(rect, color)

                # Highlight for drag target
                if (row, col) in self._highlighted_squares:
                    painter.fillRect(rect, self.DROP_HIGHLIGHT)

                # Draw piece (skip if being dragged from this square)
                if self._dragging_piece and self._dragging_piece[1:] == (row, col):
                    continue

                piece = self._state.get_piece(row, col)
                if piece:
                    self._draw_piece(painter, rect, piece)

        # Draw coordinates
        painter.setPen(self.COORD_COLOR)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        for i in range(8):
            # File letters (a-h) below board
            file_letter = chr(ord("a") + i)
            x = offset_x + i * sq_size + sq_size // 2 - 4
            y = offset_y + 8 * sq_size + 14
            painter.drawText(x, y, file_letter)

            # Rank numbers (8-1) left of board
            rank_number = str(8 - i)
            x = offset_x - 14
            y = offset_y + i * sq_size + sq_size // 2 + 4
            painter.drawText(x, y, rank_number)

        # Draw piece being dragged
        if self._dragging_piece and self._drag_pos:
            piece, _, _ = self._dragging_piece
            drag_rect = QtCore.QRect(
                self._drag_pos.x() - sq_size // 2,
                self._drag_pos.y() - sq_size // 2,
                sq_size,
                sq_size
            )
            # Draw with slight transparency
            painter.setOpacity(0.8)
            self._draw_piece(painter, drag_rect, piece)
            painter.setOpacity(1.0)

        # Draw cursor indicator for incoming piece from tray
        if self._incoming_piece and not self._dragging_piece:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            row, col = self._pos_to_square(pos)
            if self._is_valid_square(row, col):
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
        """Draw a chess piece in the given rectangle."""
        font = painter.font()
        font.setPointSize(int(rect.height() * 0.7))
        font.setBold(False)
        painter.setFont(font)

        # Draw shadow for better visibility
        shadow_offset = 1
        painter.setPen(QtGui.QColor(50, 50, 50, 100))
        shadow_rect = rect.translated(shadow_offset, shadow_offset)
        painter.drawText(shadow_rect, QtCore.Qt.AlignmentFlag.AlignCenter, piece.symbol)

        # Draw piece
        painter.setPen(QtGui.QColor(0, 0, 0))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, piece.symbol)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press - start drag or place piece from tray."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        row, col = self._pos_to_square(event.pos())

        # If we have an incoming piece from tray, place it
        if self._incoming_piece and self._is_valid_square(row, col):
            self._state.set_piece(row, col, self._incoming_piece)
            self._incoming_piece = None
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.update()
            self.board_changed.emit(self._state.to_notation())
            return

        # Otherwise, start dragging if there's a piece
        if not self._is_valid_square(row, col):
            return

        piece = self._state.get_piece(row, col)
        if piece:
            self._dragging_piece = (piece, row, col)
            self._drag_pos = event.pos()
            self._state.set_piece(row, col, None)
            self.piece_picked_up.emit(piece, row, col)
            # Highlight all squares as valid drop targets
            self._highlighted_squares = [
                (r, c) for r in range(8) for c in range(8)
            ]
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move - update drag position."""
        if self._dragging_piece:
            self._drag_pos = event.pos()
            self.update()
        elif self._incoming_piece:
            # Update preview position
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release - drop piece."""
        if not self._dragging_piece:
            return

        piece, source_row, source_col = self._dragging_piece
        row, col = self._pos_to_square(event.pos())

        # Check if dropped on valid square
        if self._is_valid_square(row, col):
            self._state.set_piece(row, col, piece)
            self.piece_dropped.emit(row, col)
        else:
            # Dropped outside board - send to tray
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


class ChessPieceTray(PieceTrayWidget):
    """Piece tray for chess showing available pieces by color.

    Displays clickable buttons for each piece type. Clicking a piece
    selects it for placement on the board.
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
            "Click a piece, then click on the board to place it.\n"
            "Drag pieces on the board to move them.\n"
            "Drag off the board to remove."
        )
        instructions.setStyleSheet("color: #666; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # White pieces
        white_group = QtWidgets.QGroupBox("White Pieces")
        white_layout = QtWidgets.QGridLayout(white_group)
        white_layout.setSpacing(4)

        white_pieces = ["K", "Q", "R", "B", "N", "P"]
        for i, notation in enumerate(white_pieces):
            btn = self._create_piece_button(notation)
            white_layout.addWidget(btn, i // 3, i % 3)
            self._buttons[notation] = btn

        layout.addWidget(white_group)

        # Black pieces
        black_group = QtWidgets.QGroupBox("Black Pieces")
        black_layout = QtWidgets.QGridLayout(black_group)
        black_layout.setSpacing(4)

        black_pieces = ["k", "q", "r", "b", "n", "p"]
        for i, notation in enumerate(black_pieces):
            btn = self._create_piece_button(notation)
            black_layout.addWidget(btn, i // 3, i % 3)
            self._buttons[notation] = btn

        layout.addWidget(black_group)

        # Clear selection button
        clear_btn = QtWidgets.QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._clear_selection)
        layout.addWidget(clear_btn)

        layout.addStretch()

    def _create_piece_button(self, notation: str) -> QtWidgets.QPushButton:
        """Create a button for a piece type."""
        piece = CHESS_PIECES[notation]
        btn = QtWidgets.QPushButton(piece.symbol)
        btn.setFixedSize(55, 55)
        btn.setFont(QtGui.QFont("", 28))
        btn.setToolTip(f"{piece.color.title()} {piece.piece_type.title()}")
        btn.setCheckable(True)
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
            piece = CHESS_PIECES[notation]
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
        return list(CHESS_PIECES.values())

    def add_piece(self, piece: GamePiece) -> None:
        """Add piece to tray (when removed from board)."""
        notation = _piece_to_notation(piece)
        if notation:
            self._piece_counts[notation] = self._piece_counts.get(notation, 0) + 1
            _LOGGER.debug(f"Added {notation} to tray, count: {self._piece_counts[notation]}")

    def remove_piece(self, piece: GamePiece) -> None:
        """Remove piece from tray (when placed on board)."""
        notation = _piece_to_notation(piece)
        if notation and self._piece_counts.get(notation, 0) > 0:
            self._piece_counts[notation] -= 1
            _LOGGER.debug(f"Removed {notation} from tray, count: {self._piece_counts[notation]}")

    def reset(self) -> None:
        """Reset tray to have unlimited pieces available."""
        # In configuration mode, pieces are unlimited
        self._piece_counts = {n: 99 for n in CHESS_PIECES}
        self._clear_selection()


class ChessConfigDialog(BoardConfigDialog):
    """Chess-specific configuration dialog.

    Allows users to set up custom chess positions using drag-and-drop
    or by editing the FEN string directly.
    """

    def _get_title(self) -> str:
        return "Configure Chess Position"

    def _get_notation_name(self) -> str:
        return "FEN"

    def _get_notation_placeholder(self) -> str:
        return STANDARD_FEN

    def _create_board_widget(self) -> EditableBoardWidget:
        return EditableChessBoard(self)

    def _create_piece_tray(self) -> PieceTrayWidget:
        return ChessPieceTray(self)

    def _get_presets(self) -> List[Tuple[str, str]]:
        return [
            ("Standard", STANDARD_FEN),
            ("Empty", EMPTY_FEN),
            ("K+R vs K", ENDGAME_KR_K),
            ("K+Q vs K", ENDGAME_KQ_K),
        ]

    def _validate_notation(self, notation: str) -> bool:
        """Validate FEN string."""
        try:
            board = chess.Board(notation)
            # Check for exactly one king per side
            white_kings = len(board.pieces(chess.KING, chess.WHITE))
            black_kings = len(board.pieces(chess.KING, chess.BLACK))
            return white_kings == 1 and black_kings == 1
        except (ValueError, chess.InvalidBoardStateError):
            return False

    def _get_validation_error(self, notation: str) -> str:
        """Get error message for invalid FEN."""
        try:
            board = chess.Board(notation)
            white_kings = len(board.pieces(chess.KING, chess.WHITE))
            black_kings = len(board.pieces(chess.KING, chess.BLACK))

            errors = []
            if white_kings != 1:
                errors.append(f"White must have exactly 1 king (found {white_kings})")
            if black_kings != 1:
                errors.append(f"Black must have exactly 1 king (found {black_kings})")

            if errors:
                return "\n".join(errors)
            return "Unknown validation error"

        except ValueError as e:
            return f"Invalid FEN format: {e}"
        except chess.InvalidBoardStateError as e:
            return f"Invalid board state: {e}"

    def _create_state_from_notation(self, notation: str) -> BoardState:
        return ChessBoardState(notation)
