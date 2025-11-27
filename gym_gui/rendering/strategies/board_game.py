"""Board game renderer strategy for PettingZoo Classic games.

This module provides interactive board rendering for Chess, Go, and Connect Four.
All board games are rendered through the Grid tab using a unified strategy pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Set

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import GameId, RenderMode
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy

_LOG = logging.getLogger(__name__)


# =============================================================================
# Board Game Renderer Strategy (Main Entry Point)
# =============================================================================


class BoardGameRendererStrategy(RendererStrategy):
    """Renderer strategy for interactive board games (Chess, Go, Connect Four).

    This strategy detects the game type from the payload and delegates rendering
    to the appropriate game-specific renderer. All board games are displayed in
    the Grid tab using this unified strategy.

    Signals are forwarded from the internal widget to allow MainWindow to handle
    game-specific actions (moves, clicks, etc.).
    """

    mode = RenderMode.GRID  # Rendered in Grid tab

    # Supported game IDs
    SUPPORTED_GAMES = frozenset({GameId.CHESS, GameId.CONNECT_FOUR, GameId.GO})

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._widget = _BoardGameWidget(parent)
        self._current_game: GameId | None = None

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    # Signal accessors for MainWindow connection
    @property
    def chess_move_made(self) -> QtCore.SignalInstance:
        """Signal: chess move made (from_sq: str, to_sq: str)."""
        return self._widget.chess_move_made

    @property
    def connect_four_column_clicked(self) -> QtCore.SignalInstance:
        """Signal: Connect Four column clicked (column: int)."""
        return self._widget.connect_four_column_clicked

    @property
    def go_intersection_clicked(self) -> QtCore.SignalInstance:
        """Signal: Go intersection clicked (row: int, col: int)."""
        return self._widget.go_intersection_clicked

    @property
    def go_pass_requested(self) -> QtCore.SignalInstance:
        """Signal: Go pass requested."""
        return self._widget.go_pass_requested

    def render(
        self, payload: Mapping[str, object], *, context: RendererContext | None = None
    ) -> None:
        """Render the board game payload."""
        game_id = self._detect_game(payload, context)
        if game_id is None:
            self.reset()
            return

        self._widget.render_game(game_id, dict(payload))
        self._current_game = game_id

    def supports(self, payload: Mapping[str, object]) -> bool:
        """Check if this strategy supports the payload."""
        # Check for board game specific keys
        if "chess" in payload or "fen" in payload:
            return True
        if "connect_four" in payload:
            return True
        if "go" in payload:
            return True
        # Check game_id
        game_id = payload.get("game_id")
        if isinstance(game_id, GameId) and game_id in self.SUPPORTED_GAMES:
            return True
        if isinstance(game_id, str):
            try:
                return GameId(game_id) in self.SUPPORTED_GAMES
            except ValueError:
                pass
        return False

    def reset(self) -> None:
        """Clear the board."""
        self._widget.reset()
        self._current_game = None

    def cleanup(self) -> None:
        """Clean up resources."""
        self.reset()

    def _detect_game(
        self, payload: Mapping[str, object], context: RendererContext | None
    ) -> GameId | None:
        """Detect which board game this payload represents."""
        # Try context first
        if context and context.game_id in self.SUPPORTED_GAMES:
            return context.game_id

        # Try explicit game_id in payload
        game_id = payload.get("game_id")
        if isinstance(game_id, GameId) and game_id in self.SUPPORTED_GAMES:
            return game_id
        if isinstance(game_id, str):
            try:
                gid = GameId(game_id)
                if gid in self.SUPPORTED_GAMES:
                    return gid
            except ValueError:
                pass

        # Detect from payload keys
        if "chess" in payload or "fen" in payload:
            return GameId.CHESS
        if "connect_four" in payload:
            return GameId.CONNECT_FOUR
        if "go" in payload:
            return GameId.GO

        # Detect from game_type value (adapter payloads use this)
        game_type = payload.get("game_type")
        if game_type == "chess":
            return GameId.CHESS
        if game_type == "connect_four":
            return GameId.CONNECT_FOUR
        if game_type == "go":
            return GameId.GO

        return None

    @staticmethod
    def get_game_from_payload(payload: Mapping[str, Any]) -> GameId | None:
        """Static method to detect game type from payload (for external use)."""
        if "chess" in payload or "fen" in payload:
            return GameId.CHESS
        if "connect_four" in payload:
            return GameId.CONNECT_FOUR
        if "go" in payload:
            return GameId.GO

        # Detect from game_type value (adapter payloads use this)
        game_type = payload.get("game_type")
        if game_type == "chess":
            return GameId.CHESS
        if game_type == "connect_four":
            return GameId.CONNECT_FOUR
        if game_type == "go":
            return GameId.GO

        return None


# =============================================================================
# Board Game Widget (Internal)
# =============================================================================


class _BoardGameWidget(QtWidgets.QStackedWidget):
    """Widget that manages different board game renderers.

    Uses a stacked widget to switch between Chess, Go, and Connect Four
    renderers on demand.
    """

    # Signals forwarded from child renderers
    chess_move_made = QtCore.Signal(str, str)  # from_sq, to_sq
    connect_four_column_clicked = QtCore.Signal(int)  # column
    go_intersection_clicked = QtCore.Signal(int, int)  # row, col
    go_pass_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Lazy-loaded renderers
        self._chess_renderer: _ChessBoardRenderer | None = None
        self._connect_four_renderer: _ConnectFourBoardRenderer | None = None
        self._go_renderer: _GoBoardRenderer | None = None

        # Placeholder for empty state
        self._placeholder = QtWidgets.QLabel("No board game loaded", self)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self.addWidget(self._placeholder)

        self._current_game: GameId | None = None

    def render_game(self, game_id: GameId, payload: Dict[str, Any]) -> None:
        """Render the specified board game."""
        if game_id == GameId.CHESS:
            renderer = self._get_chess_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        elif game_id == GameId.CONNECT_FOUR:
            renderer = self._get_connect_four_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        elif game_id == GameId.GO:
            renderer = self._get_go_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        else:
            self.setCurrentWidget(self._placeholder)

        self._current_game = game_id

    def reset(self) -> None:
        """Reset to placeholder state."""
        self.setCurrentWidget(self._placeholder)
        self._current_game = None

    def _get_chess_renderer(self) -> "_ChessBoardRenderer":
        """Lazy-load Chess renderer."""
        if self._chess_renderer is None:
            self._chess_renderer = _ChessBoardRenderer(self)
            self._chess_renderer.move_made.connect(self.chess_move_made)
            self.addWidget(self._chess_renderer)
        return self._chess_renderer

    def _get_connect_four_renderer(self) -> "_ConnectFourBoardRenderer":
        """Lazy-load Connect Four renderer."""
        if self._connect_four_renderer is None:
            self._connect_four_renderer = _ConnectFourBoardRenderer(self)
            self._connect_four_renderer.column_clicked.connect(
                self.connect_four_column_clicked
            )
            self.addWidget(self._connect_four_renderer)
        return self._connect_four_renderer

    def _get_go_renderer(self) -> "_GoBoardRenderer":
        """Lazy-load Go renderer."""
        if self._go_renderer is None:
            self._go_renderer = _GoBoardRenderer(self)
            self._go_renderer.intersection_clicked.connect(self.go_intersection_clicked)
            self._go_renderer.pass_requested.connect(self.go_pass_requested)
            self.addWidget(self._go_renderer)
        return self._go_renderer


# =============================================================================
# Chess Board Renderer
# =============================================================================

# Unicode chess piece symbols
_CHESS_PIECES: Dict[str, str] = {
    "K": "\u2654",  # White King
    "Q": "\u2655",  # White Queen
    "R": "\u2656",  # White Rook
    "B": "\u2657",  # White Bishop
    "N": "\u2658",  # White Knight
    "P": "\u2659",  # White Pawn
    "k": "\u265A",  # Black King
    "q": "\u265B",  # Black Queen
    "r": "\u265C",  # Black Rook
    "b": "\u265D",  # Black Bishop
    "n": "\u265E",  # Black Knight
    "p": "\u265F",  # Black Pawn
}

# Chess colors
_CHESS_LIGHT_SQUARE = QtGui.QColor(240, 217, 181)
_CHESS_DARK_SQUARE = QtGui.QColor(181, 136, 99)
_CHESS_SELECTED = QtGui.QColor(255, 255, 0, 150)
_CHESS_LEGAL_MOVE = QtGui.QColor(0, 255, 0, 100)
_CHESS_LAST_MOVE = QtGui.QColor(255, 255, 0, 80)
_CHESS_CHECK = QtGui.QColor(255, 0, 0, 100)
_CHESS_HOVER = QtGui.QColor(100, 100, 255, 50)


class _ChessBoardRenderer(QtWidgets.QWidget):
    """Chess board renderer with interactive move input."""

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
        self._last_move: Optional[tuple[str, str]] = None
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

    def _update_minimum_size(self) -> None:
        size = 8 * self._square_size + 2 * self._margin
        self.setMinimumSize(size, size)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update chess board from payload."""
        # Handle both wrapped {"chess": {...}} and flat payload from to_dict()
        chess_data = payload.get("chess", {})
        if not chess_data:
            # Flat payload - use top level (from adapter's to_dict())
            chess_data = payload

        # Get FEN
        fen = chess_data.get("fen")
        if fen:
            self._fen = fen
            self._parse_fen(fen)

        # Get legal moves
        legal_moves = chess_data.get("legal_moves", [])
        if isinstance(legal_moves, list):
            self._legal_moves = set(legal_moves)

        # Get last move
        last_move = chess_data.get("last_move")
        if isinstance(last_move, (list, tuple)) and len(last_move) == 2:
            self._last_move = (str(last_move[0]), str(last_move[1]))

        # Get check state
        self._is_check = bool(chess_data.get("is_check", False))
        self._king_square = chess_data.get("king_square")

        # Get current player
        current = chess_data.get("current_player")
        if current:
            self._current_player = current

        self._selected_square = None
        self._legal_destinations.clear()
        self.update()

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

    # Event handlers
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        square = self._pixel_to_square(event.position())
        if square is None:
            return super().mousePressEvent(event)

        if self._selected_square is None:
            if self._has_piece_at(square) and self._is_own_piece(square):
                self._selected_square = square
                self._update_legal_destinations()
                self.update()
        else:
            if square == self._selected_square:
                self._selected_square = None
                self._legal_destinations.clear()
                self.update()
            elif square in self._legal_destinations:
                from_sq = self._selected_square
                self._selected_square = None
                self._legal_destinations.clear()
                self.move_made.emit(from_sq, square)
                self.update()
            elif self._has_piece_at(square) and self._is_own_piece(square):
                self._selected_square = square
                self._update_legal_destinations()
                self.update()
            else:
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
                color = _CHESS_LIGHT_SQUARE if is_light else _CHESS_DARK_SQUARE
                painter.fillRect(x, y, self._square_size, self._square_size, color)

    def _draw_highlights(self, painter: QtGui.QPainter) -> None:
        # Last move
        if self._last_move:
            for sq in self._last_move:
                pos = self._square_to_pixel(sq)
                painter.fillRect(
                    pos.x(), pos.y(), self._square_size, self._square_size,
                    _CHESS_LAST_MOVE
                )

        # Check
        if self._is_check and self._king_square:
            pos = self._square_to_pixel(self._king_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHESS_CHECK
            )

        # Selected
        if self._selected_square:
            pos = self._square_to_pixel(self._selected_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHESS_SELECTED
            )

        # Legal moves
        for dest in self._legal_destinations:
            pos = self._square_to_pixel(dest)
            center_x = pos.x() + self._square_size // 2
            center_y = pos.y() + self._square_size // 2

            if self._has_piece_at(dest):
                painter.setPen(QtGui.QPen(_CHESS_LEGAL_MOVE, 4))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                radius = self._square_size // 2 - 4
            else:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(_CHESS_LEGAL_MOVE)
                radius = self._square_size // 6

            painter.drawEllipse(QtCore.QPoint(center_x, center_y), radius, radius)

        # Hover
        if self._hover_square and self._hover_square != self._selected_square:
            pos = self._square_to_pixel(self._hover_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHESS_HOVER
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

                if piece.isupper():
                    painter.setPen(QtGui.QColor(255, 255, 255))
                else:
                    painter.setPen(QtGui.QColor(0, 0, 0))

                rect = QtCore.QRect(
                    pos.x(), pos.y(), self._square_size, self._square_size
                )
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, symbol)

    def _draw_coordinates(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", max(10, int(self._square_size * 0.18)))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(40, 40, 40))
        font_metrics = QtGui.QFontMetrics(font)

        for i in range(8):
            # File labels (a-h)
            file_char = chr(ord("a") + i)
            char_width = font_metrics.horizontalAdvance(file_char)
            x = self._margin + i * self._square_size + (self._square_size - char_width) // 2
            painter.drawText(x, self._margin - 8, file_char)
            board_bottom = self._margin + 8 * self._square_size
            painter.drawText(x, board_bottom + font_metrics.ascent() + 5, file_char)

            # Rank labels (1-8)
            rank_char = str(8 - i)
            char_width = font_metrics.horizontalAdvance(rank_char)
            y = self._margin + i * self._square_size + (self._square_size + font_metrics.ascent()) // 2
            painter.drawText(self._margin - char_width - 8, y, rank_char)
            board_right = self._margin + 8 * self._square_size
            painter.drawText(board_right + 8, y, rank_char)


# =============================================================================
# Connect Four Board Renderer
# =============================================================================

_C4_BOARD = QtGui.QColor(0, 100, 200)
_C4_EMPTY = QtGui.QColor(240, 240, 240)
_C4_PLAYER_0 = QtGui.QColor(255, 50, 50)  # Red
_C4_PLAYER_1 = QtGui.QColor(255, 220, 50)  # Yellow
_C4_HOVER = QtGui.QColor(100, 255, 100, 150)
_C4_WIN = QtGui.QColor(255, 255, 255, 180)


class _ConnectFourBoardRenderer(QtWidgets.QWidget):
    """Connect Four board renderer with interactive column selection."""

    column_clicked = QtCore.Signal(int)

    ROWS = 6
    COLS = 7

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._board: List[List[int]] = [[0] * self.COLS for _ in range(self.ROWS)]
        self._current_player: str = "player_0"
        self._legal_columns: Set[int] = set(range(self.COLS))
        self._last_column: Optional[int] = None
        self._hover_column: Optional[int] = None
        self._is_game_over: bool = False
        self._winning_positions: List[tuple[int, int]] = []
        self._enabled: bool = True

        self._cell_size: int = 70
        self._padding: int = 10

        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        width = self.COLS * self._cell_size + 2 * self._padding
        label_height = int(self._cell_size * 0.4)
        height = (self.ROWS + 1) * self._cell_size + 2 * self._padding + label_height
        self.setMinimumSize(width, height)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update Connect Four board from payload."""
        # Handle both wrapped {"connect_four": {...}} and flat payload from to_dict()
        c4_data = payload.get("connect_four", {})
        if not c4_data:
            # Flat payload - use top level (from adapter's to_dict())
            c4_data = payload

        # Get board
        board = c4_data.get("board")
        if isinstance(board, list):
            self._board = [row[:] for row in board]

        # Get legal columns
        legal = c4_data.get("legal_columns", list(range(self.COLS)))
        if isinstance(legal, list):
            self._legal_columns = set(legal)

        # Get current player
        self._current_player = c4_data.get("current_player", "player_0")

        # Get last column
        self._last_column = c4_data.get("last_column")

        # Get game over state
        self._is_game_over = bool(c4_data.get("is_game_over", False))

        # Find winning positions if game over
        if self._is_game_over:
            self._find_winning_positions()
        else:
            self._winning_positions = []

        self.update()

    def _find_winning_positions(self) -> None:
        """Find winning piece positions for highlighting."""
        self._winning_positions = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece = self._board[row][col]
                if piece == 0:
                    continue

                # Horizontal
                if col <= 3 and all(self._board[row][col + i] == piece for i in range(4)):
                    self._winning_positions.extend([(row, col + i) for i in range(4)])

                # Vertical
                if row <= 2 and all(self._board[row + i][col] == piece for i in range(4)):
                    self._winning_positions.extend([(row + i, col) for i in range(4)])

                # Diagonal down-right
                if row <= 2 and col <= 3 and all(
                    self._board[row + i][col + i] == piece for i in range(4)
                ):
                    self._winning_positions.extend([(row + i, col + i) for i in range(4)])

                # Diagonal up-right
                if row >= 3 and col <= 3 and all(
                    self._board[row - i][col + i] == piece for i in range(4)
                ):
                    self._winning_positions.extend([(row - i, col + i) for i in range(4)])

    def _pixel_to_column(self, pos: QtCore.QPointF) -> Optional[int]:
        x = int(pos.x()) - self._padding
        col = x // self._cell_size
        if 0 <= col < self.COLS:
            return col
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or self._is_game_over:
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        col = self._pixel_to_column(event.position())
        if col is not None and col in self._legal_columns:
            self.column_clicked.emit(col)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        col = self._pixel_to_column(event.position())
        if col != self._hover_column:
            self._hover_column = col
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_column = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        width = event.size().width() - 2 * self._padding
        height = event.size().height() - 2 * self._padding
        cell_from_width = width // self.COLS
        cell_from_height = height // (self.ROWS + 1)
        self._cell_size = min(cell_from_width, cell_from_height)
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_preview_row(painter)
        self._draw_board(painter)
        self._draw_pieces(painter)
        self._draw_column_labels(painter)

        painter.end()

    def _draw_preview_row(self, painter: QtGui.QPainter) -> None:
        if self._is_game_over or not self._enabled:
            return
        if self._hover_column is not None and self._hover_column in self._legal_columns:
            col = self._hover_column
            x = self._padding + col * self._cell_size + self._cell_size // 2
            y = self._padding + self._cell_size // 2

            color = _C4_PLAYER_0 if self._current_player == "player_0" else _C4_PLAYER_1
            color = QtGui.QColor(color)
            color.setAlpha(150)

            radius = self._cell_size // 2 - 6
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)

    def _draw_board(self, painter: QtGui.QPainter) -> None:
        board_x = self._padding
        board_y = self._padding + self._cell_size
        board_width = self.COLS * self._cell_size
        board_height = self.ROWS * self._cell_size

        painter.setBrush(_C4_BOARD)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(board_x, board_y, board_width, board_height, 10, 10)

        painter.setBrush(_C4_EMPTY)
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self._board[row][col] == 0:
                    x = self._padding + col * self._cell_size + self._cell_size // 2
                    y = self._padding + (row + 1) * self._cell_size + self._cell_size // 2
                    radius = self._cell_size // 2 - 6
                    painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)

    def _draw_pieces(self, painter: QtGui.QPainter) -> None:
        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece = self._board[row][col]
                if piece == 0:
                    continue

                x = self._padding + col * self._cell_size + self._cell_size // 2
                y = self._padding + (row + 1) * self._cell_size + self._cell_size // 2
                radius = self._cell_size // 2 - 6

                color = _C4_PLAYER_0 if piece == 1 else _C4_PLAYER_1
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)

                if (row, col) in self._winning_positions:
                    painter.setBrush(_C4_WIN)
                    painter.drawEllipse(QtCore.QPoint(x, y), radius // 2, radius // 2)

        # Hover highlight
        if (
            self._hover_column is not None
            and self._hover_column in self._legal_columns
            and not self._is_game_over
        ):
            col = self._hover_column
            x = self._padding + col * self._cell_size
            y = self._padding + self._cell_size
            height = self.ROWS * self._cell_size

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(_C4_HOVER)
            painter.drawRect(x, y, self._cell_size, height)

    def _draw_column_labels(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", int(self._cell_size * 0.25))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(80, 80, 80))

        board_bottom = self._padding + (self.ROWS + 1) * self._cell_size
        label_margin = int(self._cell_size * 0.3)
        y = board_bottom + label_margin

        for col in range(self.COLS):
            x = self._padding + col * self._cell_size + self._cell_size // 2 - 5
            painter.drawText(x, y, str(col + 1))


# =============================================================================
# Go Board Renderer
# =============================================================================

_GO_BOARD = QtGui.QColor(220, 179, 92)
_GO_LINE = QtGui.QColor(0, 0, 0)
_GO_BLACK = QtGui.QColor(20, 20, 20)
_GO_WHITE = QtGui.QColor(250, 250, 250)
_GO_HOVER = QtGui.QColor(100, 200, 100, 150)
_GO_STAR = QtGui.QColor(0, 0, 0)


class _GoBoardRenderer(QtWidgets.QWidget):
    """Go board renderer with interactive stone placement."""

    intersection_clicked = QtCore.Signal(int, int)  # row, col
    pass_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._board_size: int = 19
        self._board: List[List[int]] = [[0] * 19 for _ in range(19)]
        self._current_player: str = "black_0"
        self._legal_moves: Set[int] = set()
        self._last_move: Optional[int] = None
        self._hover_pos: Optional[tuple[int, int]] = None
        self._is_game_over: bool = False
        self._enabled: bool = True

        self._cell_size: int = 30
        self._margin: int = 40
        self._stone_radius: int = int(self._cell_size * 0.45)

        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        board_width = (self._board_size - 1) * self._cell_size
        size = board_width + 2 * self._margin
        self.setMinimumSize(size, size)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update Go board from payload."""
        # Handle both wrapped {"go": {...}} and flat payload from to_dict()
        go_data = payload.get("go", {})
        if not go_data:
            # Flat payload - use top level (from adapter's to_dict())
            go_data = payload

        # Get board size
        size = go_data.get("board_size", 19)
        if size != self._board_size:
            self._board_size = size
            self._board = [[0] * size for _ in range(size)]
            self._update_minimum_size()

        # Get board state
        board = go_data.get("board")
        if isinstance(board, list):
            self._board = [row[:] for row in board]

        # Get legal moves
        legal = go_data.get("legal_moves", [])
        if isinstance(legal, list):
            self._legal_moves = set(legal)

        # Get current player
        self._current_player = go_data.get("current_player", "black_0")

        # Get last move
        self._last_move = go_data.get("last_move")

        # Get game over state
        self._is_game_over = bool(go_data.get("is_game_over", False))

        self.update()

    def action_to_coords(self, action: int) -> Optional[tuple[int, int]]:
        if action < 0 or action >= self._board_size**2:
            return None
        row = action // self._board_size
        col = action % self._board_size
        return (row, col)

    def coords_to_action(self, row: int, col: int) -> int:
        return row * self._board_size + col

    def _pixel_to_intersection(self, pos: QtCore.QPointF) -> Optional[tuple[int, int]]:
        x = int(pos.x()) - self._margin
        y = int(pos.y()) - self._margin

        col = round(x / self._cell_size)
        row = round(y / self._cell_size)

        ix = col * self._cell_size
        iy = row * self._cell_size
        dist = ((x - ix) ** 2 + (y - iy) ** 2) ** 0.5

        if (
            dist <= self._cell_size * 0.4
            and 0 <= row < self._board_size
            and 0 <= col < self._board_size
        ):
            return (row, col)
        return None

    def _intersection_to_pixel(self, row: int, col: int) -> QtCore.QPoint:
        x = self._margin + col * self._cell_size
        y = self._margin + row * self._cell_size
        return QtCore.QPoint(x, y)

    def _get_star_points(self) -> List[tuple[int, int]]:
        if self._board_size == 19:
            return [
                (3, 3), (3, 9), (3, 15),
                (9, 3), (9, 9), (9, 15),
                (15, 3), (15, 9), (15, 15),
            ]
        elif self._board_size == 13:
            return [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self._board_size == 9:
            return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        return []

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or self._is_game_over:
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        intersection = self._pixel_to_intersection(event.position())
        if intersection is not None:
            row, col = intersection
            action = self.coords_to_action(row, col)
            if action in self._legal_moves:
                self.intersection_clicked.emit(row, col)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        intersection = self._pixel_to_intersection(event.position())
        if intersection != self._hover_pos:
            self._hover_pos = intersection
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        available = min(event.size().width(), event.size().height())
        board_space = available - 2 * self._margin
        self._cell_size = max(15, board_space // (self._board_size - 1))
        self._stone_radius = int(self._cell_size * 0.45)
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_board_background(painter)
        self._draw_grid(painter)
        self._draw_star_points(painter)
        self._draw_coordinates(painter)
        self._draw_stones(painter)
        self._draw_hover(painter)
        self._draw_last_move_marker(painter)

        painter.end()

    def _draw_board_background(self, painter: QtGui.QPainter) -> None:
        board_width = (self._board_size - 1) * self._cell_size
        painter.fillRect(
            self._margin - self._cell_size // 2,
            self._margin - self._cell_size // 2,
            board_width + self._cell_size,
            board_width + self._cell_size,
            _GO_BOARD,
        )

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        painter.setPen(QtGui.QPen(_GO_LINE, 1))
        board_width = (self._board_size - 1) * self._cell_size

        for row in range(self._board_size):
            y = self._margin + row * self._cell_size
            painter.drawLine(self._margin, y, self._margin + board_width, y)

        for col in range(self._board_size):
            x = self._margin + col * self._cell_size
            painter.drawLine(x, self._margin, x, self._margin + board_width)

    def _draw_star_points(self, painter: QtGui.QPainter) -> None:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(_GO_STAR)
        star_radius = max(3, self._cell_size // 8)

        for row, col in self._get_star_points():
            pos = self._intersection_to_pixel(row, col)
            painter.drawEllipse(pos, star_radius, star_radius)

    def _draw_coordinates(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", max(9, self._cell_size // 3))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(40, 40, 40))
        font_metrics = QtGui.QFontMetrics(font)

        letters = "ABCDEFGHJKLMNOPQRST"
        for col in range(self._board_size):
            letter = letters[col]
            letter_width = font_metrics.horizontalAdvance(letter)
            x = self._margin + col * self._cell_size - letter_width // 2

            painter.drawText(x, self._margin - self._cell_size // 2 - 5, letter)
            board_bottom = self._margin + (self._board_size - 1) * self._cell_size
            painter.drawText(
                x, board_bottom + self._cell_size // 2 + font_metrics.ascent(), letter
            )

        for row in range(self._board_size):
            label = str(self._board_size - row)
            label_width = font_metrics.horizontalAdvance(label)
            y = self._margin + row * self._cell_size + font_metrics.ascent() // 2

            painter.drawText(
                self._margin - self._cell_size // 2 - label_width - 5, y, label
            )
            board_right = self._margin + (self._board_size - 1) * self._cell_size
            painter.drawText(board_right + self._cell_size // 2 + 5, y, label)

    def _draw_stones(self, painter: QtGui.QPainter) -> None:
        for row in range(self._board_size):
            for col in range(self._board_size):
                stone = self._board[row][col]
                if stone == 0:
                    continue

                pos = self._intersection_to_pixel(row, col)

                if stone == 1:  # Black
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(_GO_BLACK)
                else:  # White
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
                    painter.setBrush(_GO_WHITE)

                painter.drawEllipse(pos, self._stone_radius, self._stone_radius)

    def _draw_hover(self, painter: QtGui.QPainter) -> None:
        if self._hover_pos is None or self._is_game_over or not self._enabled:
            return

        row, col = self._hover_pos
        action = self.coords_to_action(row, col)

        if action not in self._legal_moves:
            return
        if self._board[row][col] != 0:
            return

        pos = self._intersection_to_pixel(row, col)
        color = QtGui.QColor(
            _GO_BLACK if "black" in self._current_player else _GO_WHITE
        )
        color.setAlpha(100)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(pos, self._stone_radius, self._stone_radius)

    def _draw_last_move_marker(self, painter: QtGui.QPainter) -> None:
        if self._last_move is None:
            return

        coords = self.action_to_coords(self._last_move)
        if coords is None:
            return

        row, col = coords
        pos = self._intersection_to_pixel(row, col)
        marker_radius = max(3, self._stone_radius // 3)
        stone = self._board[row][col]

        if stone == 1:
            painter.setBrush(QtGui.QColor(255, 255, 255))
        else:
            painter.setBrush(QtGui.QColor(0, 0, 0))

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(pos, marker_radius, marker_radius)


# =============================================================================
# Module exports
# =============================================================================

__all__ = ["BoardGameRendererStrategy"]
