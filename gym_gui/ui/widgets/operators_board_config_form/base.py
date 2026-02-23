"""Abstract base classes for board game configuration dialogs.

This module provides the extensible foundation for game-specific board editors.
New games can be added by subclassing these base classes and registering
with the BoardConfigDialogFactory.

Design Patterns:
- Strategy Pattern: Game-specific implementations share a common interface
- Template Method: Base dialog handles common UI, subclasses customize behavior
"""

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


# Metaclass to combine Qt's metaclass with ABC
class QtABCMeta(type(QWidget), ABCMeta):
    """Combined metaclass for Qt widgets that need ABC functionality."""
    pass


@dataclass
class GamePiece:
    """Represents a game piece with type, color, and display symbol.

    Attributes:
        piece_type: Type of piece (e.g., "king", "pawn", "stone", "checker")
        color: Piece color (e.g., "white", "black", "red")
        symbol: Unicode symbol for display (e.g., "♔", "●", "⛀")
        notation: Single character for notation (e.g., "K", "p", "B", "W")
    """

    piece_type: str
    color: str
    symbol: str
    notation: str = ""

    def __hash__(self):
        return hash((self.piece_type, self.color))

    def __eq__(self, other):
        if not isinstance(other, GamePiece):
            return False
        return self.piece_type == other.piece_type and self.color == other.color


class BoardState(ABC):
    """Abstract representation of a board game state.

    This class provides a common interface for manipulating board positions
    regardless of the specific game. Subclasses implement game-specific
    logic for piece placement and notation conversion.
    """

    @abstractmethod
    def get_piece(self, row: int, col: int) -> Optional[GamePiece]:
        """Get the piece at a specific position.

        Args:
            row: Row index (0 = top)
            col: Column index (0 = left)

        Returns:
            GamePiece at position, or None if empty
        """
        pass

    @abstractmethod
    def set_piece(self, row: int, col: int, piece: Optional[GamePiece]) -> None:
        """Set or remove a piece at a specific position.

        Args:
            row: Row index (0 = top)
            col: Column index (0 = left)
            piece: GamePiece to place, or None to clear
        """
        pass

    @abstractmethod
    def to_notation(self) -> str:
        """Convert board state to notation string.

        Returns:
            Game-specific notation (FEN for chess, SGF for Go, etc.)
        """
        pass

    @abstractmethod
    def from_notation(self, notation: str) -> None:
        """Load board state from notation string.

        Args:
            notation: Game-specific notation string
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """Get board dimensions.

        Returns:
            Tuple of (rows, cols)
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all pieces from the board."""
        pass

    @abstractmethod
    def copy(self) -> "BoardState":
        """Create a deep copy of this board state."""
        pass


class EditableBoardWidget(QtWidgets.QWidget, metaclass=QtABCMeta):
    """Abstract base for editable board widgets with drag-and-drop.

    This widget provides the visual board representation and handles
    mouse interactions for piece manipulation. Subclasses implement
    game-specific rendering and coordinate systems.

    Signals:
        board_changed: Emitted when board state changes (notation string)
        piece_picked_up: Emitted when piece is picked up (piece, row, col)
        piece_dropped: Emitted when piece is dropped on board (row, col)
        piece_removed: Emitted when piece is removed from board (piece)
    """

    board_changed = pyqtSignal(str)
    piece_picked_up = pyqtSignal(object, int, int)
    piece_dropped = pyqtSignal(int, int)
    piece_removed = pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state: Optional[BoardState] = None
        self._dragging_piece: Optional[Tuple[GamePiece, int, int]] = None
        self._drag_pos: Optional[QtCore.QPoint] = None
        self._highlighted_squares: List[Tuple[int, int]] = []
        self._incoming_piece: Optional[GamePiece] = None

        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    @abstractmethod
    def set_state(self, state: BoardState) -> None:
        """Set the board state to display and edit.

        Args:
            state: BoardState instance
        """
        pass

    @abstractmethod
    def get_state(self) -> BoardState:
        """Get the current board state.

        Returns:
            Current BoardState instance
        """
        pass

    def highlight_squares(self, squares: List[Tuple[int, int]]) -> None:
        """Highlight specific squares (e.g., valid drop targets).

        Args:
            squares: List of (row, col) tuples to highlight
        """
        self._highlighted_squares = squares
        self.update()

    def clear_highlights(self) -> None:
        """Clear all square highlights."""
        self._highlighted_squares = []
        self.update()

    def set_incoming_piece(self, piece: Optional[GamePiece]) -> None:
        """Set a piece that will be placed on next click.

        Used when selecting a piece from the tray to place on board.

        Args:
            piece: GamePiece to place, or None to cancel
        """
        self._incoming_piece = piece
        if piece:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    @abstractmethod
    def _get_square_size(self) -> int:
        """Calculate the size of each square in pixels."""
        pass

    @abstractmethod
    def _pos_to_square(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        """Convert widget position to board coordinates.

        Args:
            pos: QPoint in widget coordinates

        Returns:
            Tuple of (row, col) board coordinates
        """
        pass

    @abstractmethod
    def _square_to_rect(self, row: int, col: int) -> QtCore.QRect:
        """Get the rectangle for a board square.

        Args:
            row: Row index
            col: Column index

        Returns:
            QRect for the square
        """
        pass

    @abstractmethod
    def _is_valid_square(self, row: int, col: int) -> bool:
        """Check if coordinates are within board bounds.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if valid position
        """
        pass


class PieceTrayWidget(QtWidgets.QWidget, metaclass=QtABCMeta):
    """Abstract base for piece tray widgets.

    The piece tray displays available pieces that can be dragged onto
    the board. When pieces are removed from the board, they return
    to the tray.

    Signals:
        piece_selected: Emitted when a piece is selected for placement
    """

    piece_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

    @abstractmethod
    def get_available_pieces(self) -> List[GamePiece]:
        """Get list of pieces available in the tray.

        Returns:
            List of GamePiece instances
        """
        pass

    @abstractmethod
    def add_piece(self, piece: GamePiece) -> None:
        """Add a piece to the tray (when removed from board).

        Args:
            piece: GamePiece to add
        """
        pass

    @abstractmethod
    def remove_piece(self, piece: GamePiece) -> None:
        """Remove a piece from the tray (when placed on board).

        Args:
            piece: GamePiece to remove
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tray to initial state with all pieces available."""
        pass


class BoardConfigDialog(QtWidgets.QDialog, metaclass=QtABCMeta):
    """Abstract base dialog for board game configuration.

    This dialog provides the common UI structure for configuring
    custom starting positions in board games. Subclasses implement
    game-specific components and validation.

    The dialog layout:
    - Top: Board widget (left) + Piece tray (right)
    - Middle: Notation text field (FEN, SGF, etc.)
    - Bottom: Preset buttons + Cancel/Apply buttons
    """

    def __init__(
        self,
        initial_state: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialize the configuration dialog.

        Args:
            initial_state: Optional initial state notation string
            parent: Parent widget
        """
        super().__init__(parent)
        self._initial_state = initial_state
        self._board_widget: Optional[EditableBoardWidget] = None
        self._piece_tray: Optional[PieceTrayWidget] = None
        self._notation_field: Optional[QtWidgets.QLineEdit] = None

        self._setup_ui()

        if initial_state:
            self.set_state(initial_state)
        else:
            # Set default state
            presets = self._get_presets()
            if presets:
                self.set_state(presets[0][1])

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle(self._get_title())
        self.setMinimumSize(750, 550)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Main content: Board + Tray
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(16)

        # Board widget (game-specific)
        self._board_widget = self._create_board_widget()
        self._board_widget.setMinimumSize(400, 400)
        content_layout.addWidget(self._board_widget, stretch=2)

        # Piece tray (game-specific)
        self._piece_tray = self._create_piece_tray()
        self._piece_tray.setMinimumWidth(200)
        content_layout.addWidget(self._piece_tray, stretch=1)

        layout.addLayout(content_layout)

        # Notation field
        notation_layout = QtWidgets.QHBoxLayout()
        notation_label = QtWidgets.QLabel(f"{self._get_notation_name()}:")
        notation_label.setStyleSheet("font-weight: bold;")
        self._notation_field = QtWidgets.QLineEdit()
        self._notation_field.setPlaceholderText(self._get_notation_placeholder())
        self._notation_field.setFont(QtGui.QFont("Monospace", 10))
        self._notation_field.editingFinished.connect(self._on_notation_edited)
        notation_layout.addWidget(notation_label)
        notation_layout.addWidget(self._notation_field, stretch=1)
        layout.addLayout(notation_layout)

        # Preset and dialog buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Preset buttons
        for name, notation in self._get_presets():
            btn = QtWidgets.QPushButton(name)
            btn.setToolTip(f"Load {name} position")
            btn.clicked.connect(lambda checked, n=notation: self.set_state(n))
            button_layout.addWidget(btn)

        button_layout.addStretch()

        # Dialog buttons
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)

        self._apply_btn = QtWidgets.QPushButton("Apply")
        self._apply_btn.setDefault(True)
        self._apply_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(self._apply_btn)

        layout.addLayout(button_layout)

        # Connect signals
        self._board_widget.board_changed.connect(self._on_board_changed)
        self._board_widget.piece_removed.connect(self._on_piece_removed)
        self._piece_tray.piece_selected.connect(self._on_tray_piece_selected)

    def _on_board_changed(self, notation: str) -> None:
        """Update notation field when board changes."""
        self._notation_field.setText(notation)

    def _on_notation_edited(self) -> None:
        """Update board when notation is manually edited."""
        notation = self._notation_field.text().strip()
        if notation and self._validate_notation(notation):
            self.set_state(notation)
        elif notation:
            # Show validation error
            self._notation_field.setStyleSheet("border: 2px solid red;")
            QtCore.QTimer.singleShot(
                2000,
                lambda: self._notation_field.setStyleSheet("")
            )

    def _on_apply(self) -> None:
        """Validate and accept the dialog."""
        notation = self.get_state()
        if self._validate_notation(notation):
            self.accept()
        else:
            error_msg = self._get_validation_error(notation)
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Position",
                error_msg
            )

    def _on_piece_removed(self, piece: GamePiece) -> None:
        """Handle piece removed from board (sent to tray)."""
        self._piece_tray.add_piece(piece)

    def _on_tray_piece_selected(self, piece: GamePiece) -> None:
        """Handle piece selected from tray for placement."""
        self._board_widget.set_incoming_piece(piece)

    # Abstract methods for game-specific customization

    @abstractmethod
    def _get_title(self) -> str:
        """Return dialog title.

        Returns:
            Title string (e.g., "Configure Chess Position")
        """
        pass

    @abstractmethod
    def _get_notation_name(self) -> str:
        """Return notation type name.

        Returns:
            Notation name (e.g., "FEN", "SGF")
        """
        pass

    @abstractmethod
    def _get_notation_placeholder(self) -> str:
        """Return placeholder text for notation field.

        Returns:
            Example notation string
        """
        pass

    @abstractmethod
    def _create_board_widget(self) -> EditableBoardWidget:
        """Create the game-specific board widget.

        Returns:
            EditableBoardWidget subclass instance
        """
        pass

    @abstractmethod
    def _create_piece_tray(self) -> PieceTrayWidget:
        """Create the game-specific piece tray.

        Returns:
            PieceTrayWidget subclass instance
        """
        pass

    @abstractmethod
    def _get_presets(self) -> List[Tuple[str, str]]:
        """Return list of preset positions.

        Returns:
            List of (name, notation) tuples
        """
        pass

    @abstractmethod
    def _validate_notation(self, notation: str) -> bool:
        """Validate the notation string.

        Args:
            notation: Notation string to validate

        Returns:
            True if valid
        """
        pass

    @abstractmethod
    def _get_validation_error(self, notation: str) -> str:
        """Return error message for invalid notation.

        Args:
            notation: Invalid notation string

        Returns:
            Human-readable error message
        """
        pass

    @abstractmethod
    def _create_state_from_notation(self, notation: str) -> BoardState:
        """Create a BoardState from notation string.

        Args:
            notation: Notation string

        Returns:
            BoardState instance
        """
        pass

    # Public API

    def get_state(self) -> str:
        """Get the current board state as notation string.

        Returns:
            Notation string (FEN, SGF, etc.)
        """
        return self._board_widget.get_state().to_notation()

    def set_state(self, notation: str) -> None:
        """Set the board state from notation string.

        Args:
            notation: Notation string to load
        """
        try:
            state = self._create_state_from_notation(notation)
            self._board_widget.set_state(state)
            self._notation_field.setText(notation)
            self._notation_field.setStyleSheet("")
        except Exception as e:
            # Invalid notation - show error styling
            self._notation_field.setStyleSheet("border: 2px solid red;")
