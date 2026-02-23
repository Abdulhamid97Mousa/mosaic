"""Abstract base classes for grid environment configuration dialogs.

This module provides the extensible foundation for grid-specific editors.
New environments can be added by subclassing these base classes and registering
with the GridConfigDialogFactory.

Design Patterns:
- Strategy Pattern: Environment-specific implementations share a common interface
- Template Method: Base dialog handles common UI, subclasses customize behavior

Grid environments differ from board games:
- Cells contain objects (wall, door, key, goal, lava) rather than pieces
- Agent position is a key configuration element
- Objects often have properties (color, state like open/closed)
"""

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Set

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


# Metaclass to combine Qt's metaclass with ABC
class QtABCMeta(type(QWidget), ABCMeta):
    """Combined metaclass for Qt widgets that need ABC functionality."""
    pass


class ObjectType(Enum):
    """Common grid object types across environments."""
    EMPTY = "empty"
    WALL = "wall"
    FLOOR = "floor"
    DOOR = "door"
    KEY = "key"
    BALL = "ball"
    BOX = "box"
    GOAL = "goal"
    LAVA = "lava"
    AGENT = "agent"
    # Environment-specific types can be added
    CUSTOM = "custom"


class ObjectColor(Enum):
    """Standard colors used in grid environments."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    YELLOW = "yellow"
    GREY = "grey"
    WHITE = "white"
    NONE = "none"


@dataclass
class GridObject:
    """Represents an object that can be placed in a grid cell.

    Attributes:
        obj_type: Type of object (wall, door, key, etc.)
        color: Object color (for colored objects like keys, doors)
        state: Object state (e.g., open/closed for doors)
        symbol: Unicode symbol for display
        properties: Additional environment-specific properties
    """
    obj_type: ObjectType
    color: ObjectColor = ObjectColor.NONE
    state: str = ""  # e.g., "open", "closed", "locked"
    symbol: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.obj_type, self.color, self.state))

    def __eq__(self, other):
        if not isinstance(other, GridObject):
            return False
        return (
            self.obj_type == other.obj_type
            and self.color == other.color
            and self.state == other.state
        )


@dataclass
class GridCell:
    """Represents a single cell in a grid environment.

    A cell can contain multiple objects (e.g., floor + key),
    and may have the agent present.

    Attributes:
        row: Row index (0 = top)
        col: Column index (0 = left)
        objects: List of objects in this cell
        has_agent: Whether the agent is in this cell
        agent_direction: Agent facing direction (0=right, 1=down, 2=left, 3=up)
    """
    row: int
    col: int
    objects: List[GridObject] = field(default_factory=list)
    has_agent: bool = False
    agent_direction: int = 0  # 0=right, 1=down, 2=left, 3=up

    @property
    def is_empty(self) -> bool:
        """Check if cell has no objects (or only floor)."""
        return (
            len(self.objects) == 0
            or all(o.obj_type == ObjectType.EMPTY for o in self.objects)
        )

    @property
    def is_walkable(self) -> bool:
        """Check if agent can walk into this cell."""
        blocking_types = {ObjectType.WALL, ObjectType.LAVA}
        return not any(o.obj_type in blocking_types for o in self.objects)

    def get_primary_object(self) -> Optional[GridObject]:
        """Get the primary (topmost/most important) object in cell."""
        if not self.objects:
            return None
        # Priority: agent-relevant objects first
        priority = [
            ObjectType.GOAL, ObjectType.KEY, ObjectType.BALL,
            ObjectType.BOX, ObjectType.DOOR, ObjectType.LAVA,
            ObjectType.WALL, ObjectType.FLOOR, ObjectType.EMPTY
        ]
        for obj_type in priority:
            for obj in self.objects:
                if obj.obj_type == obj_type:
                    return obj
        return self.objects[0] if self.objects else None


class GridState(ABC):
    """Abstract representation of a grid environment state.

    This class provides a common interface for manipulating grid configurations
    regardless of the specific environment. Subclasses implement environment-specific
    logic for object placement and state serialization.
    """

    @abstractmethod
    def get_cell(self, row: int, col: int) -> GridCell:
        """Get the cell at a specific position.

        Args:
            row: Row index (0 = top)
            col: Column index (0 = left)

        Returns:
            GridCell at position
        """
        pass

    @abstractmethod
    def set_cell(self, cell: GridCell) -> None:
        """Set the cell at its position.

        Args:
            cell: GridCell to place
        """
        pass

    @abstractmethod
    def get_agent_position(self) -> Optional[Tuple[int, int]]:
        """Get agent's current position.

        Returns:
            Tuple of (row, col) or None if no agent
        """
        pass

    @abstractmethod
    def set_agent_position(
        self, row: int, col: int, direction: int = 0
    ) -> None:
        """Set agent's position and direction.

        Args:
            row: Row index
            col: Column index
            direction: Facing direction (0=right, 1=down, 2=left, 3=up)
        """
        pass

    @abstractmethod
    def place_object(
        self, row: int, col: int, obj: GridObject
    ) -> bool:
        """Place an object at a position.

        Args:
            row: Row index
            col: Column index
            obj: GridObject to place

        Returns:
            True if placement succeeded
        """
        pass

    @abstractmethod
    def remove_object(
        self, row: int, col: int, obj_type: Optional[ObjectType] = None
    ) -> Optional[GridObject]:
        """Remove an object from a position.

        Args:
            row: Row index
            col: Column index
            obj_type: Specific type to remove, or None for primary object

        Returns:
            Removed GridObject, or None if nothing to remove
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert grid state to dictionary representation.

        Returns:
            Dictionary suitable for JSON serialization
        """
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load grid state from dictionary.

        Args:
            data: Dictionary from to_dict()
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """Get grid dimensions.

        Returns:
            Tuple of (rows, cols)
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all objects and agent from the grid."""
        pass

    @abstractmethod
    def copy(self) -> "GridState":
        """Create a deep copy of this grid state."""
        pass

    @abstractmethod
    def get_available_objects(self) -> List[GridObject]:
        """Get list of objects that can be placed in this environment.

        Returns:
            List of GridObject templates (without position)
        """
        pass


class GridEditorWidget(QtWidgets.QWidget, metaclass=QtABCMeta):
    """Abstract base for editable grid widgets with click-to-place.

    This widget provides the visual grid representation and handles
    mouse interactions for object manipulation. Subclasses implement
    environment-specific rendering.

    Signals:
        grid_changed: Emitted when grid state changes
        cell_clicked: Emitted when a cell is clicked (row, col)
        object_placed: Emitted when object is placed (obj, row, col)
        object_removed: Emitted when object is removed (obj, row, col)
    """

    grid_changed = pyqtSignal()
    cell_clicked = pyqtSignal(int, int)
    object_placed = pyqtSignal(object, int, int)
    object_removed = pyqtSignal(object, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state: Optional[GridState] = None
        self._selected_object: Optional[GridObject] = None
        self._highlighted_cells: Set[Tuple[int, int]] = set()
        self._hover_cell: Optional[Tuple[int, int]] = None
        self._show_grid_lines: bool = True
        self._show_coordinates: bool = False

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    @abstractmethod
    def set_state(self, state: GridState) -> None:
        """Set the grid state to display and edit.

        Args:
            state: GridState instance
        """
        pass

    @abstractmethod
    def get_state(self) -> GridState:
        """Get the current grid state.

        Returns:
            Current GridState instance
        """
        pass

    def set_selected_object(self, obj: Optional[GridObject]) -> None:
        """Set the object to place on next click.

        Args:
            obj: GridObject to place, or None to deselect
        """
        self._selected_object = obj
        if obj:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.update()

    def highlight_cells(self, cells: Set[Tuple[int, int]]) -> None:
        """Highlight specific cells.

        Args:
            cells: Set of (row, col) tuples to highlight
        """
        self._highlighted_cells = cells
        self.update()

    def clear_highlights(self) -> None:
        """Clear all cell highlights."""
        self._highlighted_cells = set()
        self.update()

    def set_show_grid_lines(self, show: bool) -> None:
        """Toggle grid line visibility."""
        self._show_grid_lines = show
        self.update()

    def set_show_coordinates(self, show: bool) -> None:
        """Toggle coordinate display."""
        self._show_coordinates = show
        self.update()

    @abstractmethod
    def _get_cell_size(self) -> int:
        """Calculate the size of each cell in pixels."""
        pass

    @abstractmethod
    def _pos_to_cell(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        """Convert widget position to grid coordinates.

        Args:
            pos: QPoint in widget coordinates

        Returns:
            Tuple of (row, col) grid coordinates
        """
        pass

    @abstractmethod
    def _cell_to_rect(self, row: int, col: int) -> QtCore.QRect:
        """Get the rectangle for a grid cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            QRect for the cell
        """
        pass

    @abstractmethod
    def _is_valid_cell(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if valid position
        """
        pass

    @abstractmethod
    def _draw_cell(
        self,
        painter: QtGui.QPainter,
        row: int,
        col: int,
        rect: QtCore.QRect
    ) -> None:
        """Draw a single cell with its contents.

        Args:
            painter: QPainter to draw with
            row: Row index
            col: Column index
            rect: Rectangle to draw in
        """
        pass

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for object placement/removal."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.pos()
            row, col = self._pos_to_cell(pos)
            if self._is_valid_cell(row, col):
                self.cell_clicked.emit(row, col)
                self._handle_cell_click(row, col, event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move for hover effects."""
        pos = event.pos()
        row, col = self._pos_to_cell(pos)
        if self._is_valid_cell(row, col):
            if self._hover_cell != (row, col):
                self._hover_cell = (row, col)
                self.update()
        else:
            if self._hover_cell is not None:
                self._hover_cell = None
                self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Clear hover state when mouse leaves widget."""
        self._hover_cell = None
        self.update()
        super().leaveEvent(event)

    def _handle_cell_click(
        self,
        row: int,
        col: int,
        event: QtGui.QMouseEvent
    ) -> None:
        """Handle a click on a grid cell.

        Default implementation: place selected object or remove existing.

        Args:
            row: Clicked row
            col: Clicked column
            event: Mouse event
        """
        if self._state is None:
            return

        if self._selected_object is not None:
            # Place selected object
            if self._state.place_object(row, col, self._selected_object):
                self.object_placed.emit(self._selected_object, row, col)
                self.grid_changed.emit()
                self.update()
        else:
            # Remove existing object (right-click or shift+click)
            if (
                event.button() == QtCore.Qt.MouseButton.RightButton
                or event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
            ):
                removed = self._state.remove_object(row, col)
                if removed:
                    self.object_removed.emit(removed, row, col)
                    self.grid_changed.emit()
                    self.update()


class GridObjectTray(QtWidgets.QWidget, metaclass=QtABCMeta):
    """Abstract base for object palette/tray widgets.

    The object tray displays available objects that can be placed
    on the grid. Selecting an object enables placement mode.

    Signals:
        object_selected: Emitted when an object is selected for placement
    """

    object_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._selected_object: Optional[GridObject] = None

    @abstractmethod
    def set_available_objects(self, objects: List[GridObject]) -> None:
        """Set the objects available in the tray.

        Args:
            objects: List of GridObject templates
        """
        pass

    @abstractmethod
    def get_selected_object(self) -> Optional[GridObject]:
        """Get the currently selected object.

        Returns:
            Selected GridObject, or None
        """
        pass

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self._selected_object = None
        self.update()


class GridConfigDialog(QtWidgets.QDialog, metaclass=QtABCMeta):
    """Abstract base dialog for grid environment configuration.

    This dialog provides the common UI structure for configuring
    custom starting states in grid environments. Subclasses implement
    environment-specific components and validation.

    The dialog layout:
    - Top: Grid editor (left) + Object tray (right)
    - Middle: Optional environment-specific controls
    - Bottom: Preset buttons + Cancel/Apply buttons
    """

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialize the configuration dialog.

        Args:
            initial_state: Optional initial state dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        self._initial_state = initial_state
        self._grid_editor: Optional[GridEditorWidget] = None
        self._object_tray: Optional[GridObjectTray] = None
        self._status_label: Optional[QtWidgets.QLabel] = None

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
        self.setMinimumSize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Main content: Grid Editor + Object Tray
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(16)

        # Grid editor (environment-specific)
        self._grid_editor = self._create_grid_editor()
        self._grid_editor.setMinimumSize(450, 450)
        content_layout.addWidget(self._grid_editor, stretch=2)

        # Right panel: Object tray + controls
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(8)

        # Object tray (environment-specific)
        tray_label = QtWidgets.QLabel("Objects")
        tray_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_panel.addWidget(tray_label)

        self._object_tray = self._create_object_tray()
        self._object_tray.setMinimumWidth(200)
        right_panel.addWidget(self._object_tray)

        # Environment-specific controls
        extra_controls = self._create_extra_controls()
        if extra_controls:
            right_panel.addWidget(extra_controls)

        right_panel.addStretch()

        content_layout.addLayout(right_panel, stretch=1)
        layout.addLayout(content_layout)

        # Status bar
        self._status_label = QtWidgets.QLabel()
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._status_label)

        # Button row
        button_layout = QtWidgets.QHBoxLayout()

        # Preset buttons
        presets_label = QtWidgets.QLabel("Presets:")
        presets_label.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(presets_label)

        for name, state_dict in self._get_presets():
            btn = QtWidgets.QPushButton(name)
            btn.setToolTip(f"Load {name} configuration")
            btn.clicked.connect(
                lambda checked, s=state_dict: self.set_state(s)
            )
            button_layout.addWidget(btn)

        button_layout.addStretch()

        # Clear button
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setToolTip("Clear all objects from grid")
        clear_btn.clicked.connect(self._on_clear)
        button_layout.addWidget(clear_btn)

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
        self._grid_editor.grid_changed.connect(self._on_grid_changed)
        self._object_tray.object_selected.connect(self._on_object_selected)

    def _on_grid_changed(self) -> None:
        """Update status when grid changes."""
        self._update_status()

    def _on_object_selected(self, obj: GridObject) -> None:
        """Handle object selection from tray."""
        self._grid_editor.set_selected_object(obj)

    def _on_clear(self) -> None:
        """Clear the grid."""
        state = self._grid_editor.get_state()
        state.clear()
        self._grid_editor.set_state(state)
        self._update_status()

    def _on_apply(self) -> None:
        """Validate and accept the dialog."""
        state_dict = self.get_state()
        errors = self._validate_state(state_dict)
        if not errors:
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Configuration",
                "\n".join(errors)
            )

    def _update_status(self) -> None:
        """Update the status label with current state info."""
        state = self._grid_editor.get_state()
        rows, cols = state.get_dimensions()
        agent_pos = state.get_agent_position()
        agent_info = f"Agent at ({agent_pos[0]}, {agent_pos[1]})" if agent_pos else "No agent placed"
        self._status_label.setText(f"Grid: {rows}x{cols} | {agent_info}")

    # Abstract methods for environment-specific customization

    @abstractmethod
    def _get_title(self) -> str:
        """Return dialog title.

        Returns:
            Title string (e.g., "Configure MiniGrid Environment")
        """
        pass

    @abstractmethod
    def _create_grid_editor(self) -> GridEditorWidget:
        """Create the environment-specific grid editor.

        Returns:
            GridEditorWidget subclass instance
        """
        pass

    @abstractmethod
    def _create_object_tray(self) -> GridObjectTray:
        """Create the environment-specific object tray.

        Returns:
            GridObjectTray subclass instance
        """
        pass

    def _create_extra_controls(self) -> Optional[QtWidgets.QWidget]:
        """Create additional environment-specific controls.

        Override to add custom controls (e.g., grid size selector).

        Returns:
            QWidget with extra controls, or None
        """
        return None

    @abstractmethod
    def _get_presets(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return list of preset configurations.

        Returns:
            List of (name, state_dict) tuples
        """
        pass

    @abstractmethod
    def _validate_state(self, state_dict: Dict[str, Any]) -> List[str]:
        """Validate the state dictionary.

        Args:
            state_dict: State to validate

        Returns:
            List of error messages (empty if valid)
        """
        pass

    @abstractmethod
    def _create_state_from_dict(self, data: Dict[str, Any]) -> GridState:
        """Create a GridState from dictionary.

        Args:
            data: State dictionary

        Returns:
            GridState instance
        """
        pass

    # Public API

    def get_state(self) -> Dict[str, Any]:
        """Get the current grid state as dictionary.

        Returns:
            State dictionary suitable for JSON serialization
        """
        return self._grid_editor.get_state().to_dict()

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """Set the grid state from dictionary.

        Args:
            state_dict: State dictionary to load
        """
        try:
            state = self._create_state_from_dict(state_dict)
            self._grid_editor.set_state(state)
            self._update_status()
        except Exception as e:
            self._status_label.setText(f"Error loading state: {e}")
            self._status_label.setStyleSheet("color: red; font-size: 11px;")
