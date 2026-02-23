"""MiniGrid environment configuration dialog.

This module provides the configuration UI for MiniGrid and BabyAI environments,
allowing users to design custom grid layouts with walls, doors, keys, goals, etc.

Supported environments:
- MiniGrid-Empty-*
- MiniGrid-FourRooms-*
- MiniGrid-DoorKey-*
- MiniGrid-MultiRoom-*
- BabyAI-*
- And other MiniGrid variants
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal

from .base import (
    GridState,
    GridCell,
    GridObject,
    GridEditorWidget,
    GridObjectTray,
    GridConfigDialog,
    ObjectType,
    ObjectColor,
)

_LOGGER = logging.getLogger(__name__)


# MiniGrid object definitions with display properties
MINIGRID_OBJECTS: Dict[str, GridObject] = {
    "empty": GridObject(ObjectType.EMPTY, ObjectColor.NONE, "", " "),
    "wall": GridObject(ObjectType.WALL, ObjectColor.GREY, "", "#"),
    "floor": GridObject(ObjectType.FLOOR, ObjectColor.NONE, "", "."),
    "goal": GridObject(ObjectType.GOAL, ObjectColor.GREEN, "", "G"),
    "lava": GridObject(ObjectType.LAVA, ObjectColor.RED, "", "~"),
    # Colored keys
    "key_red": GridObject(ObjectType.KEY, ObjectColor.RED, "", "k"),
    "key_green": GridObject(ObjectType.KEY, ObjectColor.GREEN, "", "k"),
    "key_blue": GridObject(ObjectType.KEY, ObjectColor.BLUE, "", "k"),
    "key_yellow": GridObject(ObjectType.KEY, ObjectColor.YELLOW, "", "k"),
    # Colored doors
    "door_red": GridObject(ObjectType.DOOR, ObjectColor.RED, "closed", "D"),
    "door_green": GridObject(ObjectType.DOOR, ObjectColor.GREEN, "closed", "D"),
    "door_blue": GridObject(ObjectType.DOOR, ObjectColor.BLUE, "closed", "D"),
    "door_yellow": GridObject(ObjectType.DOOR, ObjectColor.YELLOW, "closed", "D"),
    # Colored balls
    "ball_red": GridObject(ObjectType.BALL, ObjectColor.RED, "", "o"),
    "ball_green": GridObject(ObjectType.BALL, ObjectColor.GREEN, "", "o"),
    "ball_blue": GridObject(ObjectType.BALL, ObjectColor.BLUE, "", "o"),
    # Colored boxes
    "box_red": GridObject(ObjectType.BOX, ObjectColor.RED, "", "B"),
    "box_green": GridObject(ObjectType.BOX, ObjectColor.GREEN, "", "B"),
    "box_blue": GridObject(ObjectType.BOX, ObjectColor.BLUE, "", "B"),
}

# Color mappings for rendering
COLOR_MAP: Dict[ObjectColor, str] = {
    ObjectColor.RED: "#FF6B6B",
    ObjectColor.GREEN: "#51CF66",
    ObjectColor.BLUE: "#74C0FC",
    ObjectColor.PURPLE: "#CC5DE8",
    ObjectColor.YELLOW: "#FFE066",
    ObjectColor.GREY: "#868E96",
    ObjectColor.WHITE: "#F8F9FA",
    ObjectColor.NONE: "#DEE2E6",
}

# Direction arrows for agent display
DIRECTION_ARROWS = ["→", "↓", "←", "↑"]


class MiniGridState(GridState):
    """Concrete GridState implementation for MiniGrid environments."""

    def __init__(self, rows: int = 8, cols: int = 8):
        """Initialize empty grid state.

        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self._rows = rows
        self._cols = cols
        self._grid: List[List[GridCell]] = []
        self._agent_pos: Optional[Tuple[int, int]] = None
        self._agent_dir: int = 0

        # Initialize empty grid
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(GridCell(r, c, []))
            self._grid.append(row)

    def get_cell(self, row: int, col: int) -> GridCell:
        """Get the cell at a specific position."""
        if not self._is_valid(row, col):
            raise IndexError(f"Cell ({row}, {col}) out of bounds")
        return self._grid[row][col]

    def set_cell(self, cell: GridCell) -> None:
        """Set the cell at its position."""
        if not self._is_valid(cell.row, cell.col):
            raise IndexError(f"Cell ({cell.row}, {cell.col}) out of bounds")
        self._grid[cell.row][cell.col] = cell

    def get_agent_position(self) -> Optional[Tuple[int, int]]:
        """Get agent's current position."""
        return self._agent_pos

    def set_agent_position(self, row: int, col: int, direction: int = 0) -> None:
        """Set agent's position and direction."""
        if not self._is_valid(row, col):
            raise IndexError(f"Position ({row}, {col}) out of bounds")

        # Clear old agent position
        if self._agent_pos:
            old_row, old_col = self._agent_pos
            self._grid[old_row][old_col].has_agent = False

        # Set new position
        self._agent_pos = (row, col)
        self._agent_dir = direction % 4
        self._grid[row][col].has_agent = True
        self._grid[row][col].agent_direction = direction % 4

    def place_object(self, row: int, col: int, obj: GridObject) -> bool:
        """Place an object at a position."""
        if not self._is_valid(row, col):
            return False

        cell = self._grid[row][col]

        # Handle agent placement specially
        if obj.obj_type == ObjectType.AGENT:
            self.set_agent_position(row, col, 0)
            return True

        # Don't place on walls or if same object already exists
        for existing in cell.objects:
            if existing.obj_type == ObjectType.WALL:
                return False
            if existing == obj:
                return False

        # Clear empty placeholder before placing
        cell.objects = [o for o in cell.objects if o.obj_type != ObjectType.EMPTY]

        # Walls replace everything
        if obj.obj_type == ObjectType.WALL:
            cell.objects = [obj]
        else:
            cell.objects.append(obj)

        return True

    def remove_object(
        self, row: int, col: int, obj_type: Optional[ObjectType] = None
    ) -> Optional[GridObject]:
        """Remove an object from a position."""
        if not self._is_valid(row, col):
            return None

        cell = self._grid[row][col]

        # Handle agent removal
        if obj_type == ObjectType.AGENT or (obj_type is None and cell.has_agent):
            if cell.has_agent:
                cell.has_agent = False
                self._agent_pos = None
                return GridObject(ObjectType.AGENT, ObjectColor.NONE, "", "A")
            return None

        if not cell.objects:
            return None

        if obj_type:
            # Remove specific type
            for i, obj in enumerate(cell.objects):
                if obj.obj_type == obj_type:
                    return cell.objects.pop(i)
            return None
        else:
            # Remove topmost non-floor object
            for i in range(len(cell.objects) - 1, -1, -1):
                if cell.objects[i].obj_type not in (ObjectType.EMPTY, ObjectType.FLOOR):
                    return cell.objects.pop(i)
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert grid state to dictionary representation."""
        cells_data = []
        for row in self._grid:
            for cell in row:
                if cell.objects or cell.has_agent:
                    cell_data = {
                        "row": cell.row,
                        "col": cell.col,
                        "objects": [
                            {
                                "type": obj.obj_type.value,
                                "color": obj.color.value,
                                "state": obj.state,
                            }
                            for obj in cell.objects
                        ],
                    }
                    if cell.has_agent:
                        cell_data["has_agent"] = True
                        cell_data["agent_direction"] = cell.agent_direction
                    cells_data.append(cell_data)

        return {
            "rows": self._rows,
            "cols": self._cols,
            "cells": cells_data,
            "agent_pos": list(self._agent_pos) if self._agent_pos else None,
            "agent_dir": self._agent_dir,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load grid state from dictionary."""
        self._rows = data.get("rows", 8)
        self._cols = data.get("cols", 8)
        self._agent_dir = data.get("agent_dir", 0)

        # Reset grid
        self._grid = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                row.append(GridCell(r, c, []))
            self._grid.append(row)

        # Load cells
        for cell_data in data.get("cells", []):
            row = cell_data["row"]
            col = cell_data["col"]
            if not self._is_valid(row, col):
                continue

            cell = self._grid[row][col]
            cell.objects = []

            for obj_data in cell_data.get("objects", []):
                try:
                    obj_type = ObjectType(obj_data["type"])
                    color = ObjectColor(obj_data.get("color", "none"))
                    state = obj_data.get("state", "")
                    cell.objects.append(GridObject(obj_type, color, state, ""))
                except (ValueError, KeyError):
                    _LOGGER.warning(f"Unknown object data: {obj_data}")

            if cell_data.get("has_agent"):
                cell.has_agent = True
                cell.agent_direction = cell_data.get("agent_direction", 0)

        # Set agent position
        agent_pos = data.get("agent_pos")
        if agent_pos:
            self._agent_pos = tuple(agent_pos)
        else:
            self._agent_pos = None

    def get_dimensions(self) -> Tuple[int, int]:
        """Get grid dimensions."""
        return (self._rows, self._cols)

    def clear(self) -> None:
        """Clear all objects and agent from the grid."""
        for row in self._grid:
            for cell in row:
                cell.objects = []
                cell.has_agent = False
        self._agent_pos = None
        self._agent_dir = 0

    def copy(self) -> "MiniGridState":
        """Create a deep copy of this grid state."""
        new_state = MiniGridState(self._rows, self._cols)
        new_state.from_dict(self.to_dict())
        return new_state

    def get_available_objects(self) -> List[GridObject]:
        """Get list of objects that can be placed in MiniGrid."""
        return list(MINIGRID_OBJECTS.values())

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are valid."""
        return 0 <= row < self._rows and 0 <= col < self._cols

    def resize(self, new_rows: int, new_cols: int) -> None:
        """Resize the grid, preserving existing content where possible."""
        new_grid: List[List[GridCell]] = []

        for r in range(new_rows):
            row = []
            for c in range(new_cols):
                if r < self._rows and c < self._cols:
                    # Copy existing cell
                    row.append(copy.deepcopy(self._grid[r][c]))
                else:
                    # New empty cell
                    row.append(GridCell(r, c, []))
            new_grid.append(row)

        self._rows = new_rows
        self._cols = new_cols
        self._grid = new_grid

        # Clear agent if out of bounds
        if self._agent_pos:
            ar, ac = self._agent_pos
            if ar >= new_rows or ac >= new_cols:
                self._agent_pos = None

    def add_border_walls(self) -> None:
        """Add walls around the perimeter of the grid."""
        wall = MINIGRID_OBJECTS["wall"]

        for c in range(self._cols):
            # Top row
            self._grid[0][c].objects = [copy.deepcopy(wall)]
            # Bottom row
            self._grid[self._rows - 1][c].objects = [copy.deepcopy(wall)]

        for r in range(self._rows):
            # Left column
            self._grid[r][0].objects = [copy.deepcopy(wall)]
            # Right column
            self._grid[r][self._cols - 1].objects = [copy.deepcopy(wall)]


class MiniGridEditor(GridEditorWidget):
    """Grid editor widget for MiniGrid environments."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state: Optional[MiniGridState] = None
        self._cell_size: int = 40
        self._margin: int = 20
        self.setMinimumSize(300, 300)

    def set_state(self, state: GridState) -> None:
        """Set the grid state to display and edit."""
        if not isinstance(state, MiniGridState):
            raise TypeError("Expected MiniGridState")
        self._state = state
        self.update()

    def get_state(self) -> GridState:
        """Get the current grid state."""
        if self._state is None:
            self._state = MiniGridState()
        return self._state

    def _get_cell_size(self) -> int:
        """Calculate the size of each cell in pixels."""
        if self._state is None:
            return 40

        rows, cols = self._state.get_dimensions()
        available_w = self.width() - 2 * self._margin
        available_h = self.height() - 2 * self._margin

        size_by_width = available_w // cols if cols > 0 else 40
        size_by_height = available_h // rows if rows > 0 else 40

        return max(20, min(size_by_width, size_by_height, 60))

    def _pos_to_cell(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        """Convert widget position to grid coordinates."""
        cell_size = self._get_cell_size()
        col = (pos.x() - self._margin) // cell_size
        row = (pos.y() - self._margin) // cell_size
        return (row, col)

    def _cell_to_rect(self, row: int, col: int) -> QtCore.QRect:
        """Get the rectangle for a grid cell."""
        cell_size = self._get_cell_size()
        x = self._margin + col * cell_size
        y = self._margin + row * cell_size
        return QtCore.QRect(x, y, cell_size, cell_size)

    def _is_valid_cell(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds."""
        if self._state is None:
            return False
        rows, cols = self._state.get_dimensions()
        return 0 <= row < rows and 0 <= col < cols

    def _draw_cell(
        self,
        painter: QtGui.QPainter,
        row: int,
        col: int,
        rect: QtCore.QRect
    ) -> None:
        """Draw a single cell with its contents."""
        if self._state is None:
            return

        cell = self._state.get_cell(row, col)

        # Background color
        bg_color = QtGui.QColor("#F8F9FA")  # Default floor

        # Get primary object for coloring
        primary = cell.get_primary_object()
        if primary:
            if primary.obj_type == ObjectType.WALL:
                bg_color = QtGui.QColor("#495057")
            elif primary.obj_type == ObjectType.LAVA:
                bg_color = QtGui.QColor("#FF6B6B")
            elif primary.obj_type == ObjectType.GOAL:
                bg_color = QtGui.QColor("#51CF66")

        # Draw cell background
        painter.fillRect(rect, bg_color)

        # Draw grid lines
        if self._show_grid_lines:
            painter.setPen(QtGui.QPen(QtGui.QColor("#CED4DA"), 1))
            painter.drawRect(rect)

        # Highlight hover/selected cells
        if self._hover_cell == (row, col):
            highlight = QtGui.QColor(100, 149, 237, 80)  # Cornflower blue
            painter.fillRect(rect, highlight)

        if (row, col) in self._highlighted_cells:
            highlight = QtGui.QColor(255, 215, 0, 100)  # Gold
            painter.fillRect(rect, highlight)

        # Draw objects
        self._draw_objects(painter, cell, rect)

        # Draw agent
        if cell.has_agent:
            self._draw_agent(painter, cell.agent_direction, rect)

    def _draw_objects(
        self,
        painter: QtGui.QPainter,
        cell: GridCell,
        rect: QtCore.QRect
    ) -> None:
        """Draw objects in a cell."""
        for obj in cell.objects:
            if obj.obj_type in (ObjectType.EMPTY, ObjectType.FLOOR, ObjectType.WALL, ObjectType.LAVA, ObjectType.GOAL):
                continue  # Already handled by background

            color = QtGui.QColor(COLOR_MAP.get(obj.color, "#DEE2E6"))
            margin = rect.width() // 6

            if obj.obj_type == ObjectType.KEY:
                # Draw key as small circle with line
                painter.setPen(QtGui.QPen(color, 2))
                painter.setBrush(color)
                cx = rect.center().x()
                cy = rect.center().y()
                painter.drawEllipse(cx - 6, cy - 6, 12, 12)
                painter.drawLine(cx, cy + 6, cx, cy + 12)

            elif obj.obj_type == ObjectType.DOOR:
                # Draw door as rectangle with gap
                painter.setPen(QtGui.QPen(color.darker(120), 2))
                painter.setBrush(color if obj.state == "closed" else QtCore.Qt.GlobalColor.transparent)
                inner = rect.adjusted(margin, margin, -margin, -margin)
                painter.drawRect(inner)
                if obj.state == "open":
                    # Draw open indicator
                    painter.drawLine(inner.topLeft(), inner.bottomRight())

            elif obj.obj_type == ObjectType.BALL:
                # Draw ball as circle
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(color)
                inner = rect.adjusted(margin, margin, -margin, -margin)
                painter.drawEllipse(inner)

            elif obj.obj_type == ObjectType.BOX:
                # Draw box as square
                painter.setPen(QtGui.QPen(color.darker(120), 2))
                painter.setBrush(color)
                inner = rect.adjusted(margin, margin, -margin, -margin)
                painter.drawRect(inner)

    def _draw_agent(
        self,
        painter: QtGui.QPainter,
        direction: int,
        rect: QtCore.QRect
    ) -> None:
        """Draw the agent with direction indicator."""
        # Agent body (red triangle pointing in direction)
        painter.setPen(QtGui.QPen(QtGui.QColor("#E03131"), 2))
        painter.setBrush(QtGui.QColor("#FF6B6B"))

        cx = rect.center().x()
        cy = rect.center().y()
        size = rect.width() // 3

        # Triangle points based on direction
        if direction == 0:  # Right
            points = [
                QtCore.QPoint(cx + size, cy),
                QtCore.QPoint(cx - size // 2, cy - size),
                QtCore.QPoint(cx - size // 2, cy + size),
            ]
        elif direction == 1:  # Down
            points = [
                QtCore.QPoint(cx, cy + size),
                QtCore.QPoint(cx - size, cy - size // 2),
                QtCore.QPoint(cx + size, cy - size // 2),
            ]
        elif direction == 2:  # Left
            points = [
                QtCore.QPoint(cx - size, cy),
                QtCore.QPoint(cx + size // 2, cy - size),
                QtCore.QPoint(cx + size // 2, cy + size),
            ]
        else:  # Up
            points = [
                QtCore.QPoint(cx, cy - size),
                QtCore.QPoint(cx - size, cy + size // 2),
                QtCore.QPoint(cx + size, cy + size // 2),
            ]

        painter.drawPolygon(QtGui.QPolygon(points))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the grid."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QtGui.QColor("#E9ECEF"))

        if self._state is None:
            return

        rows, cols = self._state.get_dimensions()

        # Draw all cells
        for row in range(rows):
            for col in range(cols):
                rect = self._cell_to_rect(row, col)
                self._draw_cell(painter, row, col, rect)

        # Draw coordinates if enabled
        if self._show_coordinates:
            painter.setPen(QtGui.QColor("#868E96"))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            cell_size = self._get_cell_size()
            for c in range(cols):
                x = self._margin + c * cell_size + cell_size // 2
                painter.drawText(x - 5, self._margin - 5, str(c))

            for r in range(rows):
                y = self._margin + r * cell_size + cell_size // 2
                painter.drawText(5, y + 3, str(r))

    def _handle_cell_click(
        self,
        row: int,
        col: int,
        event: QtGui.QMouseEvent
    ) -> None:
        """Handle a click on a grid cell."""
        if self._state is None:
            return

        # Right-click or Shift+click removes
        if (
            event.button() == QtCore.Qt.MouseButton.RightButton
            or event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            cell = self._state.get_cell(row, col)
            if cell.has_agent:
                self._state.remove_object(row, col, ObjectType.AGENT)
            else:
                removed = self._state.remove_object(row, col)
                if removed:
                    self.object_removed.emit(removed, row, col)
            self.grid_changed.emit()
            self.update()
            return

        # Left-click places selected object
        if self._selected_object is not None:
            if self._selected_object.obj_type == ObjectType.AGENT:
                # Special handling for agent - place with current direction
                cell = self._state.get_cell(row, col)
                # Cycle direction if clicking same cell with agent
                if cell.has_agent:
                    new_dir = (cell.agent_direction + 1) % 4
                    self._state.set_agent_position(row, col, new_dir)
                else:
                    self._state.set_agent_position(row, col, 0)
            else:
                obj_copy = copy.deepcopy(self._selected_object)
                if self._state.place_object(row, col, obj_copy):
                    self.object_placed.emit(obj_copy, row, col)

            self.grid_changed.emit()
            self.update()


class MiniGridObjectTray(GridObjectTray):
    """Object palette for MiniGrid environments."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._objects: List[GridObject] = []
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the tray UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # Group objects by category
        categories = {
            "Agent": [
                ("Agent", GridObject(ObjectType.AGENT, ObjectColor.RED, "", "A"))
            ],
            "Terrain": [
                ("Wall", MINIGRID_OBJECTS["wall"]),
                ("Goal", MINIGRID_OBJECTS["goal"]),
                ("Lava", MINIGRID_OBJECTS["lava"]),
            ],
            "Keys": [
                ("Red Key", MINIGRID_OBJECTS["key_red"]),
                ("Green Key", MINIGRID_OBJECTS["key_green"]),
                ("Blue Key", MINIGRID_OBJECTS["key_blue"]),
                ("Yellow Key", MINIGRID_OBJECTS["key_yellow"]),
            ],
            "Doors": [
                ("Red Door", MINIGRID_OBJECTS["door_red"]),
                ("Green Door", MINIGRID_OBJECTS["door_green"]),
                ("Blue Door", MINIGRID_OBJECTS["door_blue"]),
                ("Yellow Door", MINIGRID_OBJECTS["door_yellow"]),
            ],
            "Items": [
                ("Red Ball", MINIGRID_OBJECTS["ball_red"]),
                ("Green Ball", MINIGRID_OBJECTS["ball_green"]),
                ("Blue Ball", MINIGRID_OBJECTS["ball_blue"]),
                ("Red Box", MINIGRID_OBJECTS["box_red"]),
                ("Green Box", MINIGRID_OBJECTS["box_green"]),
                ("Blue Box", MINIGRID_OBJECTS["box_blue"]),
            ],
        }

        for category, items in categories.items():
            # Category label
            label = QtWidgets.QLabel(category)
            label.setStyleSheet("font-weight: bold; color: #495057;")
            layout.addWidget(label)

            # Create button grid for items
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(2)

            for i, (name, obj) in enumerate(items):
                btn = QtWidgets.QPushButton(name)
                btn.setCheckable(True)
                btn.setFixedHeight(28)

                # Color coding
                color = COLOR_MAP.get(obj.color, "#DEE2E6")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color};
                        border: 1px solid #ADB5BD;
                        border-radius: 3px;
                        font-size: 11px;
                    }}
                    QPushButton:checked {{
                        border: 2px solid #228BE6;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        border: 1px solid #495057;
                    }}
                """)

                btn.clicked.connect(
                    lambda checked, o=obj, b=btn: self._on_button_clicked(o, b)
                )
                self._buttons[name] = btn
                grid.addWidget(btn, i // 2, i % 2)

            layout.addLayout(grid)

        # Eraser/clear tool
        layout.addSpacing(10)
        eraser_btn = QtWidgets.QPushButton("Eraser (Clear)")
        eraser_btn.setCheckable(True)
        eraser_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #ADB5BD;
                border-radius: 3px;
            }
            QPushButton:checked {
                border: 2px solid #E03131;
                font-weight: bold;
            }
        """)
        eraser_btn.clicked.connect(lambda: self._clear_selection())
        self._buttons["eraser"] = eraser_btn
        layout.addWidget(eraser_btn)

        layout.addStretch()

    def _on_button_clicked(self, obj: GridObject, btn: QtWidgets.QPushButton) -> None:
        """Handle object button click."""
        # Uncheck other buttons
        for name, other_btn in self._buttons.items():
            if other_btn != btn:
                other_btn.setChecked(False)

        if btn.isChecked():
            self._selected_object = obj
            self.object_selected.emit(obj)
        else:
            self._selected_object = None
            self.object_selected.emit(None)

    def _clear_selection(self) -> None:
        """Clear selection (eraser mode)."""
        for btn in self._buttons.values():
            btn.setChecked(False)
        self._selected_object = None
        self.object_selected.emit(None)

    def set_available_objects(self, objects: List[GridObject]) -> None:
        """Set the objects available in the tray."""
        self._objects = objects

    def get_selected_object(self) -> Optional[GridObject]:
        """Get the currently selected object."""
        return self._selected_object


class MiniGridConfigDialog(GridConfigDialog):
    """Configuration dialog for MiniGrid environments."""

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        env_id: str = "MiniGrid-Empty-8x8-v0"
    ):
        """Initialize the dialog.

        Args:
            initial_state: Optional initial state dictionary
            parent: Parent widget
            env_id: Environment ID (used to determine default size)
        """
        self._env_id = env_id
        self._size_spinbox: Optional[QtWidgets.QSpinBox] = None
        super().__init__(initial_state, parent)

    def _get_title(self) -> str:
        """Return dialog title."""
        return f"Configure Grid - {self._env_id}"

    def _create_grid_editor(self) -> GridEditorWidget:
        """Create the MiniGrid editor widget."""
        return MiniGridEditor()

    def _create_object_tray(self) -> GridObjectTray:
        """Create the MiniGrid object tray."""
        return MiniGridObjectTray()

    def _create_extra_controls(self) -> Optional[QtWidgets.QWidget]:
        """Create grid size controls."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 10, 0, 0)

        # Grid size control
        size_group = QtWidgets.QGroupBox("Grid Size")
        size_layout = QtWidgets.QHBoxLayout(size_group)

        self._size_spinbox = QtWidgets.QSpinBox()
        self._size_spinbox.setRange(4, 20)
        self._size_spinbox.setValue(8)
        self._size_spinbox.valueChanged.connect(self._on_size_changed)

        size_layout.addWidget(QtWidgets.QLabel("Size:"))
        size_layout.addWidget(self._size_spinbox)
        size_layout.addWidget(QtWidgets.QLabel("x"))
        size_layout.addWidget(QtWidgets.QLabel("(square)"))
        size_layout.addStretch()

        layout.addWidget(size_group)

        # Quick actions
        actions_group = QtWidgets.QGroupBox("Quick Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)

        add_walls_btn = QtWidgets.QPushButton("Add Border Walls")
        add_walls_btn.clicked.connect(self._on_add_border_walls)
        actions_layout.addWidget(add_walls_btn)

        layout.addWidget(actions_group)

        return widget

    def _on_size_changed(self, size: int) -> None:
        """Handle grid size change."""
        state = self._grid_editor.get_state()
        if isinstance(state, MiniGridState):
            state.resize(size, size)
            self._grid_editor.set_state(state)
            self._update_status()

    def _on_add_border_walls(self) -> None:
        """Add border walls to the grid."""
        state = self._grid_editor.get_state()
        if isinstance(state, MiniGridState):
            state.add_border_walls()
            self._grid_editor.set_state(state)
            self._grid_editor.grid_changed.emit()

    def _get_presets(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return list of preset configurations."""
        return [
            ("Empty 8x8", self._create_empty_preset(8)),
            ("Empty 5x5", self._create_empty_preset(5)),
            ("Four Rooms", self._create_four_rooms_preset()),
            ("Corridor", self._create_corridor_preset()),
        ]

    def _create_empty_preset(self, size: int) -> Dict[str, Any]:
        """Create an empty grid preset with border walls."""
        state = MiniGridState(size, size)
        state.add_border_walls()
        state.set_agent_position(1, 1, 0)
        return state.to_dict()

    def _create_four_rooms_preset(self) -> Dict[str, Any]:
        """Create a four rooms preset."""
        state = MiniGridState(9, 9)
        state.add_border_walls()

        # Add cross walls
        wall = MINIGRID_OBJECTS["wall"]
        for i in range(9):
            if i != 2 and i != 6:
                state.place_object(4, i, copy.deepcopy(wall))
                state.place_object(i, 4, copy.deepcopy(wall))

        # Add doors at openings
        state.place_object(4, 2, copy.deepcopy(MINIGRID_OBJECTS["door_yellow"]))
        state.place_object(4, 6, copy.deepcopy(MINIGRID_OBJECTS["door_blue"]))
        state.place_object(2, 4, copy.deepcopy(MINIGRID_OBJECTS["door_green"]))
        state.place_object(6, 4, copy.deepcopy(MINIGRID_OBJECTS["door_red"]))

        state.set_agent_position(1, 1, 0)
        state.place_object(7, 7, copy.deepcopy(MINIGRID_OBJECTS["goal"]))

        return state.to_dict()

    def _create_corridor_preset(self) -> Dict[str, Any]:
        """Create a corridor preset."""
        state = MiniGridState(5, 13)
        state.add_border_walls()

        # Add internal walls with gaps
        wall = MINIGRID_OBJECTS["wall"]
        for r in range(1, 4):
            for c in [3, 6, 9]:
                if r != 2:  # Leave gap in middle
                    state.place_object(r, c, copy.deepcopy(wall))

        state.set_agent_position(2, 1, 0)
        state.place_object(2, 11, copy.deepcopy(MINIGRID_OBJECTS["goal"]))

        return state.to_dict()

    def _validate_state(self, state_dict: Dict[str, Any]) -> List[str]:
        """Validate the state dictionary."""
        errors = []

        if state_dict.get("agent_pos") is None:
            errors.append("Agent must be placed on the grid")

        rows = state_dict.get("rows", 0)
        cols = state_dict.get("cols", 0)
        if rows < 3 or cols < 3:
            errors.append("Grid must be at least 3x3")

        return errors

    def _create_state_from_dict(self, data: Dict[str, Any]) -> GridState:
        """Create a MiniGridState from dictionary."""
        state = MiniGridState()
        state.from_dict(data)

        # Update size spinbox if present
        if self._size_spinbox:
            rows, cols = state.get_dimensions()
            self._size_spinbox.blockSignals(True)
            self._size_spinbox.setValue(max(rows, cols))
            self._size_spinbox.blockSignals(False)

        return state
