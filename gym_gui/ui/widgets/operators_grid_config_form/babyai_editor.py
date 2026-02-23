"""BabyAI environment configuration dialog.

BabyAI is a language-grounded extension of MiniGrid where agents must
follow natural language instructions to complete tasks.

This editor provides:
- All 6 BabyAI colors: red, green, blue, purple, yellow, grey
- All object types: ball, box, key, door
- Mission text field for language instructions
- Automatic grid size detection from environment ID (S5=5x5, S6=6x6, etc.)
- Presets for common BabyAI tasks (GoTo, PickUp, Open, Put)

Supported environments:
- BabyAI-GoToRedBall-v0
- BabyAI-GoToObj-v0
- BabyAI-GoToObjS5-v1 (5x5 room)
- BabyAI-GoToObjS6-v1 (6x6 room)
- BabyAI-PickupLoc-v0
- BabyAI-OpenDoor-v0
- BabyAI-PutNextLocal-v0
- And other BabyAI variants
"""

import copy
import re
from typing import Dict, List, Optional, Tuple, Any

from PyQt6 import QtWidgets, QtCore, QtGui

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


# BabyAI uses 6 colors for language grounding
BABYAI_COLORS = [
    ObjectColor.RED,
    ObjectColor.GREEN,
    ObjectColor.BLUE,
    ObjectColor.PURPLE,
    ObjectColor.YELLOW,
    ObjectColor.GREY,
]

# Full object definitions for BabyAI (all 6 colors)
BABYAI_OBJECTS: Dict[str, GridObject] = {
    "empty": GridObject(ObjectType.EMPTY, ObjectColor.NONE, "", " "),
    "wall": GridObject(ObjectType.WALL, ObjectColor.GREY, "", "#"),
    "floor": GridObject(ObjectType.FLOOR, ObjectColor.NONE, "", "."),
    "goal": GridObject(ObjectType.GOAL, ObjectColor.GREEN, "", "G"),
    "lava": GridObject(ObjectType.LAVA, ObjectColor.RED, "", "~"),
}

# Add all colored objects
for color in BABYAI_COLORS:
    color_name = color.value
    BABYAI_OBJECTS[f"ball_{color_name}"] = GridObject(ObjectType.BALL, color, "", "o")
    BABYAI_OBJECTS[f"box_{color_name}"] = GridObject(ObjectType.BOX, color, "", "B")
    BABYAI_OBJECTS[f"key_{color_name}"] = GridObject(ObjectType.KEY, color, "", "k")
    BABYAI_OBJECTS[f"door_{color_name}"] = GridObject(ObjectType.DOOR, color, "closed", "D")


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


def parse_grid_size_from_env_id(env_id: str) -> int:
    """Parse grid size from BabyAI environment ID.

    BabyAI uses the "S" prefix convention to indicate room size:
    - BabyAI-GoToObjS5-v1 → 5x5 room
    - BabyAI-GoToObjS6-v1 → 6x6 room
    - BabyAI-GoToRedBall-v0 → 8x8 (default, no S prefix)

    The actual grid includes walls around the room, so S6 means a 6x6
    playable area which becomes an 8x8 grid with walls (but MiniGrid/BabyAI
    uses room_size directly, where walls are part of the size).

    Args:
        env_id: BabyAI environment identifier

    Returns:
        Grid size (both rows and cols since BabyAI uses square grids)
    """
    # Look for S followed by digits in the env_id
    match = re.search(r"S(\d+)", env_id)
    if match:
        return int(match.group(1))

    # Default size for BabyAI environments without explicit size
    return 8


class BabyAIState(GridState):
    """Grid state for BabyAI environments with mission text support."""

    def __init__(self, rows: int = 8, cols: int = 8, mission: str = ""):
        self._rows = rows
        self._cols = cols
        self._mission = mission
        self._grid: List[List[GridCell]] = []
        self._agent_pos: Optional[Tuple[int, int]] = None
        self._agent_dir: int = 0

        # Initialize empty grid
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(GridCell(r, c, []))
            self._grid.append(row)

    @property
    def mission(self) -> str:
        """Get the mission instruction text."""
        return self._mission

    @mission.setter
    def mission(self, value: str) -> None:
        """Set the mission instruction text."""
        self._mission = value

    def get_cell(self, row: int, col: int) -> GridCell:
        if not self._is_valid(row, col):
            raise IndexError(f"Cell ({row}, {col}) out of bounds")
        return self._grid[row][col]

    def set_cell(self, cell: GridCell) -> None:
        if not self._is_valid(cell.row, cell.col):
            raise IndexError(f"Cell ({cell.row}, {cell.col}) out of bounds")
        self._grid[cell.row][cell.col] = cell

    def get_agent_position(self) -> Optional[Tuple[int, int]]:
        return self._agent_pos

    def set_agent_position(self, row: int, col: int, direction: int = 0) -> None:
        if not self._is_valid(row, col):
            raise IndexError(f"Position ({row}, {col}) out of bounds")

        if self._agent_pos:
            old_row, old_col = self._agent_pos
            self._grid[old_row][old_col].has_agent = False

        self._agent_pos = (row, col)
        self._agent_dir = direction % 4
        self._grid[row][col].has_agent = True
        self._grid[row][col].agent_direction = direction % 4

    def place_object(self, row: int, col: int, obj: GridObject) -> bool:
        if not self._is_valid(row, col):
            return False

        cell = self._grid[row][col]

        if obj.obj_type == ObjectType.AGENT:
            self.set_agent_position(row, col, 0)
            return True

        for existing in cell.objects:
            if existing.obj_type == ObjectType.WALL:
                return False
            if existing == obj:
                return False

        cell.objects = [o for o in cell.objects if o.obj_type != ObjectType.EMPTY]

        if obj.obj_type == ObjectType.WALL:
            cell.objects = [obj]
        else:
            cell.objects.append(obj)

        return True

    def remove_object(
        self, row: int, col: int, obj_type: Optional[ObjectType] = None
    ) -> Optional[GridObject]:
        if not self._is_valid(row, col):
            return None

        cell = self._grid[row][col]

        if obj_type == ObjectType.AGENT or (obj_type is None and cell.has_agent):
            if self._agent_pos == (row, col):
                self._agent_pos = None
                cell.has_agent = False
                return GridObject(ObjectType.AGENT, ObjectColor.RED, "", "A")
            return None

        if not cell.objects:
            return None

        if obj_type:
            for i, obj in enumerate(cell.objects):
                if obj.obj_type == obj_type:
                    return cell.objects.pop(i)
            return None
        else:
            for i in range(len(cell.objects) - 1, -1, -1):
                if cell.objects[i].obj_type not in (ObjectType.EMPTY, ObjectType.FLOOR):
                    return cell.objects.pop(i)
            return None

    def to_dict(self) -> Dict[str, Any]:
        cells_data = []
        for row in self._grid:
            for cell in row:
                if cell.objects:
                    cells_data.append({
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
                    })

        return {
            "rows": self._rows,
            "cols": self._cols,
            "mission": self._mission,
            "agent_pos": list(self._agent_pos) if self._agent_pos else None,
            "agent_dir": self._agent_dir,
            "cells": cells_data,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self._rows = data.get("rows", 8)
        self._cols = data.get("cols", 8)
        self._mission = data.get("mission", "")
        self._agent_pos = None
        self._agent_dir = 0

        self._grid = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                row.append(GridCell(r, c, []))
            self._grid.append(row)

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
                    pass

        agent_pos = data.get("agent_pos")
        agent_dir = data.get("agent_dir", 0)
        if agent_pos:
            self.set_agent_position(agent_pos[0], agent_pos[1], agent_dir)

    def get_dimensions(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    def clear(self) -> None:
        for row in self._grid:
            for cell in row:
                cell.objects = []
                cell.has_agent = False
        self._agent_pos = None
        self._mission = ""

    def copy(self) -> "BabyAIState":
        new_state = BabyAIState(self._rows, self._cols, self._mission)
        new_state.from_dict(self.to_dict())
        return new_state

    def get_available_objects(self) -> List[GridObject]:
        return list(BABYAI_OBJECTS.values())

    def _is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self._rows and 0 <= col < self._cols

    def add_border_walls(self) -> None:
        wall = BABYAI_OBJECTS["wall"]
        for c in range(self._cols):
            self._grid[0][c].objects = [copy.deepcopy(wall)]
            self._grid[self._rows - 1][c].objects = [copy.deepcopy(wall)]
        for r in range(self._rows):
            self._grid[r][0].objects = [copy.deepcopy(wall)]
            self._grid[r][self._cols - 1].objects = [copy.deepcopy(wall)]


class BabyAIEditor(GridEditorWidget):
    """Grid editor widget for BabyAI environments."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state: Optional[BabyAIState] = None
        self._cell_size: int = 40
        self._margin: int = 20
        self.setMinimumSize(400, 400)

    def set_state(self, state: GridState) -> None:
        if not isinstance(state, BabyAIState):
            raise TypeError("Expected BabyAIState")
        self._state = state
        self.update()

    def get_state(self) -> GridState:
        if self._state is None:
            self._state = BabyAIState()
        return self._state

    def _get_cell_size(self) -> int:
        if self._state is None:
            return 40
        rows, cols = self._state.get_dimensions()
        available_w = self.width() - 2 * self._margin
        available_h = self.height() - 2 * self._margin
        size_by_width = available_w // cols if cols > 0 else 40
        size_by_height = available_h // rows if rows > 0 else 40
        return max(20, min(size_by_width, size_by_height, 60))

    def _pos_to_cell(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        cell_size = self._get_cell_size()
        col = (pos.x() - self._margin) // cell_size
        row = (pos.y() - self._margin) // cell_size
        return (row, col)

    def _cell_to_rect(self, row: int, col: int) -> QtCore.QRect:
        cell_size = self._get_cell_size()
        x = self._margin + col * cell_size
        y = self._margin + row * cell_size
        return QtCore.QRect(x, y, cell_size, cell_size)

    def _is_valid_cell(self, row: int, col: int) -> bool:
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
        if self._state is None:
            return

        cell = self._state.get_cell(row, col)
        bg_color = QtGui.QColor("#F8F9FA")

        primary = cell.get_primary_object()
        if primary:
            if primary.obj_type == ObjectType.WALL:
                bg_color = QtGui.QColor("#495057")
            elif primary.obj_type == ObjectType.GOAL:
                bg_color = QtGui.QColor("#D3F9D8")
            elif primary.obj_type == ObjectType.LAVA:
                bg_color = QtGui.QColor("#FFE3E3")

        painter.fillRect(rect, bg_color)

        if self._show_grid_lines:
            painter.setPen(QtGui.QPen(QtGui.QColor("#CED4DA"), 1))
            painter.drawRect(rect)

        if self._hover_cell == (row, col):
            painter.fillRect(rect, QtGui.QColor(100, 149, 237, 80))

        for obj in cell.objects:
            if obj.obj_type in (ObjectType.EMPTY, ObjectType.FLOOR, ObjectType.WALL):
                continue
            self._draw_object(painter, obj, rect)

        if cell.has_agent:
            self._draw_agent(painter, cell.agent_direction, rect)

    def _draw_object(
        self,
        painter: QtGui.QPainter,
        obj: GridObject,
        rect: QtCore.QRect
    ) -> None:
        color = QtGui.QColor(COLOR_MAP.get(obj.color, "#DEE2E6"))
        margin = rect.width() // 6

        if obj.obj_type == ObjectType.KEY:
            painter.setPen(QtGui.QPen(color.darker(120), 2))
            painter.setBrush(color)
            inner = rect.adjusted(margin * 2, margin, -margin * 2, -margin)
            painter.drawEllipse(inner.adjusted(0, 0, 0, -inner.height() // 2))
            painter.drawRect(
                inner.x() + inner.width() // 3,
                inner.y() + inner.height() // 2,
                inner.width() // 3,
                inner.height() // 2
            )

        elif obj.obj_type == ObjectType.DOOR:
            painter.setPen(QtGui.QPen(color.darker(120), 2))
            if obj.state == "open":
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            else:
                painter.setBrush(color)
            inner = rect.adjusted(margin, margin // 2, -margin, -margin // 2)
            painter.drawRect(inner)
            if obj.state != "open":
                handle_x = inner.x() + inner.width() - margin * 2
                handle_y = inner.center().y()
                painter.drawEllipse(handle_x, handle_y - 3, 6, 6)

        elif obj.obj_type == ObjectType.BALL:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            inner = rect.adjusted(margin, margin, -margin, -margin)
            painter.drawEllipse(inner)

        elif obj.obj_type == ObjectType.BOX:
            painter.setPen(QtGui.QPen(color.darker(120), 2))
            painter.setBrush(color)
            inner = rect.adjusted(margin, margin, -margin, -margin)
            painter.drawRect(inner)

        elif obj.obj_type == ObjectType.GOAL:
            painter.setPen(QtGui.QPen(color.darker(120), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            inner = rect.adjusted(margin, margin, -margin, -margin)
            painter.drawRect(inner)
            painter.drawLine(inner.topLeft(), inner.bottomRight())
            painter.drawLine(inner.topRight(), inner.bottomLeft())

    def _draw_agent(
        self,
        painter: QtGui.QPainter,
        direction: int,
        rect: QtCore.QRect
    ) -> None:
        color = QtGui.QColor("#FF6B6B")
        painter.setPen(QtGui.QPen(color.darker(120), 2))
        painter.setBrush(color)

        cx = rect.center().x()
        cy = rect.center().y()
        size = rect.width() // 3

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
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#E9ECEF"))

        if self._state is None:
            return

        rows, cols = self._state.get_dimensions()
        for row in range(rows):
            for col in range(cols):
                rect = self._cell_to_rect(row, col)
                self._draw_cell(painter, row, col, rect)

    def _handle_cell_click(
        self,
        row: int,
        col: int,
        event: QtGui.QMouseEvent
    ) -> None:
        if self._state is None:
            return

        if (
            event.button() == QtCore.Qt.MouseButton.RightButton
            or event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            removed = self._state.remove_object(row, col)
            if removed:
                self.object_removed.emit(removed, row, col)
            self.grid_changed.emit()
            self.update()
            return

        if self._selected_object is not None:
            obj_copy = copy.deepcopy(self._selected_object)
            if self._state.place_object(row, col, obj_copy):
                self.object_placed.emit(obj_copy, row, col)
            self.grid_changed.emit()
            self.update()


class BabyAIObjectTray(GridObjectTray):
    """Object palette for BabyAI environments with all 6 colors."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._objects: List[GridObject] = []
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # Scroll area for many objects
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setSpacing(8)

        # Terrain section
        terrain_group = QtWidgets.QGroupBox("Terrain")
        terrain_layout = QtWidgets.QGridLayout(terrain_group)
        terrain_items = [
            ("Wall", BABYAI_OBJECTS["wall"]),
            ("Goal", BABYAI_OBJECTS["goal"]),
            ("Lava", BABYAI_OBJECTS["lava"]),
        ]
        for i, (name, obj) in enumerate(terrain_items):
            btn = self._create_button(name, obj)
            terrain_layout.addWidget(btn, i // 3, i % 3)
        container_layout.addWidget(terrain_group)

        # Balls section (all 6 colors)
        balls_group = QtWidgets.QGroupBox("Balls")
        balls_layout = QtWidgets.QGridLayout(balls_group)
        for i, color in enumerate(BABYAI_COLORS):
            name = color.value.title()
            obj = BABYAI_OBJECTS[f"ball_{color.value}"]
            btn = self._create_button(name, obj)
            balls_layout.addWidget(btn, i // 3, i % 3)
        container_layout.addWidget(balls_group)

        # Boxes section (all 6 colors)
        boxes_group = QtWidgets.QGroupBox("Boxes")
        boxes_layout = QtWidgets.QGridLayout(boxes_group)
        for i, color in enumerate(BABYAI_COLORS):
            name = color.value.title()
            obj = BABYAI_OBJECTS[f"box_{color.value}"]
            btn = self._create_button(name, obj)
            boxes_layout.addWidget(btn, i // 3, i % 3)
        container_layout.addWidget(boxes_group)

        # Keys section (all 6 colors)
        keys_group = QtWidgets.QGroupBox("Keys")
        keys_layout = QtWidgets.QGridLayout(keys_group)
        for i, color in enumerate(BABYAI_COLORS):
            name = color.value.title()
            obj = BABYAI_OBJECTS[f"key_{color.value}"]
            btn = self._create_button(name, obj)
            keys_layout.addWidget(btn, i // 3, i % 3)
        container_layout.addWidget(keys_group)

        # Doors section (all 6 colors)
        doors_group = QtWidgets.QGroupBox("Doors")
        doors_layout = QtWidgets.QGridLayout(doors_group)
        for i, color in enumerate(BABYAI_COLORS):
            name = color.value.title()
            obj = BABYAI_OBJECTS[f"door_{color.value}"]
            btn = self._create_button(name, obj)
            doors_layout.addWidget(btn, i // 3, i % 3)
        container_layout.addWidget(doors_group)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _create_button(self, name: str, obj: GridObject) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(name)
        btn.setCheckable(True)
        btn.setFixedHeight(28)

        color = COLOR_MAP.get(obj.color, "#DEE2E6")
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: 1px solid #ADB5BD;
                border-radius: 3px;
                font-size: 10px;
                padding: 2px 4px;
            }}
            QPushButton:checked {{
                border: 2px solid #228BE6;
                font-weight: bold;
            }}
        """)

        btn.clicked.connect(
            lambda checked, o=obj, b=btn: self._on_button_clicked(o, b)
        )
        self._buttons[f"{obj.obj_type.value}_{obj.color.value}"] = btn
        return btn

    def _on_button_clicked(self, obj: GridObject, btn: QtWidgets.QPushButton) -> None:
        for other_btn in self._buttons.values():
            if other_btn != btn:
                other_btn.setChecked(False)

        if btn.isChecked():
            self._selected_object = obj
            self.object_selected.emit(obj)
        else:
            self._selected_object = None
            self.object_selected.emit(None)

    def set_available_objects(self, objects: List[GridObject]) -> None:
        self._objects = objects

    def get_selected_object(self) -> Optional[GridObject]:
        return self._selected_object


class BabyAIConfigDialog(GridConfigDialog):
    """Configuration dialog for BabyAI language-grounded environments."""

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        env_id: str = "BabyAI-GoToRedBall-v0"
    ):
        self._env_id = env_id
        self._grid_size = parse_grid_size_from_env_id(env_id)
        self._mission_edit: Optional[QtWidgets.QLineEdit] = None

        # If no initial state provided, create default state with correct grid size
        if initial_state is None:
            default_state = BabyAIState(self._grid_size, self._grid_size)
            default_state.add_border_walls()
            default_state.set_agent_position(1, 1, 0)
            initial_state = default_state.to_dict()

        super().__init__(initial_state, parent)

    def _get_title(self) -> str:
        return f"Configure BabyAI - {self._env_id}"

    def _setup_ui(self) -> None:
        """Override to add mission text field."""
        super()._setup_ui()

        # Add mission field after the grid editor
        mission_layout = QtWidgets.QHBoxLayout()
        mission_label = QtWidgets.QLabel("Mission:")
        mission_label.setStyleSheet("font-weight: bold;")
        self._mission_edit = QtWidgets.QLineEdit()
        self._mission_edit.setPlaceholderText("e.g., 'go to the red ball'")
        self._mission_edit.textChanged.connect(self._on_mission_changed)
        mission_layout.addWidget(mission_label)
        mission_layout.addWidget(self._mission_edit, 1)

        # Insert mission field before the buttons
        main_layout = self.layout()
        if main_layout:
            main_layout.insertLayout(main_layout.count() - 1, mission_layout)

    def _on_mission_changed(self, text: str) -> None:
        """Update state when mission text changes."""
        if self._grid_editor and isinstance(self._grid_editor.get_state(), BabyAIState):
            state = self._grid_editor.get_state()
            state.mission = text

    def _create_grid_editor(self) -> GridEditorWidget:
        return BabyAIEditor()

    def _create_object_tray(self) -> GridObjectTray:
        return BabyAIObjectTray()

    def _get_presets(self) -> List[Tuple[str, Dict[str, Any]]]:
        size = self._grid_size
        return [
            ("GoTo Red Ball", self._create_goto_preset("red", "ball")),
            ("GoTo Blue Box", self._create_goto_preset("blue", "box")),
            ("PickUp Key", self._create_pickup_preset()),
            ("Open Door", self._create_open_door_preset()),
            (f"Empty {size}x{size}", self._create_empty_preset(size, size)),
        ]

    def _create_empty_preset(self, rows: int, cols: int) -> Dict[str, Any]:
        state = BabyAIState(rows, cols)
        state.add_border_walls()
        state.set_agent_position(1, 1, 0)
        return state.to_dict()

    def _create_goto_preset(self, color: str, obj_type: str) -> Dict[str, Any]:
        size = self._grid_size
        state = BabyAIState(size, size, f"go to the {color} {obj_type}")
        state.add_border_walls()
        state.set_agent_position(1, 1, 0)

        # Place target object in lower-right area (accounting for walls)
        obj = BABYAI_OBJECTS.get(f"{obj_type}_{color}")
        if obj:
            target_pos = size - 2  # One cell inside the wall
            state.place_object(target_pos, target_pos, copy.deepcopy(obj))

        return state.to_dict()

    def _create_pickup_preset(self) -> Dict[str, Any]:
        size = self._grid_size
        state = BabyAIState(size, size, "pick up the yellow key")
        state.add_border_walls()
        state.set_agent_position(1, 1, 0)

        # Place key in center area
        key = BABYAI_OBJECTS["key_yellow"]
        center = size // 2
        state.place_object(center, center, copy.deepcopy(key))

        return state.to_dict()

    def _create_open_door_preset(self) -> Dict[str, Any]:
        size = self._grid_size
        state = BabyAIState(size, size, "open the purple door")
        state.add_border_walls()
        state.set_agent_position(1, 1, 0)

        # Place door in center and key nearby
        door = BABYAI_OBJECTS["door_purple"]
        key = BABYAI_OBJECTS["key_purple"]
        center = size // 2
        state.place_object(center, center, copy.deepcopy(door))
        state.place_object(2, 2, copy.deepcopy(key))

        return state.to_dict()

    def _validate_state(self, state_dict: Dict[str, Any]) -> List[str]:
        errors = []
        if not state_dict.get("agent_pos"):
            errors.append("Agent must be placed on the grid")
        return errors

    def _create_state_from_dict(self, data: Dict[str, Any]) -> GridState:
        state = BabyAIState()
        state.from_dict(data)

        # Update mission field if present
        if self._mission_edit:
            self._mission_edit.setText(state.mission)

        return state

    def get_state(self) -> Dict[str, Any]:
        """Override to include mission text."""
        state = super().get_state()
        if self._mission_edit:
            state["mission"] = self._mission_edit.text()
        return state


__all__ = [
    "BABYAI_OBJECTS",
    "BabyAIState",
    "BabyAIEditor",
    "BabyAIObjectTray",
    "BabyAIConfigDialog",
    "parse_grid_size_from_env_id",
]
