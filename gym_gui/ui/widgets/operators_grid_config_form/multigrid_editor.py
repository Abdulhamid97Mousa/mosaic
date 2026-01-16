"""MultiGrid environment configuration dialog.

MultiGrid is a multi-agent extension of MiniGrid. Multiple agents act
simultaneously on a shared grid with walls, doors, keys, goals, etc.

Supported environments:
- SoccerGame4HEnv* - 2v2 soccer
- CollectGame4HEnv* - Cooperative collection
- Other MultiGrid variants

Key differences from MiniGrid:
- Multiple agents (2-4 typically)
- Agents have team colors (red vs blue)
- Simultaneous stepping
"""

import copy
import logging
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

_LOGGER = logging.getLogger(__name__)


# MultiGrid specific objects - extends MiniGrid with team agents
MULTIGRID_OBJECTS: Dict[str, GridObject] = {
    "empty": GridObject(ObjectType.EMPTY, ObjectColor.NONE, "", " "),
    "wall": GridObject(ObjectType.WALL, ObjectColor.GREY, "", "#"),
    "floor": GridObject(ObjectType.FLOOR, ObjectColor.NONE, "", "."),
    "goal": GridObject(ObjectType.GOAL, ObjectColor.GREEN, "", "G"),
    # Team agents
    "agent_red_0": GridObject(ObjectType.AGENT, ObjectColor.RED, "0", "R0"),
    "agent_red_1": GridObject(ObjectType.AGENT, ObjectColor.RED, "1", "R1"),
    "agent_blue_0": GridObject(ObjectType.AGENT, ObjectColor.BLUE, "0", "B0"),
    "agent_blue_1": GridObject(ObjectType.AGENT, ObjectColor.BLUE, "1", "B1"),
    # Collectible items
    "ball_red": GridObject(ObjectType.BALL, ObjectColor.RED, "", "o"),
    "ball_blue": GridObject(ObjectType.BALL, ObjectColor.BLUE, "", "o"),
    "ball_green": GridObject(ObjectType.BALL, ObjectColor.GREEN, "", "o"),
    # Soccer specific
    "ball_yellow": GridObject(ObjectType.BALL, ObjectColor.YELLOW, "", "S"),  # Soccer ball
    "goal_red": GridObject(ObjectType.GOAL, ObjectColor.RED, "", "GR"),
    "goal_blue": GridObject(ObjectType.GOAL, ObjectColor.BLUE, "", "GB"),
}

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


class MultiGridState(GridState):
    """Grid state for MultiGrid environments with multiple agents."""

    def __init__(self, rows: int = 10, cols: int = 15):
        self._rows = rows
        self._cols = cols
        self._grid: List[List[GridCell]] = []
        self._agents: Dict[str, Tuple[int, int, int]] = {}  # agent_id -> (row, col, dir)

        # Initialize empty grid
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(GridCell(r, c, []))
            self._grid.append(row)

    def get_cell(self, row: int, col: int) -> GridCell:
        if not self._is_valid(row, col):
            raise IndexError(f"Cell ({row}, {col}) out of bounds")
        return self._grid[row][col]

    def set_cell(self, cell: GridCell) -> None:
        if not self._is_valid(cell.row, cell.col):
            raise IndexError(f"Cell ({cell.row}, {cell.col}) out of bounds")
        self._grid[cell.row][cell.col] = cell

    def get_agent_position(self) -> Optional[Tuple[int, int]]:
        """Get first agent's position (for compatibility)."""
        if self._agents:
            agent_id = next(iter(self._agents))
            row, col, _ = self._agents[agent_id]
            return (row, col)
        return None

    def get_all_agents(self) -> Dict[str, Tuple[int, int, int]]:
        """Get all agent positions and directions."""
        return self._agents.copy()

    def set_agent_position(self, row: int, col: int, direction: int = 0) -> None:
        """Set agent position (adds as agent_0 if no agents exist)."""
        self.add_agent("agent_0", row, col, direction)

    def add_agent(
        self, agent_id: str, row: int, col: int, direction: int = 0
    ) -> bool:
        """Add or move an agent."""
        if not self._is_valid(row, col):
            return False

        # Remove from old position if exists
        if agent_id in self._agents:
            old_row, old_col, _ = self._agents[agent_id]
            self._grid[old_row][old_col].has_agent = False

        # Set new position
        self._agents[agent_id] = (row, col, direction % 4)
        self._grid[row][col].has_agent = True
        self._grid[row][col].agent_direction = direction % 4
        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the grid."""
        if agent_id not in self._agents:
            return False

        row, col, _ = self._agents[agent_id]
        del self._agents[agent_id]

        # Check if other agents are in this cell
        has_other_agent = any(
            (r, c) == (row, col)
            for aid, (r, c, _) in self._agents.items()
        )
        if not has_other_agent:
            self._grid[row][col].has_agent = False

        return True

    def place_object(self, row: int, col: int, obj: GridObject) -> bool:
        if not self._is_valid(row, col):
            return False

        cell = self._grid[row][col]

        # Handle agent placement
        if obj.obj_type == ObjectType.AGENT:
            agent_id = f"agent_{obj.color.value}_{obj.state}"
            return self.add_agent(agent_id, row, col, 0)

        # Don't place on walls
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

        # Handle agent removal
        if obj_type == ObjectType.AGENT or (obj_type is None and cell.has_agent):
            # Find and remove agent at this position
            for agent_id, (r, c, d) in list(self._agents.items()):
                if (r, c) == (row, col):
                    self.remove_agent(agent_id)
                    return GridObject(ObjectType.AGENT, ObjectColor.NONE, "", "A")
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
            "cells": cells_data,
            "agents": {
                aid: {"row": r, "col": c, "dir": d}
                for aid, (r, c, d) in self._agents.items()
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self._rows = data.get("rows", 10)
        self._cols = data.get("cols", 15)
        self._agents = {}

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
                    pass

        # Load agents
        for agent_id, agent_data in data.get("agents", {}).items():
            row = agent_data["row"]
            col = agent_data["col"]
            direction = agent_data.get("dir", 0)
            self.add_agent(agent_id, row, col, direction)

    def get_dimensions(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    def clear(self) -> None:
        for row in self._grid:
            for cell in row:
                cell.objects = []
                cell.has_agent = False
        self._agents = {}

    def copy(self) -> "MultiGridState":
        new_state = MultiGridState(self._rows, self._cols)
        new_state.from_dict(self.to_dict())
        return new_state

    def get_available_objects(self) -> List[GridObject]:
        return list(MULTIGRID_OBJECTS.values())

    def _is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self._rows and 0 <= col < self._cols

    def add_border_walls(self) -> None:
        wall = MULTIGRID_OBJECTS["wall"]
        for c in range(self._cols):
            self._grid[0][c].objects = [copy.deepcopy(wall)]
            self._grid[self._rows - 1][c].objects = [copy.deepcopy(wall)]
        for r in range(self._rows):
            self._grid[r][0].objects = [copy.deepcopy(wall)]
            self._grid[r][self._cols - 1].objects = [copy.deepcopy(wall)]


class MultiGridEditor(GridEditorWidget):
    """Grid editor widget for MultiGrid multi-agent environments."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state: Optional[MultiGridState] = None
        self._cell_size: int = 35
        self._margin: int = 20
        self.setMinimumSize(400, 300)

    def set_state(self, state: GridState) -> None:
        if not isinstance(state, MultiGridState):
            raise TypeError("Expected MultiGridState")
        self._state = state
        self.update()

    def get_state(self) -> GridState:
        if self._state is None:
            self._state = MultiGridState()
        return self._state

    def _get_cell_size(self) -> int:
        if self._state is None:
            return 35
        rows, cols = self._state.get_dimensions()
        available_w = self.width() - 2 * self._margin
        available_h = self.height() - 2 * self._margin
        size_by_width = available_w // cols if cols > 0 else 35
        size_by_height = available_h // rows if rows > 0 else 35
        return max(15, min(size_by_width, size_by_height, 50))

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
                color = COLOR_MAP.get(primary.color, "#51CF66")
                bg_color = QtGui.QColor(color)

        painter.fillRect(rect, bg_color)

        if self._show_grid_lines:
            painter.setPen(QtGui.QPen(QtGui.QColor("#CED4DA"), 1))
            painter.drawRect(rect)

        if self._hover_cell == (row, col):
            painter.fillRect(rect, QtGui.QColor(100, 149, 237, 80))

        # Draw objects
        for obj in cell.objects:
            if obj.obj_type in (ObjectType.EMPTY, ObjectType.FLOOR, ObjectType.WALL, ObjectType.GOAL):
                continue
            self._draw_object(painter, obj, rect)

        # Draw agents at this cell
        if self._state:
            for agent_id, (r, c, d) in self._state.get_all_agents().items():
                if (r, c) == (row, col):
                    color = ObjectColor.RED if "red" in agent_id else ObjectColor.BLUE
                    self._draw_agent(painter, d, rect, color, agent_id)

    def _draw_object(
        self,
        painter: QtGui.QPainter,
        obj: GridObject,
        rect: QtCore.QRect
    ) -> None:
        color = QtGui.QColor(COLOR_MAP.get(obj.color, "#DEE2E6"))
        margin = rect.width() // 6

        if obj.obj_type == ObjectType.BALL:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            inner = rect.adjusted(margin, margin, -margin, -margin)
            painter.drawEllipse(inner)

    def _draw_agent(
        self,
        painter: QtGui.QPainter,
        direction: int,
        rect: QtCore.QRect,
        team_color: ObjectColor,
        agent_id: str
    ) -> None:
        color = QtGui.QColor(COLOR_MAP.get(team_color, "#FF6B6B"))
        painter.setPen(QtGui.QPen(color.darker(120), 2))
        painter.setBrush(color)

        cx = rect.center().x()
        cy = rect.center().y()
        size = rect.width() // 3

        # Triangle pointing in direction
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

        # Draw agent number
        painter.setPen(QtGui.QColor("white"))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        # Extract number from agent_id like "agent_red_0" -> "0"
        num = agent_id.split("_")[-1] if "_" in agent_id else "?"
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, num)

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
            # Remove agent or object
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


class MultiGridObjectTray(GridObjectTray):
    """Object palette for MultiGrid environments."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._objects: List[GridObject] = []
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        categories = {
            "Red Team": [
                ("Agent R0", MULTIGRID_OBJECTS["agent_red_0"]),
                ("Agent R1", MULTIGRID_OBJECTS["agent_red_1"]),
            ],
            "Blue Team": [
                ("Agent B0", MULTIGRID_OBJECTS["agent_blue_0"]),
                ("Agent B1", MULTIGRID_OBJECTS["agent_blue_1"]),
            ],
            "Terrain": [
                ("Wall", MULTIGRID_OBJECTS["wall"]),
                ("Goal", MULTIGRID_OBJECTS["goal"]),
                ("Goal Red", MULTIGRID_OBJECTS["goal_red"]),
                ("Goal Blue", MULTIGRID_OBJECTS["goal_blue"]),
            ],
            "Items": [
                ("Ball", MULTIGRID_OBJECTS["ball_yellow"]),
                ("Red Ball", MULTIGRID_OBJECTS["ball_red"]),
                ("Blue Ball", MULTIGRID_OBJECTS["ball_blue"]),
            ],
        }

        for category, items in categories.items():
            label = QtWidgets.QLabel(category)
            label.setStyleSheet("font-weight: bold; color: #495057;")
            layout.addWidget(label)

            grid = QtWidgets.QGridLayout()
            grid.setSpacing(2)

            for i, (name, obj) in enumerate(items):
                btn = QtWidgets.QPushButton(name)
                btn.setCheckable(True)
                btn.setFixedHeight(26)

                color = COLOR_MAP.get(obj.color, "#DEE2E6")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color};
                        border: 1px solid #ADB5BD;
                        border-radius: 3px;
                        font-size: 10px;
                    }}
                    QPushButton:checked {{
                        border: 2px solid #228BE6;
                        font-weight: bold;
                    }}
                """)

                btn.clicked.connect(
                    lambda checked, o=obj, b=btn: self._on_button_clicked(o, b)
                )
                self._buttons[name] = btn
                grid.addWidget(btn, i // 2, i % 2)

            layout.addLayout(grid)

        layout.addStretch()

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


class MultiGridConfigDialog(GridConfigDialog):
    """Configuration dialog for MultiGrid multi-agent environments."""

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        env_id: str = "SoccerGame4HEnv10x15N2"
    ):
        self._env_id = env_id
        super().__init__(initial_state, parent)

    def _get_title(self) -> str:
        return f"Configure MultiGrid - {self._env_id}"

    def _create_grid_editor(self) -> GridEditorWidget:
        return MultiGridEditor()

    def _create_object_tray(self) -> GridObjectTray:
        return MultiGridObjectTray()

    def _get_presets(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("Soccer 10x15", self._create_soccer_preset()),
            ("Collect 10x10", self._create_collect_preset()),
            ("Empty 8x8", self._create_empty_preset(8, 8)),
        ]

    def _create_empty_preset(self, rows: int, cols: int) -> Dict[str, Any]:
        state = MultiGridState(rows, cols)
        state.add_border_walls()
        state.add_agent("agent_red_0", 1, 1, 0)
        state.add_agent("agent_blue_0", rows - 2, cols - 2, 2)
        return state.to_dict()

    def _create_soccer_preset(self) -> Dict[str, Any]:
        state = MultiGridState(10, 15)
        state.add_border_walls()

        # Goals on left/right sides
        for r in range(3, 7):
            state.place_object(r, 0, copy.deepcopy(MULTIGRID_OBJECTS["goal_red"]))
            state.place_object(r, 14, copy.deepcopy(MULTIGRID_OBJECTS["goal_blue"]))

        # Agents
        state.add_agent("agent_red_0", 3, 3, 0)
        state.add_agent("agent_red_1", 6, 3, 0)
        state.add_agent("agent_blue_0", 3, 11, 2)
        state.add_agent("agent_blue_1", 6, 11, 2)

        # Ball in center
        state.place_object(5, 7, copy.deepcopy(MULTIGRID_OBJECTS["ball_yellow"]))

        return state.to_dict()

    def _create_collect_preset(self) -> Dict[str, Any]:
        state = MultiGridState(10, 10)
        state.add_border_walls()

        # Agents
        state.add_agent("agent_red_0", 1, 1, 0)
        state.add_agent("agent_blue_0", 8, 8, 2)

        # Collectible items
        state.place_object(2, 5, copy.deepcopy(MULTIGRID_OBJECTS["ball_green"]))
        state.place_object(5, 2, copy.deepcopy(MULTIGRID_OBJECTS["ball_green"]))
        state.place_object(7, 5, copy.deepcopy(MULTIGRID_OBJECTS["ball_green"]))
        state.place_object(5, 7, copy.deepcopy(MULTIGRID_OBJECTS["ball_green"]))

        return state.to_dict()

    def _validate_state(self, state_dict: Dict[str, Any]) -> List[str]:
        errors = []
        if not state_dict.get("agents"):
            errors.append("At least one agent must be placed")
        return errors

    def _create_state_from_dict(self, data: Dict[str, Any]) -> GridState:
        state = MultiGridState()
        state.from_dict(data)
        return state
