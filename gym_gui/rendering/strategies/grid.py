"""Grid renderer strategy that wraps the legacy GridRenderer helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import GameId, RenderMode
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_WARNING,
)
from gym_gui.rendering.assets import (
    CliffWalkingAssets,
    FrozenLakeAssets,
    TaxiAssets,
    get_asset_manager,
)
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy


class GridRendererStrategy(RendererStrategy):
    """Renderer strategy for grid-based environments."""

    mode = RenderMode.GRID

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._view = QtWidgets.QGraphicsView(parent)
        self._view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(240, 240, 240)))
        self._renderer = _GridRenderer(self._view)
        self._current_game: GameId | None = None

    # ------------------------------------------------------------------
    # RendererStrategy API
    # ------------------------------------------------------------------
    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._view

    def render(self, payload: Mapping[str, object], *, context: RendererContext | None = None) -> None:
        if not self.supports(payload):
            self.reset()
            return

        grid_payload = payload.get("grid")
        if grid_payload is None:
            self.reset()
            return

        rows = _normalise_grid(grid_payload)
        if not rows:
            self.reset()
            return

        game_id = _resolve_game_id(context, payload) or GameId.FROZEN_LAKE
        agent_pos = payload.get("agent_position")
        taxi_state = payload.get("taxi_state")
        terminated = bool(payload.get("terminated", False))

        self._renderer.render(
            rows,
            game_id,
            agent_position=_as_tuple(agent_pos),
            taxi_state=_as_dict(taxi_state),
            terminated=terminated,
            payload=dict(payload),
        )
        self._current_game = game_id

    def supports(self, payload: Mapping[str, object]) -> bool:
        return "grid" in payload

    def reset(self) -> None:
        scene = self._view.scene()
        if scene is not None:
            scene.clear()
        self._current_game = None

    def cleanup(self) -> None:
        """Clean up resources before widget destruction.

        This prevents segmentation faults from Qt trying to access
        deleted scene items after the widget is destroyed.
        """
        try:
            # Clear the scene to remove all items
            scene = self._view.scene()
            if scene is not None:
                scene.clear()

            # Clear renderer state
            self._renderer._current_grid = []
            self._renderer._last_actor_position = None
            self._current_game = None
        except Exception:
            # Silently ignore errors during cleanup
            pass


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _normalise_grid(raw_grid: object) -> List[List[str]]:
    rows: List[List[str]] = []
    if isinstance(raw_grid, str):
        rows.append(list(raw_grid))
        return rows
    if isinstance(raw_grid, Mapping):  # incompatible structure
        return rows

    iterable: Iterable[Any]
    if isinstance(raw_grid, Iterable):
        iterable = raw_grid  # type: ignore[assignment]
    else:
        return rows

    for row in iterable:
        if isinstance(row, str):
            rows.append(list(row))
        elif isinstance(row, Iterable):
            rows.append([str(cell) for cell in list(row)])
        else:
            rows.append([str(row)])
    return rows


def _resolve_game_id(context: RendererContext | None, payload: Mapping[str, object]) -> GameId | None:
    if context and context.game_id is not None:
        return context.game_id
    raw_game = payload.get("game_id")
    if raw_game is None:
        return None
    if isinstance(raw_game, GameId):
        return raw_game
    if isinstance(raw_game, str):
        try:
            return GameId(raw_game)
        except ValueError:
            return None
    return None


def _as_tuple(value: object) -> tuple[int, int] | None:
    if isinstance(value, tuple) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, list) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _as_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


class _GridRenderer(LogConstantMixin):
    """Legacy asset-backed grid renderer used by the grid strategy."""

    def __init__(self, graphics_view: QtWidgets.QGraphicsView) -> None:
        self._logger = logging.getLogger(__name__)
        self._view = graphics_view
        self._scene = QtWidgets.QGraphicsScene()
        self._view.setScene(self._scene)

        self._view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._asset_manager = get_asset_manager()
        self._current_game: GameId | None = None
        self._current_grid: List[List[str]] = []
        self._tile_size = 120  # Increased from 48 for better visibility
        self._last_actor_position: tuple[int, int] | None = None

    def render(
        self,
        grid: List[List[str]],
        game_id: GameId,
        *,
        agent_position: tuple[int, int] | None = None,
        taxi_state: Dict[str, Any] | None = None,
        terminated: bool = False,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        self._current_game = game_id
        self._current_grid = grid
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Log at WARNING level to ensure visibility
        self._logger.warning(f"_GridRenderer.render() CALLED: game_id={game_id}, grid_size={rows}x{cols}, agent_pos={agent_position}, has_taxi_state={taxi_state is not None}")
        self.log_constant(
            LOG_UI_RENDER_TABS_INFO,
            message=f"_GridRenderer.render() START: game_id={game_id}, grid_size={rows}x{cols}, agent_position={agent_position}"
        )

        self._scene.clear()

        effective_actor_position = agent_position
        if terminated and effective_actor_position is None and self._last_actor_position is not None:
            effective_actor_position = self._last_actor_position

        pixmap_count = 0
        for r, row in enumerate(grid):
            for c, cell_value in enumerate(row):
                pixmap = self._create_cell_pixmap(
                    r,
                    c,
                    cell_value,
                    effective_actor_position,
                    taxi_state,
                    terminated,
                    payload,
                )
                if pixmap and not pixmap.isNull():
                    pixmap_count += 1
                    item = self._scene.addPixmap(pixmap)
                    if item is not None:
                        item.setPos(c * self._tile_size, r * self._tile_size)

        # Log completion at WARNING level for visibility
        self._logger.warning(f"_GridRenderer.render() COMPLETE: Added {pixmap_count} pixmaps to scene (game={self._current_game}, grid={rows}x{cols})")
        self.log_constant(
            LOG_UI_RENDER_TABS_INFO,
            message=f"_GridRenderer.render() COMPLETE: Added {pixmap_count} pixmaps to scene"
        )
        self._scene.setSceneRect(0, 0, cols * self._tile_size, rows * self._tile_size)
        self._view.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        if agent_position is not None:
            self._last_actor_position = agent_position
        elif not terminated:
            self._last_actor_position = None

    def _create_cell_pixmap(
        self,
        row: int,
        col: int,
        cell_value: str,
        actor_position: tuple[int, int] | None,
        taxi_state: Dict[str, Any] | None,
        terminated: bool = False,
        payload: Dict[str, Any] | None = None,
    ) -> QtGui.QPixmap | None:
        is_actor_cell = actor_position is not None and actor_position == (row, col)

        if self._current_game == GameId.TAXI:
            pixmap = self._asset_manager.get_pixmap("taxi_background.png")
            if pixmap is None:
                self._logger.error(f"TAXI ASSET MISSING: taxi_background.png not found at cell ({row},{col})")
                self.log_constant(
                    LOG_UI_RENDER_TABS_WARNING,
                    message=f"TAXI cell ({row},{col}): taxi_background.png not found"
                )
                return None
            self._logger.debug(f"TAXI cell ({row},{col}): Got taxi_background.png")
            self.log_constant(
                LOG_UI_RENDER_TABS_TRACE,
                message=f"TAXI cell ({row},{col}): Got taxi_background.png"
            )
        elif self._current_game in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2):
            base_pixmap = self._asset_manager.get_pixmap(FrozenLakeAssets.ICE)
            if base_pixmap is None:
                return None
            pixmap = base_pixmap

            tile_code = cell_value.strip().upper()
            overlay_asset: str | None
            if terminated and is_actor_cell and tile_code == "H":
                overlay_asset = FrozenLakeAssets.CRACKED_HOLE
            elif tile_code == "H":
                overlay_asset = FrozenLakeAssets.HOLE
            elif tile_code == "G":
                overlay_asset = FrozenLakeAssets.GOAL
            elif tile_code == "S":
                overlay_asset = FrozenLakeAssets.STOOL
            else:
                overlay_asset = None

            if overlay_asset:
                overlay_pixmap = self._asset_manager.get_pixmap(overlay_asset)
                if overlay_pixmap:
                    pixmap = self._composite_pixmaps(pixmap, overlay_pixmap)
        elif self._current_game == GameId.CLIFF_WALKING:
            layer_names = CliffWalkingAssets.get_tile_layers(cell_value, row, col)
            pixmap = None
            for asset_name in layer_names:
                layer_pixmap = self._asset_manager.get_pixmap(asset_name)
                if layer_pixmap is None:
                    continue
                if pixmap is None:
                    pixmap = layer_pixmap
                else:
                    pixmap = self._composite_pixmaps(pixmap, layer_pixmap)
            if pixmap is None:
                return None
        else:
            asset_name = self._get_tile_asset(cell_value, row, col)
            pixmap = self._asset_manager.get_pixmap(asset_name)
            if pixmap is None:
                return None

        if self._current_game == GameId.TAXI and cell_value != ':' and cell_value.strip():
            if cell_value not in ('R', 'G', 'Y', 'B'):
                structural_asset = self._get_tile_asset(cell_value, row, col)
                structural_pixmap = self._asset_manager.get_pixmap(structural_asset)
                if structural_pixmap:
                    pixmap = self._composite_pixmaps(pixmap, structural_pixmap)

        if self._current_game == GameId.TAXI:
            if cell_value in ('R', 'G', 'Y', 'B'):
                depot_pixmap = self._create_depot_letter_pixmap(cell_value)
                if depot_pixmap:
                    pixmap = self._composite_pixmaps(pixmap, depot_pixmap)

            if taxi_state:
                pass_idx = taxi_state.get("passenger_index", -1)
                if pass_idx < 4:
                    pass_pos = self._get_taxi_depot_position(pass_idx)
                    if pass_pos == (row, col):
                        passenger_pixmap = self._asset_manager.get_pixmap("passenger.png")
                        if passenger_pixmap:
                            pixmap = self._composite_pixmaps(pixmap, passenger_pixmap)

                dest_idx = taxi_state.get("destination_index", -1)
                if dest_idx < 4:
                    dest_pos = self._get_taxi_depot_position(dest_idx)
                    if dest_pos == (row, col):
                        hotel_pixmap = self._asset_manager.get_pixmap("hotel.png")
                        if hotel_pixmap:
                            scaled_size = int(self._tile_size * 0.65)
                            hotel_scaled = hotel_pixmap.scaled(
                                scaled_size,
                                scaled_size,
                                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                QtCore.Qt.TransformationMode.SmoothTransformation,
                            )
                            pixmap = self._composite_pixmaps(pixmap, hotel_scaled)

        skip_actor_overlay = (
            self._current_game in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2)
            and terminated
            and is_actor_cell
            and cell_value.strip().upper() == "H"
        )

        if is_actor_cell and not skip_actor_overlay:
            actor_asset = self._get_actor_asset(taxi_state, payload)
            actor_pixmap = self._asset_manager.get_pixmap(actor_asset)
            if actor_pixmap is not None:
                self._logger.warning(f"TAXI ACTOR LOADED at ({row},{col}): {actor_asset}")
                pixmap = self._composite_pixmaps(pixmap, actor_pixmap)
            else:
                self._logger.error(f"TAXI ACTOR ASSET MISSING at ({row},{col}): {actor_asset} not found")

        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self._tile_size,
                self._tile_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

        return pixmap

    def _get_tile_asset(self, cell_value: str, row: int, col: int) -> str:
        if self._current_game in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2):
            return FrozenLakeAssets.get_tile_asset(cell_value)
        if self._current_game == GameId.TAXI:
            return TaxiAssets.get_tile_asset(cell_value, row, col, self._current_grid)
        if self._current_game == GameId.CLIFF_WALKING:
            return CliffWalkingAssets.get_tile_asset(cell_value, row, col)
        return "ice.png"

    def _get_actor_asset(
        self,
        taxi_state: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> str:
        if self._current_game in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2):
            return self._frozen_lake_actor_asset(payload)
        if self._current_game == GameId.TAXI:
            return self._taxi_actor_asset(taxi_state)
        if self._current_game == GameId.CLIFF_WALKING:
            return self._cliff_walking_actor_asset(payload)
        return "elf_down.png"

    def _frozen_lake_actor_asset(self, payload: Dict[str, Any] | None) -> str:
        default_direction = "down"
        action = self._safe_int((payload or {}).get("last_action"))
        direction_lookup = {
            0: "left",
            1: "down",
            2: "right",
            3: "up",
        }
        direction = (
            direction_lookup.get(action, default_direction)
            if action is not None
            else default_direction
        )
        return FrozenLakeAssets.get_actor_asset(direction)

    def _taxi_actor_asset(self, taxi_state: Dict[str, Any] | None) -> str:
        action = None
        if taxi_state is not None:
            action = self._safe_int(taxi_state.get("last_action"))
        direction_map = {
            0: "front",
            1: "rear",
            2: "right",
            3: "left",
        }
        direction = direction_map.get(action, "front") if action is not None else "front"
        return TaxiAssets.get_cab_asset(direction)

    def _cliff_walking_actor_asset(self, payload: Dict[str, Any] | None) -> str:
        action = self._safe_int((payload or {}).get("last_action"))
        direction_map = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        }
        direction = direction_map.get(action, "down") if action is not None else "down"
        return CliffWalkingAssets.get_actor_asset(direction)

    @staticmethod
    def _safe_int(value: Any | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_taxi_depot_position(depot_index: int) -> tuple[int, int]:
        depot_positions = {
            0: (1, 1),
            1: (1, 9),
            2: (5, 1),
            3: (5, 7),
        }
        return depot_positions.get(depot_index, (1, 1))

    def _create_depot_letter_pixmap(self, letter: str) -> QtGui.QPixmap | None:
        color_map = {
            'R': QtGui.QColor(255, 0, 0),
            'G': QtGui.QColor(0, 255, 0),
            'Y': QtGui.QColor(255, 255, 0),
            'B': QtGui.QColor(0, 0, 255),
        }

        color = color_map.get(letter)
        if not color:
            return None

        pixmap = QtGui.QPixmap(self._tile_size, self._tile_size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(0, 0, self._tile_size, self._tile_size, color)
        painter.end()
        return pixmap

    @staticmethod
    def _composite_pixmaps(base: QtGui.QPixmap, overlay: QtGui.QPixmap) -> QtGui.QPixmap:
        result = QtGui.QPixmap(base.size())
        result.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(result)
        painter.drawPixmap(0, 0, base)
        x_offset = (base.width() - overlay.width()) // 2
        y_offset = (base.height() - overlay.height()) // 2
        painter.drawPixmap(x_offset, y_offset, overlay)
        painter.end()

        return result

    def clear_cache(self) -> None:
        self._asset_manager.clear_cache()


__all__ = ["GridRendererStrategy"]