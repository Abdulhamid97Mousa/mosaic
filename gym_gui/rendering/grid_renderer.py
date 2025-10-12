"""Grid renderer that uses AssetManager to display Gymnasium toy-text environments."""

from __future__ import annotations

from typing import Any, Dict, List

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.rendering.assets import (
    CliffWalkingAssets,
    FrozenLakeAssets,
    TaxiAssets,
    get_asset_manager,
)


class GridRenderer:
    """Renders toy-text environment grids using image assets with QGraphicsView for responsive scaling."""

    def __init__(self, graphics_view: QtWidgets.QGraphicsView) -> None:
        """
        Initialize the grid renderer.

        Args:
            graphics_view: The QGraphicsView to render into
        """
        self._view = graphics_view
        self._scene = QtWidgets.QGraphicsScene()
        self._view.setScene(self._scene)
        
        # Configure view for responsive rendering
        self._view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._asset_manager = get_asset_manager()
        self._current_game: GameId | None = None
        self._current_grid: List[List[str]] = []  # Store for context-aware rendering
        self._tile_size = 48  # Base tile size in scene coordinates
        self._last_actor_position: tuple[int, int] | None = None

    def render(
        self,
        grid: List[List[str]],
        game_id: GameId,
        agent_position: tuple[int, int] | None = None,
        taxi_state: Dict[str, Any] | None = None,
        terminated: bool = False,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """
        Render a grid with assets using Graphics View for responsive scaling.

        Args:
            grid: 2D list of characters
            game_id: Which game to render for
            agent_position: (row, col) of the agent
            taxi_state: Taxi-specific state dict (taxi_position, passenger_index, destination_index)
            terminated: Whether the episode terminated (for FrozenLake cracked_hole)
            payload: General payload dict containing game-specific data (last_action, etc.)
        """
        self._current_game = game_id
        self._current_grid = grid  # Store for context-aware asset selection
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # Clear existing scene
        self._scene.clear()

        # Render each cell as a QGraphicsPixmapItem
        effective_actor_position = agent_position
        if terminated and effective_actor_position is None and self._last_actor_position is not None:
            effective_actor_position = self._last_actor_position

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
                    # Create pixmap item at scene coordinates
                    item = self._scene.addPixmap(pixmap)
                    if item is not None:
                        item.setPos(c * self._tile_size, r * self._tile_size)
                    
        # Set scene rect to match grid dimensions
        self._scene.setSceneRect(0, 0, cols * self._tile_size, rows * self._tile_size)
        
        # Fit the scene in the view (responsive scaling)
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
        """Create a pixmap for a single grid cell with appropriate assets composited."""
        is_actor_cell = actor_position is not None and actor_position == (row, col)

        # LAYER 1: Base background (for Taxi game, every cell gets taxi_background.png)
        if self._current_game == GameId.TAXI:
            pixmap = self._asset_manager.get_pixmap("taxi_background.png")
            if pixmap is None:
                return None
        elif self._current_game == GameId.FROZEN_LAKE:
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
            # Build layered pixmap for CliffWalking tiles (background + overlay)
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
            # For other games, get the tile-specific asset
            asset_name = self._get_tile_asset(cell_value, row, col)
            pixmap = self._asset_manager.get_pixmap(asset_name)
            if pixmap is None:
                return None

        # LAYER 2: Walls, borders, and structural elements (for Taxi game only)
        if self._current_game == GameId.TAXI and cell_value != ':' and cell_value.strip():
            # Skip depot letters - they'll be handled in layer 3
            if cell_value not in ('R', 'G', 'Y', 'B'):
                structural_asset = self._get_tile_asset(cell_value, row, col)
                structural_pixmap = self._asset_manager.get_pixmap(structural_asset)
                if structural_pixmap:
                    pixmap = self._composite_pixmaps(pixmap, structural_pixmap)

        # LAYER 3: Depot letters and passenger/destination overlays for Taxi game
        if self._current_game == GameId.TAXI:
            # Render depot letters (R, G, Y, B) with colored squares
            if cell_value in ('R', 'G', 'Y', 'B'):
                depot_pixmap = self._create_depot_letter_pixmap(cell_value)
                if depot_pixmap:
                    pixmap = self._composite_pixmaps(pixmap, depot_pixmap)
            
            # Passenger and destination overlays (only when taxi_state is provided)
            if taxi_state:
                # Passenger location overlay (if passenger is at a depot, not in taxi)
                pass_idx = taxi_state.get("passenger_index", -1)
                if pass_idx < 4:  # 0-3 = at depot R/G/Y/B
                    pass_pos = self._get_taxi_depot_position(pass_idx)
                    if pass_pos == (row, col):
                        passenger_pixmap = self._asset_manager.get_pixmap("passenger.png")
                        if passenger_pixmap:
                            pixmap = self._composite_pixmaps(pixmap, passenger_pixmap)
                
                # Destination marker overlay - scaled down to show colored background
                dest_idx = taxi_state.get("destination_index", -1)
                if dest_idx < 4:
                    dest_pos = self._get_taxi_depot_position(dest_idx)
                    if dest_pos == (row, col):
                        hotel_pixmap = self._asset_manager.get_pixmap("hotel.png")
                        if hotel_pixmap:
                            # Scale hotel icon to 65% of tile size to show colored border
                            scaled_size = int(self._tile_size * 0.65)
                            hotel_scaled = hotel_pixmap.scaled(
                                scaled_size, scaled_size,
                                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                QtCore.Qt.TransformationMode.SmoothTransformation
                            )
                            pixmap = self._composite_pixmaps(pixmap, hotel_scaled)

        # LAYER 4: Agent overlay if present
        skip_actor_overlay = (
            self._current_game == GameId.FROZEN_LAKE
            and terminated
            and is_actor_cell
            and cell_value.strip().upper() == "H"
        )

        if is_actor_cell and not skip_actor_overlay:
            actor_asset = self._get_actor_asset(taxi_state, payload)
            actor_pixmap = self._asset_manager.get_pixmap(actor_asset)
            if actor_pixmap is not None:
                pixmap = self._composite_pixmaps(pixmap, actor_pixmap)
        
        # Scale to tile size
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self._tile_size, self._tile_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )

        return pixmap

    def _get_tile_asset(self, cell_value: str, row: int, col: int) -> str:
        """Get the appropriate tile asset name for a cell."""
        if self._current_game == GameId.FROZEN_LAKE:
            return FrozenLakeAssets.get_tile_asset(cell_value)
        elif self._current_game == GameId.TAXI:
            return TaxiAssets.get_tile_asset(cell_value, row, col, self._current_grid)
        elif self._current_game == GameId.CLIFF_WALKING:
            return CliffWalkingAssets.get_tile_asset(cell_value, row, col)
        else:
            # Fallback
            return "ice.png"

    def _get_actor_asset(self, taxi_state: Dict[str, Any] | None = None, payload: Dict[str, Any] | None = None) -> str:
        """Get the actor sprite asset name, with optional state for directional rendering."""
        if self._current_game == GameId.FROZEN_LAKE:
            direction = "down"
            if payload and payload.get("last_action") is not None:
                action = int(payload["last_action"])
                # FrozenLake actions: 0=Left, 1=Down, 2=Right, 3=Up
                direction_lookup = {
                    0: "left",
                    1: "down",
                    2: "right",
                    3: "up",
                }
                direction = direction_lookup.get(action, direction)
            return FrozenLakeAssets.get_actor_asset(direction)
        elif self._current_game == GameId.TAXI:
            # Select cab direction based on last action
            direction = "front"  # default
            if taxi_state and "last_action" in taxi_state:
                action = taxi_state["last_action"]
                # Taxi actions: 0=SOUTH (down), 1=NORTH (up), 2=EAST (right), 3=WEST (left), 4=PICKUP, 5=DROPOFF
                if action == 0:  # SOUTH
                    direction = "front"
                elif action == 1:  # NORTH
                    direction = "rear"
                elif action == 2:  # EAST
                    direction = "right"
                elif action == 3:  # WEST
                    direction = "left"
                # For PICKUP/DROPOFF, keep current direction (default to front)
            return TaxiAssets.get_cab_asset(direction)
        elif self._current_game == GameId.CLIFF_WALKING:
            # Select elf direction based on last action
            direction = "down"  # default
            if payload and "last_action" in payload:
                action = payload["last_action"]
                # CliffWalking actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
                if action == 0:  # UP
                    direction = "up"
                elif action == 1:  # RIGHT
                    direction = "right"
                elif action == 2:  # DOWN
                    direction = "down"
                elif action == 3:  # LEFT
                    direction = "left"
            return CliffWalkingAssets.get_actor_asset(direction)
        else:
            return "elf_down.png"

    @staticmethod
    def _get_taxi_depot_position(depot_index: int) -> tuple[int, int]:
        """
        Get grid position for Taxi depot by index in 11×11 grid coordinates.
        
        Args:
            depot_index: 0=Red, 1=Green, 2=Yellow, 3=Blue
            
        Returns:
            (row, col) position in 11×11 grid
        """
        # Depot positions in 11×11 character grid:
        # R: [1, 1]
        # G: [1, 9]
        # Y: [5, 1]
        # B: [5, 7]
        depot_positions = {
            0: (1, 1),  # Red
            1: (1, 9),  # Green
            2: (5, 1),  # Yellow
            3: (5, 7),  # Blue
        }
        return depot_positions.get(depot_index, (1, 1))

    def _create_depot_letter_pixmap(self, letter: str) -> QtGui.QPixmap | None:
        """
        Create a pixmap with a colored square background for depot (R, G, Y, B).
        
        Args:
            letter: The depot letter ('R', 'G', 'Y', or 'B')
            
        Returns:
            QPixmap with colored square background, or None if letter not recognized
        """
        # Color mapping for depot backgrounds
        color_map = {
            'R': QtGui.QColor(255, 0, 0),      # Red
            'G': QtGui.QColor(0, 255, 0),      # Green  
            'Y': QtGui.QColor(255, 255, 0),    # Yellow
            'B': QtGui.QColor(0, 0, 255),      # Blue
        }
        
        color = color_map.get(letter)
        if not color:
            return None
        
        # Create pixmap with transparent background
        pixmap = QtGui.QPixmap(self._tile_size, self._tile_size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Fill entire tile with depot color (solid square)
        painter.fillRect(0, 0, self._tile_size, self._tile_size, color)
        
        painter.end()
        return pixmap

    @staticmethod
    def _composite_pixmaps(base: QtGui.QPixmap, overlay: QtGui.QPixmap) -> QtGui.QPixmap:
        """Composite overlay onto base pixmap."""
        result = QtGui.QPixmap(base.size())
        result.fill(QtCore.Qt.GlobalColor.transparent)
        
        painter = QtGui.QPainter(result)
        painter.drawPixmap(0, 0, base)
        # Center overlay
        x_offset = (base.width() - overlay.width()) // 2
        y_offset = (base.height() - overlay.height()) // 2
        painter.drawPixmap(x_offset, y_offset, overlay)
        painter.end()
        
        return result

    def clear_cache(self) -> None:
        """Clear the asset cache to free memory."""
        self._asset_manager.clear_cache()


__all__ = ["GridRenderer"]
