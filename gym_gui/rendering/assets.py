"""Asset management for Gymnasium toy_text environment images."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from qtpy import QtGui

# Use local asset directory for version control and independence
def _get_local_img_path() -> Path:
    """Get the local toy_text img directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "toy_text_images"

_GYMNASIUM_IMG_PATH = _get_local_img_path()


class AssetManager:
    """Manages loading and caching of Gymnasium environment image assets."""

    def __init__(self) -> None:
        """Initialize the asset manager with an empty cache."""
        self._cache: Dict[str, QtGui.QPixmap] = {}
        self._img_dir = _GYMNASIUM_IMG_PATH

    def get_pixmap(self, asset_name: str) -> Optional[QtGui.QPixmap]:
        """
        Load and return a QPixmap for the given asset name.

        Args:
            asset_name: The asset filename (e.g., "ice.png", "elf_up.png")

        Returns:
            QPixmap if the asset exists, None otherwise
        """
        if asset_name in self._cache:
            return self._cache[asset_name]

        asset_path = self._img_dir / asset_name
        if not asset_path.exists():
            return None

        pixmap = QtGui.QPixmap(str(asset_path))
        if pixmap.isNull():
            return None

        self._cache[asset_name] = pixmap
        return pixmap

    def clear_cache(self) -> None:
        """Clear the pixmap cache to free memory."""
        self._cache.clear()


_ASSET_MANAGER_SINGLETON: AssetManager | None = None


def get_asset_manager() -> AssetManager:
    """Return the process-wide asset manager instance."""

    global _ASSET_MANAGER_SINGLETON
    if _ASSET_MANAGER_SINGLETON is None:
        _ASSET_MANAGER_SINGLETON = AssetManager()
    return _ASSET_MANAGER_SINGLETON


class FrozenLakeAssets:
    """Asset mappings for FrozenLake environment."""

    # Tile types
    ICE = "ice.png"
    HOLE = "hole.png"
    CRACKED_HOLE = "cracked_hole.png"
    GOAL = "goal.png"
    STOOL = "stool.png"

    # Agent sprites (elf)
    AGENT_UP = "elf_up.png"
    AGENT_DOWN = "elf_down.png"
    AGENT_LEFT = "elf_left.png"
    AGENT_RIGHT = "elf_right.png"

    @staticmethod
    def get_tile_asset(cell_value: str) -> str:
        """
        Map grid cell character to tile asset name.

        Args:
            cell_value: Grid cell character ('S', 'F', 'H', 'G')

        Returns:
            Asset filename for the tile
        """
        mapping = {
            "S": FrozenLakeAssets.ICE,  # Start position
            "F": FrozenLakeAssets.ICE,  # Frozen surface
            "H": FrozenLakeAssets.HOLE,  # Hole
            "G": FrozenLakeAssets.GOAL,  # Goal
        }
        return mapping.get(cell_value.strip().upper(), FrozenLakeAssets.ICE)

    @staticmethod
    def get_agent_asset(direction: str = "down") -> str:
        """
        Get agent sprite based on movement direction.

        Args:
            direction: Movement direction ('up', 'down', 'left', 'right')

        Returns:
            Asset filename for the agent sprite
        """
        mapping = {
            "up": FrozenLakeAssets.AGENT_UP,
            "down": FrozenLakeAssets.AGENT_DOWN,
            "left": FrozenLakeAssets.AGENT_LEFT,
            "right": FrozenLakeAssets.AGENT_RIGHT,
        }
        return mapping.get(direction.lower(), FrozenLakeAssets.AGENT_DOWN)


class TaxiAssets:
    """Asset mappings for Taxi environment."""

    # Taxi sprites
    CAB_FRONT = "cab_front.png"
    CAB_REAR = "cab_rear.png"
    CAB_LEFT = "cab_left.png"
    CAB_RIGHT = "cab_right.png"

    # Background
    BACKGROUND = "taxi_background.png"
    
    # Passenger
    PASSENGER = "passenger.png"
    
    # Hotel (destination marker)
    HOTEL = "hotel.png"
    
    # Gridworld medians (walls/grass borders)
    MEDIAN_LEFT = "gridworld_median_left.png"
    MEDIAN_RIGHT = "gridworld_median_right.png"
    MEDIAN_TOP = "gridworld_median_top.png"
    MEDIAN_BOTTOM = "gridworld_median_bottom.png"
    MEDIAN_HORIZ = "gridworld_median_horiz.png"
    MEDIAN_VERT = "gridworld_median_vert.png"

    @staticmethod
    def get_cab_asset(direction: str = "front") -> str:
        """
        Get taxi cab sprite based on direction.

        Args:
            direction: Cab orientation ('front', 'rear', 'left', 'right')

        Returns:
            Asset filename for the cab sprite
        """
        mapping = {
            "front": TaxiAssets.CAB_FRONT,
            "rear": TaxiAssets.CAB_REAR,
            "left": TaxiAssets.CAB_LEFT,
            "right": TaxiAssets.CAB_RIGHT,
            "down": TaxiAssets.CAB_FRONT,
            "up": TaxiAssets.CAB_REAR,
        }
        return mapping.get(direction.lower(), TaxiAssets.CAB_FRONT)
    
    @staticmethod
    def get_tile_asset(cell_value: str, row: int = 0, col: int = 0, grid: list | None = None) -> str:
        """
        Get the tile asset for a Taxi grid cell with context-aware edge detection.
        
        Taxi ANSI grid (7×11) structure:
        Row 0: +---------+  (top border)
        Rows 1-5: |..data..|  (data rows with vertical walls)
        Row 6: +---------+  (bottom border)
        
        Segments need edge caps:
        - Vertical '|': top cap, middle pieces, bottom cap
        - Horizontal '-': left cap, middle pieces, right cap
        
        Args:
            cell_value: The character in the grid
            row: Row index in grid
            col: Column index in grid
            grid: Full grid for context (list of lists)
            
        Returns:
            Asset filename for the tile
        """
        # Vertical wall/border '|'
        if cell_value == '|':
            if grid and row < len(grid):
                above = grid[row - 1][col] if row > 0 else ''
                below = grid[row + 1][col] if row < len(grid) - 1 else ''
                
                # Top cap: '+' or '-' above (border connection)
                if above in ('+', '-'):
                    return TaxiAssets.MEDIAN_TOP
                # Bottom cap: '+' or '-' below (border connection)
                elif below in ('+', '-'):
                    return TaxiAssets.MEDIAN_BOTTOM
                # For interior segments, check if this is a 2-piece wall
                # If there's a non-wall above and below, this shouldn't happen
                # If there's a wall above AND below, use middle piece
                elif above == '|' and below == '|':
                    return TaxiAssets.MEDIAN_VERT
                # If wall above but not below (bottom of interior segment)
                elif above == '|' and below != '|':
                    return TaxiAssets.MEDIAN_BOTTOM
                # If wall below but not above (top of interior segment)
                elif above != '|' and below == '|':
                    return TaxiAssets.MEDIAN_TOP
            # Default to middle piece
            return TaxiAssets.MEDIAN_VERT
        
        # Horizontal border '-'
        elif cell_value == '-':
            if grid and row < len(grid):
                left = grid[row][col - 1] if col > 0 else ''
                right = grid[row][col + 1] if col < len(grid[row]) - 1 else ''
                
                # Left cap: '+' on left (first '-' after corner)
                if left == '+':
                    return TaxiAssets.MEDIAN_LEFT
                # Right cap: '+' on right (last '-' before corner)
                elif right == '+':
                    return TaxiAssets.MEDIAN_RIGHT
            # Middle piece
            return TaxiAssets.MEDIAN_HORIZ
        
        # Corners use background
        elif cell_value == '+':
            return TaxiAssets.BACKGROUND
        
        # All other characters (depot letters, colons, spaces) use background
        # Depot overlays (hotel.png) are added by GridRenderer separately
        return TaxiAssets.BACKGROUND


class CliffWalkingAssets:
    """Asset mappings for CliffWalking environment."""

    # Mountain-specific assets
    MOUNTAIN_BG1 = "mountain_bg1.png"
    MOUNTAIN_BG2 = "mountain_bg2.png"
    MOUNTAIN_CLIFF = "mountain_cliff.png"
    MOUNTAIN_NEAR_CLIFF1 = "mountain_near-cliff1.png"
    MOUNTAIN_NEAR_CLIFF2 = "mountain_near-cliff2.png"
    STOOL = "stool.png"  # Start position for CliffWalking
    COOKIE = "cookie.png"  # Goal tile for CliffWalking

    # Agent sprites (reuse elf from FrozenLake)
    AGENT_UP = FrozenLakeAssets.AGENT_UP
    AGENT_DOWN = FrozenLakeAssets.AGENT_DOWN
    AGENT_LEFT = FrozenLakeAssets.AGENT_LEFT
    AGENT_RIGHT = FrozenLakeAssets.AGENT_RIGHT

    @staticmethod
    def get_tile_asset(cell_value: str, row: int = 0, col: int = 0) -> str:
        """
        Map grid cell character to tile asset with position-based variation.

        Args:
            cell_value: Grid cell character ('o', 'C', 'x', 'T')
            row: Row position in the grid (0-indexed)
            col: Column position in the grid (0-indexed)

        Returns:
            Asset filename for the tile
        """
        # CliffWalking grid is 4x12:
        # Row 0-2: safe ground with visual variety
        # Row 3 (bottom): start(x), cliffs(C), goal(T)
        cell = cell_value.strip().upper()
        
        if cell == "C":
            return CliffWalkingAssets.MOUNTAIN_CLIFF
        elif cell == "T":
            return CliffWalkingAssets.COOKIE  # Goal position uses cookie
        elif cell == "X":
            # Start position - use stool
            return CliffWalkingAssets.STOOL
        elif cell == "O" or cell == " " or not cell:
            # Safe ground - add variation based on position
            if row == 2:  # Row just above the cliff/goal row
                # Use near-cliff variants for visual interest
                return CliffWalkingAssets.MOUNTAIN_NEAR_CLIFF1 if col % 2 == 0 else CliffWalkingAssets.MOUNTAIN_NEAR_CLIFF2
            else:
                # Alternate between bg1 and bg2 for visual variety
                return CliffWalkingAssets.MOUNTAIN_BG1 if (row + col) % 2 == 0 else CliffWalkingAssets.MOUNTAIN_BG2
        
        return CliffWalkingAssets.MOUNTAIN_BG1

    @staticmethod
    def get_agent_asset(direction: str = "down") -> str:
        """
        Get agent sprite based on movement direction.

        Args:
            direction: Movement direction ('up', 'down', 'left', 'right')

        Returns:
            Asset filename for the agent sprite
        """
        mapping = {
            "up": CliffWalkingAssets.AGENT_UP,
            "down": CliffWalkingAssets.AGENT_DOWN,
            "left": CliffWalkingAssets.AGENT_LEFT,
            "right": CliffWalkingAssets.AGENT_RIGHT,
        }
        return mapping.get(direction.lower(), CliffWalkingAssets.AGENT_DOWN)


__all__ = [
    "AssetManager",
    "get_asset_manager",
    "FrozenLakeAssets",
    "TaxiAssets",
    "CliffWalkingAssets",
]