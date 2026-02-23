"""Jumanji game documentation module.

Jumanji is a suite of JAX-based reinforcement learning environments for
combinatorial optimization and logic puzzles. It leverages JAX for
hardware acceleration and JIT compilation.

Phase 1 - Logic Environments:
    - Game2048-v1: Classic 2048 sliding tile game
    - Minesweeper-v0: Classic Minesweeper puzzle
    - RubiksCube-v0: 3x3 Rubik's Cube puzzle
    - SlidingTilePuzzle-v0: N-Puzzle sliding tile puzzle
    - Sudoku-v0: Classic Sudoku puzzle
    - GraphColoring-v1: Graph coloring combinatorial optimization

Phase 2 - Packing Environments:
    - BinPack-v2: Pack items into minimum bins
    - FlatPack-v0: 2D rectangular bin packing
    - JobShop-v0: Job shop scheduling
    - Knapsack-v1: 0/1 Knapsack optimization
    - Tetris-v0: Classic Tetris game

Phase 3 - Routing Environments:
    - Cleaner-v0: Grid cleaning task
    - Connector-v3: Path connection puzzle
    - CVRP-v1: Capacitated Vehicle Routing
    - Maze-v0: Maze navigation
    - MMST-v0: Minimum Spanning Tree
    - MultiCVRP-v0: Multi-vehicle routing
    - PacMan-v1: Classic Pac-Man game
    - RobotWarehouse-v0: Warehouse robot coordination
    - Snake-v1: Classic Snake game
    - Sokoban-v0: Box pushing puzzle
    - TSP-v1: Traveling Salesman Problem

Reference:
    https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations

# Phase 1: Logic
from .Game2048 import GAME2048_HTML, get_game2048_html
from .Minesweeper import MINESWEEPER_HTML, get_minesweeper_html
from .RubiksCube import RUBIKS_CUBE_HTML, get_rubiks_cube_html
from .SlidingPuzzle import SLIDING_PUZZLE_HTML, get_sliding_puzzle_html
from .Sudoku import SUDOKU_HTML, get_sudoku_html
from .GraphColoring import GRAPH_COLORING_HTML, get_graph_coloring_html

# Phase 2: Packing
from .BinPack import BINPACK_HTML, get_binpack_html
from .FlatPack import FLATPACK_HTML, get_flatpack_html
from .JobShop import JOBSHOP_HTML, get_jobshop_html
from .Knapsack import KNAPSACK_HTML, get_knapsack_html
from .Tetris import TETRIS_HTML, get_tetris_html

# Phase 3: Routing
from .Cleaner import CLEANER_HTML, get_cleaner_html
from .Connector import CONNECTOR_HTML, get_connector_html
from .CVRP import CVRP_HTML, get_cvrp_html
from .Maze import MAZE_HTML, get_maze_html
from .MMST import MMST_HTML, get_mmst_html
from .MultiCVRP import MULTI_CVRP_HTML, get_multi_cvrp_html
from .PacMan import PACMAN_HTML, get_pacman_html
from .RobotWarehouse import ROBOT_WAREHOUSE_HTML, get_robot_warehouse_html
from .Snake import SNAKE_HTML, get_snake_html
from .Sokoban import SOKOBAN_HTML, get_sokoban_html
from .TSP import TSP_HTML, get_tsp_html

__all__ = [
    # Phase 1: Logic
    "GAME2048_HTML",
    "get_game2048_html",
    "MINESWEEPER_HTML",
    "get_minesweeper_html",
    "RUBIKS_CUBE_HTML",
    "get_rubiks_cube_html",
    "SLIDING_PUZZLE_HTML",
    "get_sliding_puzzle_html",
    "SUDOKU_HTML",
    "get_sudoku_html",
    "GRAPH_COLORING_HTML",
    "get_graph_coloring_html",
    # Phase 2: Packing
    "BINPACK_HTML",
    "get_binpack_html",
    "FLATPACK_HTML",
    "get_flatpack_html",
    "JOBSHOP_HTML",
    "get_jobshop_html",
    "KNAPSACK_HTML",
    "get_knapsack_html",
    "TETRIS_HTML",
    "get_tetris_html",
    # Phase 3: Routing
    "CLEANER_HTML",
    "get_cleaner_html",
    "CONNECTOR_HTML",
    "get_connector_html",
    "CVRP_HTML",
    "get_cvrp_html",
    "MAZE_HTML",
    "get_maze_html",
    "MMST_HTML",
    "get_mmst_html",
    "MULTI_CVRP_HTML",
    "get_multi_cvrp_html",
    "PACMAN_HTML",
    "get_pacman_html",
    "ROBOT_WAREHOUSE_HTML",
    "get_robot_warehouse_html",
    "SNAKE_HTML",
    "get_snake_html",
    "SOKOBAN_HTML",
    "get_sokoban_html",
    "TSP_HTML",
    "get_tsp_html",
]
