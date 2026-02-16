Jumanji
=======

JAX-based hardware-accelerated environments from Google DeepMind.  Organised
into logic puzzles, packing problems, and routing tasks.

:Install: ``pip install -e ".[jumanji]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``
:Note: Requires JAX with compatible hardware backend (CPU/GPU/TPU)

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Environments
   * - Logic Puzzles
     - Game2048, Minesweeper, RubiksCube, SlidingTilePuzzle, Sudoku, GraphColoring
   * - Packing
     - BinPack, FlatPack, JobShop, Knapsack, Tetris
   * - Routing
     - Cleaner, Connector, CVRP, Maze, MMST, MultiCVRP, PacMan, RobotWarehouse, Snake, Sokoban, TSP

Citation
--------

.. code-block:: bibtex

   @inproceedings{bonnet2024jumanji,
     author       = {Cl{\'e}ment Bonnet and Daniel Luo and Donal Byrne and Shikha Surana and Sa{\"i}d Chadly and Sasha Abramowitz and Victor Le and Paul Breuil and Thomas Barrett and Arnu Pretorius and Alexandre Laterre},
     title        = {Jumanji: a Diverse Suite of Scalable Reinforcement Learning Environments in JAX},
     booktitle    = {International Conference on Learning Representations (ICLR)},
     year         = {2024},
   }
