MiniGrid
========

Procedurally generated grid-world environments for navigation and reasoning.

:Install: ``pip install -e ".[minigrid]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``
:Docs: `minigrid.farama.org <https://minigrid.farama.org/>`_

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/MiniGrid/MINIGRID_1.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

MOSAIC supports **35 MiniGrid environments** across 9 task categories:

Empty
-----

Navigate to a goal in an empty room.  The simplest MiniGrid tasks — good for
verifying your setup.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Size
   * - MiniGrid-Empty-5x5-v0
     - 5×5
   * - MiniGrid-Empty-Random-5x5-v0
     - 5×5 (random start)
   * - MiniGrid-Empty-6x6-v0
     - 6×6
   * - MiniGrid-Empty-Random-6x6-v0
     - 6×6 (random start)
   * - MiniGrid-Empty-8x8-v0
     - 8×8
   * - MiniGrid-Empty-16x16-v0
     - 16×16

DoorKey
-------

Find a key, unlock a door, and navigate to the goal.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Size
   * - MiniGrid-DoorKey-5x5-v0
     - 5×5
   * - MiniGrid-DoorKey-6x6-v0
     - 6×6
   * - MiniGrid-DoorKey-8x8-v0
     - 8×8
   * - MiniGrid-DoorKey-16x16-v0
     - 16×16

LavaGap
-------

Cross a gap in a lava field to reach the goal.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Gap Size
   * - MiniGrid-LavaGapS5-v0
     - S5
   * - MiniGrid-LavaGapS6-v0
     - S6
   * - MiniGrid-LavaGapS7-v0
     - S7

Dynamic Obstacles
-----------------

Navigate around randomly moving obstacles.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Size
   * - MiniGrid-Dynamic-Obstacles-5x5-v0
     - 5×5
   * - MiniGrid-Dynamic-Obstacles-Random-5x5-v0
     - 5×5 (random start)
   * - MiniGrid-Dynamic-Obstacles-6x6-v0
     - 6×6
   * - MiniGrid-Dynamic-Obstacles-Random-6x6-v0
     - 6×6 (random start)
   * - MiniGrid-Dynamic-Obstacles-8x8-v0
     - 8×8
   * - MiniGrid-Dynamic-Obstacles-16x16-v0
     - 16×16

MultiRoom
---------

Navigate through multiple connected rooms to reach the goal.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Rooms
   * - MiniGrid-MultiRoom-N2-S4-v0
     - 2 rooms, size 4
   * - MiniGrid-MultiRoom-N4-S5-v0
     - 4 rooms, size 5
   * - MiniGrid-MultiRoom-N6-v0
     - 6 rooms

Obstructed Maze
---------------

Find hidden keys in obstructed mazes with blocked paths.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Variant
   * - MiniGrid-ObstructedMaze-1Dlhb-v1
     - 1D with locked/hidden/blocked
   * - MiniGrid-ObstructedMaze-Full-v1
     - Full obstructed maze

Crossing
--------

Cross walls or lava rivers to reach the goal.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Type
   * - MiniGrid-LavaCrossingS9N1-v0
     - Lava, 9×9, 1 crossing
   * - MiniGrid-LavaCrossingS9N2-v0
     - Lava, 9×9, 2 crossings
   * - MiniGrid-LavaCrossingS9N3-v0
     - Lava, 9×9, 3 crossings
   * - MiniGrid-LavaCrossingS11N5-v0
     - Lava, 11×11, 5 crossings
   * - MiniGrid-SimpleCrossingS9N1-v0
     - Wall, 9×9, 1 crossing
   * - MiniGrid-SimpleCrossingS9N2-v0
     - Wall, 9×9, 2 crossings
   * - MiniGrid-SimpleCrossingS9N3-v0
     - Wall, 9×9, 3 crossings
   * - MiniGrid-SimpleCrossingS11N5-v0
     - Wall, 11×11, 5 crossings

RedBlueDoors
------------

Open doors in the correct colour order.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Size
   * - MiniGrid-RedBlueDoors-6x6-v0
     - 6×6
   * - MiniGrid-RedBlueDoors-8x8-v0
     - 8×8

Other
-----

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Description
   * - MiniGrid-BlockedUnlockPickup-v0
     - Unblock a path, unlock a door, pick up an object

Citation
--------

.. code-block:: bibtex

   @article{MinigridMiniworld23,
     author       = {Maxime Chevalier-Boisvert and Bolun Dai and Mark Towers and Rodrigo de Lazcano and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
     title        = {Minigrid \& Miniworld: Modular \& Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
     journal      = {CoRR},
     volume       = {abs/2306.13831},
     year         = {2023},
   }
