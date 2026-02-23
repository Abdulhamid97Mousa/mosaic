INI MultiGrid
=============

Cooperative exploration multi-agent environments with ``view_size=7``.

:Install: ``pip install -e 3rd_party/multigrid-ini/``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - MultiGrid-Empty (5×5 to 16×16)
     - Cooperative navigation in empty grids
   * - MultiGrid-BlockedUnlockPickup
     - Blocked door puzzle requiring cooperation
   * - MultiGrid-LockedHallway (2/4/6 Rooms)
     - Multi-room exploration with locked doors
   * - MultiGrid-Playground
     - Playground with various interactive objects
   * - MultiGrid-RedBlueDoors (6×6, 8×8)
     - Cooperative door-colour sequencing

Citation
--------

.. code-block:: bibtex

   @article{oguntola2023theory,
     title        = {Theory of mind as intrinsic motivation for multi-agent reinforcement learning},
     author       = {Ini Oguntola and Joseph Campbell and Simon Stepputtis and Katia Sycara},
     journal      = {arXiv preprint arXiv:2307.01158},
     year         = {2023}
   }
