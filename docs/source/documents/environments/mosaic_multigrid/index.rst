MOSAIC MultiGrid
================

Competitive team-based multi-agent grid-world games.  Developed as part of
MOSAIC with ``view_size=3`` (agent-centric partial observability).

:Install: ``pip install -e ".[mosaic_multigrid]"``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``
:PyPI: `mosaic-multigrid v4.4.0 <https://pypi.org/project/mosaic-multigrid/>`_

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - Soccer-2vs2-IndAgObs
     - 2v2 soccer on a 16×11 FIFA grid, ball respawn, first-to-2-goals
   * - Soccer-1vs1-IndAgObs
     - 1v1 soccer variant
   * - Collect-IndAgObs
     - 3 agents, individual ball collection competition
   * - Collect-2vs2-IndAgObs
     - 2v2 teams, 7 balls (no draws)
   * - Collect-1vs1-IndAgObs
     - 1v1 variant, 3 balls
   * - Basketball-3vs3-IndAgObs
     - 6 agents, 3v3 basketball with court rendering
   * - *TeamObs variants*
     - Soccer-2vs2, Collect-2vs2, Basketball-3vs3 with SMAC-style teammate awareness
