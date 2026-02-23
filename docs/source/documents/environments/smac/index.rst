SMAC v1
=======

StarCraft Multi-Agent Challenge — hand-designed cooperative micromanagement maps.

:Install: ``pip install -e ".[smac]"``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``
:System: Requires StarCraft II binary (set ``SC2PATH`` env var)

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Map
     - Difficulty
     - Description
   * - 3m
     - Easy
     - 3 Marines vs 3 Marines (symmetric)
   * - 8m
     - Easy
     - 8 Marines vs 8 Marines (symmetric)
   * - 2s3z
     - Easy
     - 2 Stalkers + 3 Zealots vs same (mixed)
   * - 3s5z
     - Easy
     - 3 Stalkers + 5 Zealots vs same (mixed)
   * - 5m_vs_6m
     - Hard
     - 5 Marines vs 6 Marines (asymmetric)
   * - MMM2
     - Super Hard
     - 1 Medivac + 2 Marauders + 7 Marines (mixed)

Citation
--------

.. code-block:: bibtex

   @article{samvelyan19smac,
     author       = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philip H. S. Torr and Jakob Foerster and Shimon Whiteson},
     title        = {The StarCraft Multi-Agent Challenge},
     journal      = {CoRR},
     volume       = {abs/1902.04043},
     year         = {2019},
   }
