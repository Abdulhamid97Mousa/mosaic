Procgen
=======

16 procedurally generated environments for testing generalisation.

:Install: ``pip install -e ".[procgen]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``
:Note: Python 3.10 uses ``procgen``; Python 3.11+ uses ``procgen-mirror``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Environment
     - Environment
   * - BigFish
     - BossFight
   * - CaveFlyer
     - Chaser
   * - Climber
     - CoinRun
   * - DodgeBall
     - FruitBot
   * - Heist
     - Jumper
   * - Leaper
     - Maze
   * - Miner
     - Ninja
   * - Plunder
     - StarPilot

Citation
--------

.. code-block:: bibtex

   @inproceedings{cobbe2020procgen,
     author       = {Karl Cobbe and Christopher Hesse and Jacob Hilton and John Schulman},
     title        = {Leveraging Procedural Generation to Benchmark Reinforcement Learning},
     booktitle    = {Proceedings of the International Conference on Machine Learning (ICML)},
     year         = {2020},
   }
