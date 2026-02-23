SMACv2
======

StarCraft Multi-Agent Challenge v2 — procedural unit generation per episode.

:Install: ``pip install -e ".[smacv2]"``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``
:System: Requires StarCraft II binary (set ``SC2PATH`` env var)

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Map
     - Description
   * - 10gen_terran
     - 10 units, random Terran composition each episode
   * - 10gen_protoss
     - 10 units, random Protoss composition each episode
   * - 10gen_zerg
     - 10 units, random Zerg composition each episode

Citation
--------

.. code-block:: bibtex

   @article{ellis2023smacv2,
     author       = {Benjamin Ellis and Jonathan Cook and Skander Moalla and Mikayel Samvelyan and Mingfei Sun and Anuj Mahajan and Jakob Foerster and Shimon Whiteson},
     title        = {SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning},
     journal      = {CoRR},
     volume       = {abs/2212.07489},
     year         = {2023},
   }
