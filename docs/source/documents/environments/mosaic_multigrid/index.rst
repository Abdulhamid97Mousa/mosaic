MOSAIC MultiGrid
================

Competitive team-based multi-agent grid-world games.  Developed as part of
MOSAIC with ``view_size=3`` (agent-centric partial observability).

:Install: ``pip install -e ".[mosaic_multigrid]"``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``
:PyPI: `mosaic-multigrid v6.0.0 <https://pypi.org/project/mosaic-multigrid/>`_

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


Citation
--------


.. code-block:: bibtex 
  
    @article{mousa2026mosaic,
      title = {MOSAIC MultiGrid: Research-Grade Multi-Agent Gridworld Environments},
      author = {Mousa, Abdulhamid},
      journal = {GitHub repository},
      year = {2026},
      url = {https://github.com/Abdulhamid97Mousa/mosaic_multigrid},
    }

    @misc{gym_multigrid,
      author = {Fickinger, Arnaud},
      title = {Multi-Agent Gridworld Environment for OpenAI Gym},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ArnaudFickinger/gym-multigrid}},
    }

    @article{oguntola2023theory,
      title = {Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning},
      author = {Oguntola, Ini and Campbell, Joseph and Stepputtis, Simon and Sycara, Katia},
      journal = {arXiv preprint arXiv:2307.01158},
      year = {2023},
      url = {https://github.com/ini/multigrid},
    }

    @misc{mosaic_multigrid,
      author = {Mousa, Abdulhamid},
      title = {mosaic\_multigrid: Research-Grade Multi-Agent Gridworld Environments},
      year = {2026},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Abdulhamid97Mousa/mosaic_multigrid}},
    }

    @article{terry2021pettingzoo,
      title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
      author = {Terry, J. K and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario
                and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens
                and Horsch, Caroline and Perez-Vicente, Rodrigo and Williams, Niall L
                and Lokesh, Yashas and Ravi, Praveen},
      journal = {Advances in Neural Information Processing Systems},
      volume = {34},
      pages = {2242--2254},
      year = {2021},
      url = {https://pettingzoo.farama.org/},
    }