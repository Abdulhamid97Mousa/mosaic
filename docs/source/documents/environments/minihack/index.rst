MiniHack
========

Sandbox RL environments built on the NetHack Learning Environment (NLE).

:Install: ``pip install -e ".[nethack]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``
:System: Requires ``build-essential``, ``cmake``, ``flex``, ``bison``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Environments
   * - Navigation
     - Room-5×5/15×15, Corridor-R2/R3/R5, MazeWalk-9×9/15×15/45×19, River, River-Narrow
   * - Skills
     - Eat, Wear, Wield, Zap, Read, Quaff, PutOn, LavaCross, WoD-Easy/Medium/Hard
   * - Exploration
     - ExploreMaze-Easy/Hard, HideNSeek, Memento-F2/F4

Citation
--------

.. code-block:: bibtex

   @inproceedings{samvelyan2021minihack,
     author       = {Mikayel Samvelyan and Robert Kirk and Vitaly Kurin and Jack Parker-Holder and Minqi Jiang and Eric Hambro and Fabio Petroni and Heinrich K{\"u}ttler and Edward Grefenstette and Tim Rockt{\"a}schel},
     title        = {MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research},
     booktitle    = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
     year         = {2021},
   }
