NetHack
=======

The full NetHack roguelike game via NLE.

:Install: ``pip install -e ".[nethack]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - NetHackChallenge-v0
     - Full game with all mechanics
   * - NetHackScore-v0
     - Maximise score
   * - NetHackStaircase-v0
     - Reach the next dungeon level
   * - NetHackStaircasePet-v0
     - Reach stairs with your pet alive
   * - NetHackOracle-v0
     - Find the Oracle of Delphi
   * - NetHackGold-v0
     - Collect as much gold as possible
   * - NetHackEat-v0
     - Eat a food item
   * - NetHackScout-v0
     - Explore as many dungeon tiles as possible

Citation
--------

.. code-block:: bibtex

   @inproceedings{kuettler2020nethack,
     author       = {Heinrich K{\"u}ttler and Nantas Nardelli and Alexander H. Miller and Roberta Raileanu and Marco Selvatici and Edward Grefenstette and Tim Rockt{\"a}schel},
     title        = {The NetHack Learning Environment},
     booktitle    = {Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)},
     year         = {2020},
   }
