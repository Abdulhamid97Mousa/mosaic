ViZDoom
=======

Doom-based first-person visual RL environments.

:Install: ``pip install -e ".[vizdoom]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - Basic
     - Shoot a single monster in a room
   * - Deadly Corridor
     - Navigate a corridor full of enemies
   * - Defend the Center / Line
     - Survive waves of approaching enemies
   * - Health Gathering (+ Supreme)
     - Collect health packs on toxic floor
   * - My Way Home
     - Navigate a maze to find a vest
   * - Predict Position
     - Shoot a moving target
   * - Take Cover
     - Dodge fireballs behind pillars
   * - Deathmatch
     - Full deathmatch scenario

Citation
--------

.. code-block:: bibtex

   @inproceedings{wydmuch2019vizdoom,
     author       = {Marek Wydmuch and Micha{\l} Kempka and Wojciech Ja{\'s}kowski},
     title        = {ViZDoom Competitions: Playing Doom from Pixels},
     booktitle    = {IEEE Transactions on Games},
     volume       = {11},
     number       = {3},
     pages        = {248--259},
     year         = {2019},
   }
