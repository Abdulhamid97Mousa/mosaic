Atari / ALE
============

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/atari_ale.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

128 classic Atari 2600 games via the Arcade Learning Environment.

:Install: ``pip install -e ".[atari]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``
:Note: Requires ROM license acceptance (``autorom --accept-license``)

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - ALE/Breakout-v5
     - Break bricks with a bouncing ball
   * - ALE/Pong-v5
     - Classic 2-player table tennis
   * - ALE/SpaceInvaders-v5
     - Defend Earth from descending aliens
   * - ALE/MontezumaRevenge-v5
     - Exploration-heavy platformer (sparse reward)
   * - ALE/MsPacman-v5
     - Navigate mazes and eat pellets
   * - *... and 123 more*
     - Full 128-game ALE library supported

Citation
--------

.. code-block:: bibtex

   @article{bellemare2013ale,
     author       = {Marc G. Bellemare and Yavar Naddaf and Joel Veness and Michael Bowling},
     title        = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
     journal      = {Journal of Artificial Intelligence Research},
     volume       = {47},
     pages        = {253--279},
     year         = {2013},
   }
