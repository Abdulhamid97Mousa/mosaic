Toy Text
========

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/toy_text.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

Simple tabular environments useful for learning and debugging.

:Install: ``pip install -e ".[gymnasium]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``
:Docs: `gymnasium.farama.org/environments/toy_text/ <https://gymnasium.farama.org/environments/toy_text/>`_

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - FrozenLake-v1 / v2
     - Navigate a frozen lake grid without falling into holes
   * - CliffWalking-v0
     - Walk along a cliff edge to reach the goal
   * - Taxi-v3
     - Pick up and drop off passengers in a grid city
   * - Blackjack-v1
     - Classic card game against a dealer
