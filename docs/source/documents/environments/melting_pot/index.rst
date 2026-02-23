Melting Pot
===========

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/meltingpot.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

Multi-agent social scenario benchmark from Google DeepMind.  Tests cooperation,
competition, deception, and trust with up to 16 agents.

:Install: ``pip install -e ".[meltingpot]"``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``
:Note: Linux/macOS only

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Substrates
   * - Public Goods
     - clean_up, commons_harvest (closed/open/partnership)
   * - Cooperation
     - collaborative_cooking (7 layouts), coop_mining, chemistry (4 variants), boat_race
   * - Coordination
     - bach_or_stravinsky, pure_coordination, rationalizable_coordination, stag_hunt
   * - Competition
     - paintball (capture_the_flag/king_of_the_hill), territory (3 variants)
   * - Social Dilemma
     - prisoners_dilemma, chicken, running_with_scissors (3 variants)
   * - Other
     - allelopathic_harvest, coins, daycare, externality_mushrooms, factory_commons,
       fruit_market, gift_refinements, hidden_agenda, predator_prey (4 variants)

Citation
--------

.. code-block:: bibtex

   @article{leibo2021meltingpot,
     author       = {Joel Z. Leibo and Edgar Du{\'e}{\~n}ez-Guzm{\'a}n and Alexander Sasha Vezhnevets and John P. Agapiou and Peter Sunehag and Raphael Koster and Jayd Matyas and Charles Beattie and Igor Mordatch and Thore Graepel},
     title        = {Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot},
     journal      = {CoRR},
     volume       = {abs/2107.06857},
     year         = {2021},
   }
