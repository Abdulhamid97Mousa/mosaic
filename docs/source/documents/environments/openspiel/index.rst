OpenSpiel
=========

Board games via Google DeepMind's OpenSpiel + Shimmy PettingZoo wrapper.
Includes custom draughts/checkers variants with proper international rules.

:Install: ``pip install -e ".[openspiel]"``
:Paradigm: Multi-agent (turn-based)
:Stepping: ``SEQUENTIAL``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - open_spiel/checkers
     - Standard checkers via OpenSpiel
   * - draughts/american_checkers
     - American Checkers (8×8, no backward captures)
   * - draughts/russian_checkers
     - Russian Checkers (8×8, backward captures, flying kings)
   * - draughts/international_draughts
     - International Draughts (10×10, 20 pieces, flying kings)

Citation
--------

.. code-block:: bibtex

   @article{lanctot2019openspiel,
     author       = {Marc Lanctot and Edward Lockhart and Jean-Baptiste Lespiau and Vinicius Zambaldi and Satyaki Upadhyay and Julien P{\'e}rolat and Sriram Srinivasan and Finbarr Timbers and Karl Tuyls and Shayegan Omidshafiei and Daniel Hennes and Dustin Morrill and Paul Muller and Timo Ewalds and Ryan Faulkner and J{\'a}nos Kram{\'a}r and Bart De Vylder and Brennan Saeta and James Bradbury and David Ding and Sebastian Borgeaud and Matthew Lai and Julian Schrittwieser and Thomas Anthony and Edward Hughes and Ivo Danihelka and Jonah Ryan-Davis},
     title        = {OpenSpiel: A Framework for Reinforcement Learning in Games},
     journal      = {CoRR},
     volume       = {abs/1908.09453},
     year         = {2019},
   }
