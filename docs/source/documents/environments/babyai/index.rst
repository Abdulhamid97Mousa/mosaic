BabyAI
======

Language-grounded instruction following built on MiniGrid.

:Install: ``pip install -e ".[minigrid]"`` (or ``pip install -e ".[minigrid-balrog]"`` for BALROG tasks)
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Sub-family
     - Environments
   * - GoTo
     - GoToRedBall, GoToObj, GoToLocal, GoTo, GoToImpUnlock, GoToSeq, GoToDoor, GoToObjDoor
   * - Open
     - Open, OpenRedDoor, OpenDoor, OpenTwoDoors, OpenDoorsOrderN2/N4
   * - Pickup
     - Pickup, UnblockPickup, PickupLoc, PickupDist, PickupAbove
   * - Unlock
     - Unlock, UnlockLocal, KeyInBox, UnlockPickup, BlockedUnlockPickup, UnlockToUnlock
   * - PutNext
     - PutNextLocal, PutNext
   * - Complex
     - ActionObjDoor, FindObjS5, KeyCorridorS3R1–R3, OneRoomS8, Synth, BossLevel

Citation
--------

.. code-block:: bibtex

   @article{chevalier2018babyai,
     author       = {Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Salem Lahlou and Lucas Willems and Chitwan Saharia and Thien Huu Nguyen and Yoshua Bengio},
     title        = {BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning},
     journal      = {arXiv preprint arXiv:1810.08272},
     year         = {2018},
   }
