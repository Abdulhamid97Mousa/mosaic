Crafter
=======

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/crafter.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

Open-world survival benchmark testing a wide spectrum of agent capabilities.

:Install: ``pip install -e ".[crafter]"``
:Paradigm: Single-agent
:Stepping: ``SINGLE_AGENT``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - CrafterReward-v1
     - Full game with reward signals
   * - CrafterNoReward-v1
     - Reward-free variant for unsupervised learning

Citation
--------

.. code-block:: bibtex

   @inproceedings{hafner2022crafter,
     author       = {Danijar Hafner},
     title        = {Benchmarking the Spectrum of Agent Capabilities},
     booktitle    = {International Conference on Learning Representations (ICLR)},
     year         = {2022},
   }
