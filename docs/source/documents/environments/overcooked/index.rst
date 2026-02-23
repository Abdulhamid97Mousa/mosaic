Overcooked
==========

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/overcooked.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

Cooperative cooking game for studying human-AI coordination.

:Install: ``pip install -e 3rd_party/overcooked_ai/``
:Paradigm: Multi-agent (simultaneous, 2 agents)
:Stepping: ``SIMULTANEOUS``
:Note: **Python 3.10 only** (requires >=3.10,<3.11)

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Layout
     - Description
   * - cramped_room
     - Tight kitchen, close-quarters coordination
   * - asymmetric_advantages
     - Asymmetric access to ingredients
   * - coordination_ring
     - Circular kitchen layout
   * - forced_coordination
     - Explicit coordination required for success
   * - counter_circuit
     - Circuit-style counter layout

Citation
--------

.. code-block:: bibtex

   @inproceedings{carroll2019overcooked,
     author       = {Micah Carroll and Rohin Shah and Mark K. Ho and Tom Griffiths and Sanjit Seshia and Pieter Abbeel and Anca Dragan},
     title        = {On the Utility of Learning about Humans for Human-AI Coordination},
     booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
     year         = {2019},
   }
