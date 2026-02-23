Gymnasium
=========

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/toy_text.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

`Gymnasium <https://gymnasium.farama.org/>`_ is the standard API for
single-agent reinforcement learning, maintained by the
`Farama Foundation <https://farama.org/>`_.  MOSAIC wraps four Gymnasium
sub-families as separate optional extras — install only the ones you need.

.. code-block:: bash

   # Install everything Gymnasium in one go
   pip install -e ".[all-gymnasium]"   # = box2d + mujoco + atari + minigrid

.. toctree::
   :maxdepth: 1

   toy_text
   classic_control
   box2d
   mujoco
