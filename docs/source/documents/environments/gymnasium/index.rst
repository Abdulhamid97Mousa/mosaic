Gymnasium
=========

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
