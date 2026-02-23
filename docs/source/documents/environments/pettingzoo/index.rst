PettingZoo
==========

`PettingZoo <https://pettingzoo.farama.org/>`_ is the standard API for
multi-agent reinforcement learning, maintained by the
`Farama Foundation <https://farama.org/>`_.  PettingZoo provides two
stepping paradigms:

- **AEC** (Alternating Environment Cycle) — agents take turns sequentially
- **Parallel** — all agents act simultaneously

MOSAIC currently supports environments from the **Classic** category — turn-based
board games using the AEC API.

.. code-block:: bash

   pip install -e ".[pettingzoo]"

.. toctree::
   :maxdepth: 1

   classic
