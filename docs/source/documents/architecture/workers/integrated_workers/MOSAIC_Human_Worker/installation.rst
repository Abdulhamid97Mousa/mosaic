Installation
============

The MOSAIC Human Worker is installed as an editable Python package from the
project's ``3rd_party/mosaic/human_worker`` directory.

.. code-block:: bash

   pip install -e 3rd_party/mosaic/human_worker

The worker itself has **zero external dependencies** (pure Python). Environment
packages must be installed separately.

Verify Installation
-------------------

.. code-block:: bash

   # Check CLI is available
   human-worker --help

   # Quick test with MiniGrid (requires minigrid package)
   human-worker --mode interactive --run-id verify \
       --env-name minigrid --task MiniGrid-Empty-5x5-v0 --seed 42

Environment Dependencies
------------------------

Install environment packages depending on which environments you plan to
use with the human worker:

.. code-block:: bash

   # MiniGrid / BabyAI
   pip install -e ".[minigrid]"

   # MosaicMultiGrid (Soccer, Basketball, Collect)
   pip install -e 3rd_party/mosaic_multigrid

   # Crafter
   pip install -e ".[crafter]"

   # PettingZoo (Chess, Go, Connect Four)
   pip install -e ".[pettingzoo]"

Running Tests
-------------

.. code-block:: bash

   # Run all 35 tests
   python -m pytest 3rd_party/mosaic/human_worker/tests/ -v

   # Run only config tests (no env dependencies)
   python -m pytest 3rd_party/mosaic/human_worker/tests/ -v -k "Config"
