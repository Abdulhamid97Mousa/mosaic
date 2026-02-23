Installation
============

The MOSAIC Random Worker is installed as an editable Python package from the
project's ``3rd_party/mosaic/random_worker`` directory.

.. code-block:: bash

   pip install -e 3rd_party/mosaic/random_worker

That's it, no API keys, no GPU, no optional extras required.

Verify Installation
-------------------

.. code-block:: bash

   # Check CLI is available
   random-worker --help

   # Quick test with MiniGrid
   random-worker --run-id verify-install \
       --task MiniGrid-Empty-8x8-v0 \
       --behavior random --seed 42

Dependencies
------------

- ``gymnasium >= 0.29.0``
- ``numpy >= 1.20.0``

Environment-specific packages (MiniGrid, MosaicMultiGrid, etc.) must be
installed separately depending on which environments you plan to use:

.. code-block:: bash

   # For MiniGrid / BabyAI
   pip install -e ".[minigrid]"

   # For MosaicMultiGrid (Soccer, Basketball, Collect)
   pip install -e 3rd_party/mosaic_multigrid

Running Tests
-------------

.. code-block:: bash

   # Run all 117 tests
   python -m pytest 3rd_party/mosaic/random_worker/tests/ -v

   # Skip slow subprocess tests
   python -m pytest 3rd_party/mosaic/random_worker/tests/ -v -k "not Subprocess"
