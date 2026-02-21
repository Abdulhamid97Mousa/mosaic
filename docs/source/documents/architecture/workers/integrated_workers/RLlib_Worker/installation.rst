Installation
============

.. tabs::

   .. tab:: CPU only

      .. code-block:: bash

         pip install -e ".[ray]"

   .. tab:: GPU (CUDA)

      .. code-block:: bash

         pip install -e ".[ray]"
         pip install torch --index-url https://download.pytorch.org/whl/cu121

Verify the installation:

.. code-block:: bash

   python -c "import ray; import ray_worker; print('Ray', ray.__version__)"

.. note::

   Ray initialises a local cluster on first use.  For multi-node
   distributed training, start a Ray head node first with
   ``ray start --head`` and point the worker at it via the
   ``RAY_ADDRESS`` environment variable.
