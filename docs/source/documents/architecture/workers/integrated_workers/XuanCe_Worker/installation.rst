Installation
============

.. tabs::

   .. tab:: PyTorch (recommended)

      .. code-block:: bash

         pip install -e ".[xuance]"

   .. tab:: PyTorch + TensorFlow

      .. code-block:: bash

         pip install -e ".[xuance,tensorflow]"

   .. tab:: MindSpore

      .. code-block:: bash

         pip install -e ".[xuance,mindspore]"

Verify the installation:

.. code-block:: bash

   python -c "import xuance_worker; print(xuance_worker.__version__)"

.. note::

   MPI support (for SMAC on HPC clusters) requires ``mpi4py`` and an
   MPI runtime (OpenMPI or MPICH).  Set ``MPI4PY_RC_INITIALIZE=0``
   (done automatically by the worker shim) to prevent ``MPI_Init()``
   blocking on non-MPI launches.
