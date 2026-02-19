CleanRL Worker Errors
=====================

Errors specific to the **CleanRL worker** (``mosaic-cleanrl``).  For a full
reference of CleanRL runtime errors (shape mismatches, FastLane issues,
curriculum failures, etc.) see also
:doc:`/documents/architecture/workers/integrated_workers/CleanRL_Worker/common_errors`.


Installation Errors
-------------------

``error: package directory 'cleanrl/cleanrl' does not exist``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running** ``pip install -e 3rd_party/cleanrl_worker`` **):**

.. code-block:: text

   running egg_info
   writing mosaic_cleanrl.egg-info/PKG-INFO
   ...
   error: package directory 'cleanrl/cleanrl' does not exist

**Cause:** The CleanRL worker package (``mosaic-cleanrl``) bundles both the
MOSAIC harness (``cleanrl_worker/``) and the upstream CleanRL library source
code (``cleanrl/cleanrl/``, ``cleanrl/cleanrl_utils/``).  The upstream source
lives in a **git submodule** at ``3rd_party/cleanrl_worker/cleanrl/``.

If the submodule was never initialised, or if the directory is empty, pip
cannot find the mapped package directories and the build fails.

**Fix -- initialise the submodule and install:**

.. code-block:: bash

   # 1. Initialise the CleanRL submodule
   cd /home/zahra/projects_hamid/GUI_BDI_RL
   git submodule update --init 3rd_party/cleanrl_worker/cleanrl

   # If the above fails with "did not match any file(s) known to git",
   # clone the submodule manually:
   git clone --depth 1 https://github.com/vwxyzjn/cleanrl.git \
       3rd_party/cleanrl_worker/cleanrl

   # 2. Install the worker in editable mode
   source .venv/bin/activate
   pip install -e 3rd_party/cleanrl_worker

   # 3. Verify
   python -c "import cleanrl_worker; print('cleanrl_worker OK')"
   python -c "import cleanrl; print('cleanrl OK')"

.. note::

   The main project's ``pyproject.toml`` also exposes the CleanRL worker
   packages via ``[tool.setuptools.packages.find]``.  When you run
   ``pip install -e ".[cleanrl]"`` from the project root, it installs the
   **Python dependencies** (torch, tensorboard, wandb, tyro, etc.) but does
   **not** initialise the git submodule.  You must do both steps:

   .. code-block:: bash

      # Step A: Python dependencies
      pip install -e ".[cleanrl]"

      # Step B: CleanRL source submodule
      git submodule update --init 3rd_party/cleanrl_worker/cleanrl
      pip install -e 3rd_party/cleanrl_worker

``ModuleNotFoundError: No module named 'cleanrl'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (at runtime when the worker starts):**

.. code-block:: text

   ModuleNotFoundError: No module named 'cleanrl'

**Cause:** Same root cause as above.  The ``cleanrl`` Python package is
provided by the git submodule at ``3rd_party/cleanrl_worker/cleanrl/``, not
by PyPI.  The worker harness (``cleanrl_worker``) was installed but the
upstream library was not.

**Fix:** Follow the submodule initialisation steps above.

``ModuleNotFoundError: No module named 'tyro'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ModuleNotFoundError: No module named 'tyro'

**Cause:** The CleanRL Python dependencies were not installed.  ``tyro`` is
the CLI argument parser used by upstream CleanRL scripts.

**Fix:**

.. code-block:: bash

   pip install -e ".[cleanrl]"

This installs ``tyro``, ``torch``, ``tensorboard``, ``wandb``, ``tenacity``,
and ``moviepy`` in one step.

-----

Runtime Warnings
----------------

``optional_deps`` shows ``"cleanrl_worker": false``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** The MOSAIC settings log at startup shows:

.. code-block:: json

   "optional_deps": {
     "cleanrl_worker": false
   }

**Cause:** MOSAIC uses ``importlib.util.find_spec("cleanrl_worker")`` at
startup.  If the package is not installed (or the submodule is missing),
the worker appears as unavailable.  The GUI will still launch, but CleanRL
training forms will not work.

**Fix:** Install both the submodule and the harness as described above.
After restarting MOSAIC, the setting should show ``"cleanrl_worker": true``.
