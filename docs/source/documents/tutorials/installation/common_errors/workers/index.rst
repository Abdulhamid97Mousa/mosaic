Worker-Specific Errors
======================

These errors are specific to running **MOSAIC workers** (training backends).
For shared errors that apply to all platforms, see the parent :doc:`../index`
page.

.. toctree::
   :maxdepth: 1
   :caption: Per-Worker Error Guides

   cleanrl_worker_errors/index
   xuance_worker_errors/index

-----

Shared Worker Errors
--------------------

The errors below can occur with any worker that uses the affected tool or
library.

``jq: command not found`` in Curriculum Training Scripts
--------------------------------------------------------

**Error:**

.. code-block:: text

   /home/zahra/.../scripts/curriculum_babyai_goto.sh: line 78: jq: command not found

**Cause:** The ``jq`` command-line JSON processor is not installed on the
system.  CleanRL curriculum training scripts (e.g.
``curriculum_babyai_goto.sh``) use ``jq`` to build the curriculum schedule
JSON from the base MOSAIC config file.  This is a system dependency, not a
Python package, so ``pip install`` will not provide it.

**Context:** This error occurs when you launch a **custom script** through
the CleanRL training form (``cleanrl_script_form.py``).  The script reads
``MOSAIC_CONFIG_FILE``, then pipes it through ``jq`` to inject curriculum
stages, algorithm parameters, and output directories into the JSON config.

**Fix:**

.. code-block:: bash

   # Ubuntu / Debian / WSL
   sudo apt-get update && sudo apt-get install -y jq

   # Verify installation
   jq --version

After installing ``jq``, re-run the training script from the GUI.  No
restart of the MOSAIC application is required -- the next subprocess launch
will pick up the newly installed binary.

.. note::

   If you are provisioning a fresh environment (CI, Docker, new WSL distro),
   add ``jq`` to your system-level dependency list alongside other non-Python
   prerequisites such as ``git``, ``curl``, and ``build-essential``.
