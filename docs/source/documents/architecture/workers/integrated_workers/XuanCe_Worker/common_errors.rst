Common Errors
=============

MPI hangs on startup
--------------------

**Symptom:** Worker process hangs indefinitely on startup.

**Cause:** ``mpi4py`` calls ``MPI_Init()`` at import time outside an
MPI launch environment.

**Fix:** Ensure the environment variable ``MPI4PY_RC_INITIALIZE=0`` is
set before spawning the worker.  The MOSAIC shim sets this automatically
via ``sitecustomize.py``.

Parameter sharing dimension mismatch
-------------------------------------

**Symptom:** ``RuntimeError: mat1 and mat2 shapes cannot be multiplied``
when loading a checkpoint trained on a different number of agents.

**Cause:** Training with ``use_parameter_sharing=True`` embeds
``n_agents`` in the first linear layer input dimension.

**Fix:** Train with ``use_parameter_sharing=False`` for checkpoints that
must generalise across agent counts, or retrain on the target agent count.

XuanCe output written to wrong directory
-----------------------------------------

**Symptom:** Logs, TensorBoard files, and checkpoints appear in the
current working directory instead of ``var/trainer/runs/``.

**Cause:** ``xuance_shims.py`` path redirection not applied.

**Fix:** Ensure the worker is launched via the MOSAIC shim
(``python -m xuance_worker.cli``), not directly via XuanCe's own entry
point.
