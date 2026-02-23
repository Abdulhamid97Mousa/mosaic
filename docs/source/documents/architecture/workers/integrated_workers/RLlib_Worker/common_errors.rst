Common Errors
=============

Ray already initialised
------------------------

**Symptom:** ``RuntimeError: Maybe you called ray.init twice by accident?``

**Cause:** Ray was already running in the same process.

**Fix:** The MOSAIC shim calls ``ray.shutdown()`` before ``ray.init()``.
If launching manually, call ``ray.shutdown()`` first.

Out of shared memory
---------------------

**Symptom:** ``ray.exceptions.OutOfMemoryError`` or worker crashes with
``/dev/shm`` full.

**Cause:** Ray uses shared memory for object storage.  Default limit is
30 % of RAM.

**Fix:** Reduce ``num_workers`` in the resource config, or increase
``/dev/shm`` size: ``--shm-size=4g`` in Docker, or
``mount -o remount,size=4g /dev/shm`` on Linux.

PettingZoo API type mismatch
------------------------------

**Symptom:** ``AttributeError: 'ParallelEnv' object has no attribute 'last'``

**Cause:** Choosing ``PARALLEL`` API type for an AEC-only environment
(e.g. Chess, Go).

**Fix:** Set ``api_type`` to ``AEC`` for turn-based board games.
