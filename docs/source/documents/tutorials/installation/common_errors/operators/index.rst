Operator-Specific Errors
========================

These errors are specific to **MOSAIC operators** (the Operators tab in the
GUI). For shared errors that apply to all platforms, see the parent
:doc:`../index` page.

-----

mosaic_multigrid
----------------

Preview hangs -- "Loading environment preview" with no result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:**

The user selects a ``mosaic_multigrid`` environment (e.g.,
``MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0``) and clicks **Load Environment**.
The log shows:

.. code-block:: text

   INFO | Loading environment preview for mosaic_multigrid/MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0

but the preview never appears. No error message is displayed in the status
bar. The GUI remains responsive (daemon polling continues) but the render
panel stays empty.

**Cause:** The ``mosaic-multigrid`` PyPI package (v4.0.0+) registers **13
environments** via ``gym.register()`` in ``mosaic_multigrid.envs``. Each
registered ID maps to a specific environment class:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Gym ID
     - Class
   * - ``MosaicMultiGrid-Soccer-v0``
     - ``SoccerGame4HEnv10x15N2``
   * - ``MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0``
     - ``SoccerGame4HIndAgObsEnv16x11N2``
   * - ``MosaicMultiGrid-Soccer-2vs2-TeamObs-v0``
     - ``SoccerTeamObsEnv``
   * - ``MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0``
     - ``BasketballGame6HIndAgObsEnv19x11N3``

If the preview code instantiates a hardcoded class (e.g.,
``SoccerGame4HEnv10x15N2``) instead of using ``gym.make(task)``, the wrong
environment is created. This can cause silent failures, missing observations,
or hanging because the environment's internal state does not match the
expected gym ID.

**Fix:** The preview code in ``main_window.py`` must use
``gymnasium.make()`` for all mosaic_multigrid environments instead of
hardcoded class imports:

.. code-block:: python

   import gymnasium
   import mosaic_multigrid.envs  # triggers gymnasium.register() calls

   env = gymnasium.make(task)    # creates the correct class for any registered ID
   env.render_mode = 'rgb_array'
   env.reset()

This was fixed in the ``mosaic_multigrid``/``ini_multigrid`` family split
(February 2026).

.. note::

   Both ``mosaic_multigrid`` (v4.4.0+) and ``ini_multigrid`` use the modern
   **Gymnasium** API (``import gymnasium``), not the deprecated OpenAI Gym.
   Always use ``import gymnasium`` when working with these environments.

``render_mode`` property has no setter (OrderEnforcing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   Cannot preview mosaic_multigrid MosaicMultiGrid-Collect-IndAgObs-v0:
   property 'render_mode' of 'OrderEnforcing' object has no setter

**Cause:** Gymnasium wraps every environment created via
``gymnasium.make()`` in an ``OrderEnforcing`` wrapper. This wrapper
exposes ``render_mode`` as a **read-only property** -- it cannot be
assigned after creation.

Code like this will fail:

.. code-block:: python

   # WRONG -- raises AttributeError
   env = gymnasium.make(task)
   env.render_mode = 'rgb_array'

**Fix:** Pass ``render_mode`` as a keyword argument to
``gymnasium.make()`` at creation time:

.. code-block:: python

   # CORRECT -- render_mode set during construction
   env = gymnasium.make(task, render_mode='rgb_array')

The same rule applies when instantiating environment classes directly
(without ``gymnasium.make``):

.. code-block:: python

   # CORRECT -- pass render_mode in kwargs
   env = env_cls(**config_kwargs, render_mode='rgb_array')

   # WRONG -- setting after construction
   env = env_cls(**config_kwargs)
   env.render_mode = 'rgb_array'   # may fail depending on the class

.. note::

   This applies to **all** Gymnasium environments, not just multigrid.
   The old OpenAI Gym allowed post-construction assignment of
   ``render_mode``, but Gymnasium does not.

mosaic_multigrid not installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (preview):**

.. code-block:: text

   mosaic_multigrid not installed - cannot preview: No module named 'mosaic_multigrid'

**Error (agent detection):**

.. code-block:: text

   Failed to auto-detect agent count for mosaic_multigrid/...: gymnasium package not installed

**Cause:** The ``mosaic-multigrid`` package is not installed in the virtual
environment.

**Fix -- Option A (recommended): Install via pyproject.toml optional dependency:**

.. code-block:: bash

   source .venv/bin/activate
   cd /home/zahra/projects_hamid/GUI_BDI_RL
   pip install ".[mosaic_multigrid]"

This installs ``mosaic-multigrid==4.4.0`` plus all base dependencies
defined in ``pyproject.toml``.

**Fix -- Option B: Install via requirements file:**

.. code-block:: bash

   source .venv/bin/activate
   pip install -r requirements/mosaic_multigrid.txt

**Fix -- Option C: Install the package directly:**

.. code-block:: bash

   source .venv/bin/activate
   pip install mosaic-multigrid==4.4.0

**Verify the installation:**

.. code-block:: bash

   python -c "
   import mosaic_multigrid
   print(f'mosaic-multigrid version: {mosaic_multigrid.__version__}')

   import gymnasium
   import mosaic_multigrid.envs
   env = gymnasium.make('MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0')
   env.reset()
   print(f'Environment: {env.unwrapped.__class__.__name__}, agents={len(env.unwrapped.agents)}')
   env.close()
   "

Expected output:

.. code-block:: text

   mosaic-multigrid version: 4.4.0
   Environment: SoccerGame4HIndAgObsEnv16x11N2, agents=4

-----

ini_multigrid
-------------

ini_multigrid not available -- cannot preview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ini_multigrid not available - cannot preview: No module named 'multigrid'

**Cause:** The INI multigrid package (``3rd_party/multigrid-ini``) is not on
the Python path or the submodule was not initialized.

**Fix -- Option A (recommended): Install via pyproject.toml:**

.. code-block:: bash

   source .venv/bin/activate
   cd /home/zahra/projects_hamid/GUI_BDI_RL
   pip install ".[multigrid_ini]"

**Fix -- Option B: Install from local directory:**

.. code-block:: bash

   source .venv/bin/activate

   # Initialize the submodule if the directory is empty
   git submodule update --init 3rd_party/multigrid-ini

   # Install in editable mode
   pip install -e 3rd_party/multigrid-ini

-----

Environment Family Reference
-----------------------------

MOSAIC splits multigrid environments into two independent families:

.. list-table::
   :widths: 20 30 25 25
   :header-rows: 1

   * - Family
     - Description
     - view_size
     - Environments
   * - ``mosaic_multigrid``
     - Competitive team sports
     - 3
     - 13 (Soccer, Collect, Basketball variants)
   * - ``ini_multigrid``
     - Cooperative exploration
     - 7
     - 13 (Empty, RedBlueDoors, LockedHallway, etc.)

Both families appear as separate entries in the Operators tab **Env Family**
dropdown. Both show the multigrid settings panel (observation mode,
coordination strategy). Role assignment (forward/defender) is only available
for ``mosaic_multigrid`` Soccer environments.
