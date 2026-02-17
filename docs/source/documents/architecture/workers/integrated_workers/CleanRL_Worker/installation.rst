Installation
============

Prerequisites
-------------

- **Python 3.11** (the project-wide minimum; earlier versions are not
  supported)
- **pip** with editable-install support (``pip >= 21.3``)
- **CUDA toolkit** (optional) -- required only for GPU-accelerated
  training.  CPU training works out of the box.

Installing the CleanRL Extras
-----------------------------

The CleanRL worker and its core dependencies are installed via the
``cleanrl`` optional-dependency group defined in the project root
``pyproject.toml``:

.. code-block:: bash

   pip install -e ".[cleanrl]"

This installs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``torch >= 2.0.0``
     - Neural network framework (PyTorch)
   * - ``tensorboard >= 2.11.0``
     - Training metric logging and visualization
   * - ``wandb >= 0.22.3``
     - Weights & Biases experiment tracking (optional at runtime)
   * - ``tyro >= 0.5.0``
     - CLI argument parsing used by upstream CleanRL scripts
   * - ``tenacity >= 8.0.0``
     - Retry logic for transient failures (e.g. W&B uploads)
   * - ``moviepy >= 1.0.3``
     - Video recording for ``capture_video`` mode

Environment-Specific Extras
----------------------------

Depending on which environments you plan to train in, install the
corresponding extras:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Extra
     - Command
     - Packages
   * - MiniGrid
     - ``pip install -e ".[minigrid]"``
     - ``gymnasium >= 1.1.0``, ``minigrid >= 2.0.0, < 3.0.0``
   * - Atari
     - ``pip install -e ".[atari]"``
     - ``gymnasium[atari] >= 1.1.0``, ``ale-py == 0.10.1``,
       ``autorom[accept-rom-license] >= 0.6.0``
   * - MuJoCo
     - ``pip install -e ".[mujoco]"``
     - ``gymnasium[mujoco] >= 1.1.0``
   * - Procgen
     - ``pip install -e ".[procgen]"``
     - ``procgen >= 0.10.7`` (or ``procgen-mirror`` on Python >= 3.11)

You can combine extras in a single install:

.. code-block:: bash

   pip install -e ".[cleanrl,minigrid,atari]"

Or use the ``full`` convenience bundle to install everything:

.. code-block:: bash

   pip install -e ".[full]"

Syllabus-RL Setup (Curriculum Learning)
---------------------------------------

Curriculum training requires `Syllabus-RL <https://github.com/RyanNavillus/Syllabus>`_,
which is vendored as a Git submodule under ``3rd_party/Syllabus/``.

.. code-block:: bash

   # Initialize the submodule (first time only)
   git submodule update --init 3rd_party/Syllabus

   # Install Syllabus in editable mode
   pip install -e 3rd_party/Syllabus

After installation, the following imports should succeed:

.. code-block:: python

   from syllabus.core import ReinitTaskWrapper, GymnasiumSyncWrapper
   from syllabus.curricula import SequentialCurriculum
   from syllabus.task_space import DiscreteTaskSpace

Syllabus-RL is only required for curriculum training mode.  Standard
training, policy evaluation, and resume training work without it.

Verifying the Installation
--------------------------

**Import check** -- confirm the worker package is importable:

.. code-block:: bash

   python -c "from cleanrl_worker import get_worker_metadata; print(get_worker_metadata())"

This should print the ``WorkerCapabilities`` descriptor without errors.

**Dry run** -- validate that a config resolves correctly without
launching training:

.. code-block:: bash

   python -m cleanrl_worker.cli --config path/to/config.json --dry-run

A successful dry run prints the resolved module name and exits with
code 0.

**Entry point discovery** -- verify MOSAIC can discover the worker:

.. code-block:: bash

   python -c "
   from importlib.metadata import entry_points
   eps = entry_points(group='mosaic.workers')
   print([ep.name for ep in eps])
   "

The output should include ``'cleanrl'``.

GUI Integration
---------------

The CleanRL worker is integrated into the MOSAIC GUI through Python
entry points.  Once installed, the worker appears automatically in the
GUI's training form selector.

**Entry point registration** (from ``pyproject.toml``):

.. code-block:: toml

   [project.entry-points."mosaic.workers"]
   cleanrl = "cleanrl_worker:get_worker_metadata"

The GUI discovers the worker via ``WorkerDiscovery``, which scans the
``mosaic.workers`` entry point group at startup.

**Form widgets** -- four dedicated dialogs are provided:

- **Train** (``cleanrl_train_form.py``) -- standard training with
  algorithm selection, hyperparameter tuning, and environment filtering.
- **Script** (``cleanrl_script_form.py``) -- launch custom shell scripts
  for multi-phase training workflows.
- **Resume** (``cleanrl_resume_form.py``) -- continue training from a
  saved ``.cleanrl_model`` checkpoint.
- **Policy Eval** (``cleanrl_policy_form.py``) -- load and evaluate
  a trained policy with configurable episode counts and rendering.

Configuration via .env
----------------------

Runtime settings such as Weights & Biases credentials and gRPC
verbosity are read from the ``.env`` file in the project root.
Copy ``.env.example`` as a starting point:

.. code-block:: bash

   cp .env.example .env

Key variables:

.. code-block:: bash

   # Weights & Biases
   WANDB_API_KEY=your_wandb_api_key_here
   WANDB_PROJECT_NAME=MOSAIC
   WANDB_ENTITY_NAME=your_wandb_username
   WANDB_EMAIL=your_email@example.com

   # Optional VPN proxy for W&B
   # WANDB_VPN_HTTPS_PROXY=https://127.0.0.1:7890
   # WANDB_VPN_HTTP_PROXY=http://127.0.0.1:7890

   # gRPC debug logging
   GRPC_VERBOSITY=debug

These values can also be overridden per-run through the GUI training
form.
