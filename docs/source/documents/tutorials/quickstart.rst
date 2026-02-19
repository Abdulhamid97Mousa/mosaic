Quickstart
==========

This guide will get you running your first MOSAIC experiment in minutes.

Prerequisites
-------------

Before launching, make sure:

1. You have completed the :doc:`Installation <installation/index>` guide
2. Your virtual environment is activated:

   .. code-block:: bash

      source .venv/bin/activate

3. The ``.env`` file exists in the project root (copy from the example
   if needed):

   .. code-block:: bash

      cp .env.example .env

   The ``.env`` file controls Qt rendering, GPU settings, environment
   defaults, API keys, and worker-specific configuration.  See the
   comments inside ``.env.example`` for details.

Launching MOSAIC
----------------

MOSAIC uses a **two-process architecture**: a Trainer Daemon (gRPC
backend) and the Qt6 GUI (frontend).  The launch script handles both:

.. code-block:: bash

   ./run.sh

What ``run.sh`` does:

1. Loads environment variables from ``.env``
2. Activates the virtual environment (if ``.venv/`` exists)
3. Creates ``var/logs/`` and ``var/trainer/`` directories
4. Kills any stale trainer daemon processes
5. Starts the **Trainer Daemon** in the background
   (``gym_gui.services.trainer_daemon`` on ``127.0.0.1:50055``)
6. Waits for the daemon to be ready (gRPC health check, up to 10 retries)
7. Launches the **MOSAIC GUI** (``gym_gui.app``)

On exit, the script automatically shuts down the daemon.

.. tip::

   To use a specific Python version:

   .. code-block:: bash

      PYTHON_BIN=python3.11 ./run.sh

Running a Single-Agent Environment
-----------------------------------

1. Select the **Single Agent** tab
2. Choose an environment family and environment, for example:
   **Gymnasium** > **Classic Control** > **CartPole-v1**
3. Click **Load Environment**
4. Select **Human Keyboard** as the actor
5. Click **Play** and use arrow keys to control the cart

Running a Multi-Agent Environment
----------------------------------

1. Select the **Multi Agent** tab
2. Choose an environment, for example:
   **PettingZoo** > **Classic** > **Chess**
3. Click **Load Environment**
4. Configure agents:

   - Player 0: Human Keyboard
   - Player 1: Random Policy

5. Click **Play**

Project Structure
-----------------

After launching, MOSAIC creates runtime directories under ``var/``:

.. code-block:: text

   var/
   ├── logs/
   │   └── trainer_daemon.log    # Daemon stdout/stderr
   ├── trainer/
   │   └── trainer.pid           # Daemon PID file
   └── operators/
       └── telemetry/            # Per-run JSONL telemetry

Logs are the first place to look if something goes wrong.

Troubleshooting
---------------

**Daemon fails to start**
   Check ``var/logs/trainer_daemon.log`` for errors.  Common causes:

   - Port 50055 already in use (another daemon running)
   - Missing ``.env`` file or incorrect ``QT_API`` setting
   - gRPC/protobuf version mismatch

**GUI shows a blank window**
   On WSL, make sure these are set in ``.env``:

   .. code-block:: text

      QT_QPA_PLATFORM=xcb
      QT_QUICK_BACKEND=software
      LIBGL_ALWAYS_SOFTWARE=1

**"No module named gym_gui"**
   Make sure MOSAIC is installed in development mode:

   .. code-block:: bash

      pip install -e .

Next Steps
----------

- :doc:`../architecture/workers/index` -- how workers train agents
- :doc:`../architecture/operators/index` -- how operators evaluate agents
