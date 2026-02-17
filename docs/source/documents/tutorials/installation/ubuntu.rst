Installation on Ubuntu (Native)
================================

This guide covers installing MOSAIC on a **native Ubuntu** system (the
primary supported platform). For WSL, see :doc:`wsl`.

Prerequisites
-------------

MOSAIC requires **Python 3.10, 3.11, or 3.12** and several system-level
build tools. Install everything at once:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y \
       python3.11 python3.11-venv python3.11-dev \
       build-essential \
       cmake \
       swig \
       flex \
       bison \
       libbz2-dev \
       libopenmpi-dev \
       libegl1 libgl1 libopengl0 libxkbcommon0 \
       libnss3 libnspr4 libasound2 \
       libxcomposite1 libxdamage1 libxkbfile1 \
       libxcb-cursor0 \
       fontconfig fonts-dejavu-extra fonts-liberation fonts-noto \
       stockfish \
       xvfb

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Required By
   * - ``python3.11``, ``python3.11-venv``, ``python3.11-dev``
     - Python interpreter and venv support
   * - ``build-essential``
     - C/C++ compiler for native extensions
   * - ``cmake``
     - NetHack Learning Environment (``nle``)
   * - ``swig``
     - Box2D environments (``box2d-py``)
   * - ``flex``, ``bison``
     - NetHack Learning Environment (``nle``)
   * - ``libbz2-dev``
     - NetHack Learning Environment (``nle``)
   * - ``libopenmpi-dev``
     - XuanCe worker (``mpi4py``)
   * - ``stockfish``
     - PettingZoo Chess environments
   * - ``xvfb``
     - Virtual display for headless servers (optional)

Setup
-----

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
   cd MOSAIC

   # Create virtual environment
   python3.11 -m venv .venv
   source .venv/bin/activate

   # Upgrade pip
   pip install --upgrade pip setuptools wheel

   # Install core GUI (minimal)
   pip install -e .

   # Install with optional environment families and workers
   pip install -e ".[minigrid,mosaic_multigrid,multigrid_ini,pettingzoo,mujoco,atari,vizdoom,crafter,nethack,overcooked,rware,cleanrl,xuance,chat]"

.. tip::

   You do not need to install every extra. Pick only what you need:

   .. code-block:: bash

      # Example: just CleanRL + Atari
      pip install -e ".[cleanrl,atari]"

   See :doc:`index` for the full list of extras.

Install Local 3rd-Party Packages
---------------------------------

Some environments install from local source in ``3rd_party/``:

.. code-block:: bash

   # MOSAIC MultiGrid (competitive team sports)
   pip install -e 3rd_party/mosaic_multigrid/

   # INI MultiGrid (cooperative exploration)
   pip install -e 3rd_party/multigrid-ini/

   # Overcooked-AI (cooperative cooking)
   pip install -e 3rd_party/overcooked_ai/

   # RWARE (warehouse delivery)
   pip install -e 3rd_party/robotic-warehouse/

Common Errors
-------------

If you encounter build failures or dependency conflicts during installation,
see the :doc:`common_errors` page for solutions to all known issues, including:

- ``swig`` not found (Box2D)
- ``nle`` build failure (NetHack)
- ``smac``/``smacv2`` protobuf version conflict
- ``mpi4py`` MPI compiler not found
- Broken venv / wrong Python version

Verify Installation
-------------------

.. code-block:: bash

   # Launch with trainer daemon
   ./run.sh

   # Or launch GUI only
   python -m gym_gui

If successful, the console will log detected optional dependencies and
system info:

.. code-block:: text

   bash ./run.sh
   Checking for existing trainer processes...
   Starting trainer daemon...
   Waiting for trainer daemon...
   Trainer daemon ready.
   Launching MOSAIC...
   [gym_gui] Loaded settings:
   {
     "qt_api": "PyQt6",
     "log_level": "DEBUG",
     "default_control_mode": "human_only",
     "default_seed": 1,
     "allow_seed_reuse": false,
     "ui": {
       "chat_panel_collapsed": true
     },
     "vllm": {
       "max_servers": 4,
       "gpu_memory_utilization": 0.85
     },
     "system": {
       "cpu_model": "x86_64",
       "cpu_cores_physical": 24,
       "cpu_cores_logical": 32,
       "cpu_freq_max_mhz": 4575,
       "ram_total_gb": 31.1,
       "ram_available_gb": 3.0,
       "ram_used_gb": 28.1,
       "ram_percent_used": 90.4
     },
     "cuda": {
       "available": true,
       "device_count": 1,
       "current_device": 0,
       "device_name": "NVIDIA GeForce RTX 4090 Laptop GPU",
       "memory_total_gb": 16.0,
       "memory_free_gb": 14.8,
       "memory_used_gb": 1.2
     },
     "display": {
       "DISPLAY": ":1",
       "WAYLAND_DISPLAY": "not set",
       "XDG_RUNTIME_DIR": "/run/user/1000",
       "QT_QPA_PLATFORM": "auto",
       "wslg_available": false,
       "x11_socket_exists": false
     },
     "env": {
       "MUJOCO_GL": "egl",
       "QT_DEBUG_PLUGINS": "0",
       "PLATFORM": "ubuntu",
       "MPI4PY_RC_INITIALIZE": "0"
     },
     "optional_deps": {
       "chat": true,
       "minigrid": true,
       "babyai": true,
       "mosaic_multigrid": true,
       "multigrid_ini": true,
       "pettingzoo": true,
       "mujoco": true,
       "atari": true,
       "vizdoom": true,
       "crafter": true,
       "nethack": true,
       "overcooked_ai": true,
       "smac": true,
       "smacv2": true,
       "rware": true,
       "ray_worker": true,
       "cleanrl_worker": true,
       "xuance_worker": true
     },
     "protobuf": {
       "compiled": true,
       "location": "gym_gui/services/trainer/proto",
       "pb2_files_count": 1,
       "grpc_files_count": 1
     }
   }
   [gym_gui] Qt platform plugin: xcb
