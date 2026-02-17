Installation on WSL (Windows Subsystem for Linux)
==================================================

This guide covers installing MOSAIC on **Windows** using **WSL 2** with an
Ubuntu distribution. MOSAIC was designed for native Ubuntu, but WSL 2
provides a compatible Linux environment with some additional caveats.

For native Ubuntu, see :doc:`ubuntu`.

.. note::

   WSL 2 is required (not WSL 1). WSL 2 runs a real Linux kernel, which is
   needed for native package builds and GPU passthrough.

Requirements
------------

- **Windows 11** (or Windows 10 build 19044+)
- **WSL 2** with an Ubuntu distribution (22.04 or 24.04 recommended)
- **NVIDIA GPU driver** installed on Windows (for CUDA support)

Setting Up WSL
--------------

If you don't have WSL installed yet:

.. code-block:: powershell

   # From PowerShell (Admin)
   wsl --install -d Ubuntu-22.04

After installation, open the Ubuntu terminal from the Start menu or run:

.. code-block:: powershell

   wsl -d Ubuntu-22.04

.. warning::

   **Always work inside a native WSL terminal.** Do not use PowerShell or
   Git Bash to run Python or pip against the WSL filesystem -- this uses
   Windows Python, not Linux Python, and will fail.

   .. list-table::
      :widths: 40 30 30
      :header-rows: 1

      * - Method
        - Shell
        - Python Used
      * - PowerShell (``cd \\\\wsl.localhost\\...``)
        - Windows
        - Windows Python (wrong)
      * - Git Bash (``cd //wsl.localhost/...``)
        - MSYS2/MinGW
        - Windows Python (wrong)
      * - **WSL terminal** (``wsl -d Ubuntu-22.04``)
        - **Linux Bash**
        - **WSL Python (correct)**

Install System Dependencies
-----------------------------

Inside your WSL terminal:

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
       stockfish

.. tip::

   If ``python3.11`` is not available, add the deadsnakes PPA:

   .. code-block:: bash

      sudo add-apt-repository ppa:deadsnakes/ppa
      sudo apt-get update
      sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

Setup
-----

All commands below must be run **inside the WSL terminal**:

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

   # Install with optional extras
   pip install -e ".[minigrid,mosaic_multigrid,multigrid_ini,pettingzoo,mujoco,atari,vizdoom,crafter,nethack,overcooked,rware,cleanrl,xuance,chat]"

Install Local 3rd-Party Packages
---------------------------------

.. code-block:: bash

   pip install -e 3rd_party/mosaic_multigrid/
   pip install -e 3rd_party/multigrid-ini/
   pip install -e 3rd_party/overcooked_ai/
   pip install -e 3rd_party/robotic-warehouse/

WSL-Specific Issues
-------------------

Broken ``.venv`` symlinks (I/O errors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom (when browsing from Windows Explorer or PowerShell):**

.. code-block:: text

   ls: cannot access '.venv/lib64': Input/output error
   ls: cannot access '.venv/bin/python3': Input/output error

**Cause:** Python's ``venv`` module creates Linux symlinks inside
``.venv/bin/`` and ``lib64``. These are valid inside WSL but appear as
broken I/O errors when accessed from Windows tools (PowerShell, Explorer,
Git Bash) via the ``\\wsl.localhost\`` network path.

**Fix:** Delete and recreate the venv **from inside WSL**:

.. code-block:: bash

   # Inside WSL terminal
   cd ~/path/to/MOSAIC
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate

**Prevention:** Never create or activate virtual environments from Windows
tools when working on the WSL filesystem.

``python3.11: command not found``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** Ubuntu 22.04 ships with Python 3.10 by default. Python 3.11 must
be installed separately.

**Fix:**

.. code-block:: bash

   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

GUI not appearing -- wrong ``DISPLAY`` variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   qt.qpa.xcb: could not connect to display 172.26.192.1:0
   This application failed to start because no Qt platform plugin could be initialized.

**Cause:** The ``DISPLAY`` environment variable points to an external X server
(e.g. ``172.26.192.1:0`` for VcXsrv or X410) instead of WSLg's built-in
display (``DISPLAY=:0``). This commonly happens when a user previously
configured an external X server in ``~/.bashrc`` and later upgraded to a
Windows 11 build that includes **WSLg** (built-in GUI support).

The MOSAIC settings output shows the current display configuration in the
``"display"`` section:

.. code-block:: json

   "display": {
     "DISPLAY": "172.26.192.1:0",
     "WAYLAND_DISPLAY": "wayland-0",
     "QT_QPA_PLATFORM": "xcb",
     "wslg_available": true,
     "x11_socket_exists": true
   }

If ``wslg_available`` is ``true`` but ``DISPLAY`` points to a network address,
you should switch to WSLg.

**Fix:** Comment out the old ``DISPLAY`` setting and use WSLg:

.. code-block:: bash

   # Remove the hardcoded DISPLAY from ~/.bashrc
   sed -i 's|export DISPLAY=.*|# &  # Commented out: use WSLg instead|' ~/.bashrc

   # Set WSLg display for this session
   export DISPLAY=:0

   # Verify
   echo $DISPLAY    # Should show :0

   # Relaunch
   bash ./run.sh

.. tip::

   WSLg (Windows Subsystem for Linux GUI) is included in WSL 2 on
   **Windows 11**. It automatically sets ``DISPLAY=:0`` and provides an X11
   socket at ``/tmp/.X11-unix/X0``. You do **not** need an external X server
   (VcXsrv, X410, MobaXterm) when WSLg is available.

   If WSLg is not available (older Windows 10 builds), you can either:

   - Install and run an external X server on Windows, or
   - Use a virtual framebuffer for headless operation:

   .. code-block:: bash

      sudo apt-get install -y xvfb
      xvfb-run python -m gym_gui

CUDA / GPU Not Detected
^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** On WSL 2, CUDA uses the **Windows** NVIDIA driver -- you should
**not** install the Linux NVIDIA driver inside WSL.

**Fix:**

1. Install the latest NVIDIA GPU driver **on Windows** (from
   `nvidia.com/drivers <https://www.nvidia.com/download/index.aspx>`_).

2. Verify inside WSL:

   .. code-block:: bash

      nvidia-smi                   # Should show your GPU
      python -c "import torch; print(torch.cuda.is_available())"

3. If PyTorch doesn't detect CUDA, reinstall with the correct CUDA version:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cu121

Filesystem Performance
^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** ``pip install`` or ``git`` operations are very slow.

**Cause:** Accessing files across the WSL/Windows boundary
(``/mnt/c/...`` or ``\\wsl.localhost\...``) is significantly slower than
working within the native WSL filesystem.

**Fix:** Keep the project on the Linux filesystem (``~/projects/MOSAIC``),
not on a Windows-mounted drive (``/mnt/c/Users/.../MOSAIC``).

Common Build Errors
-------------------

Build failures and dependency conflicts are shared across Ubuntu and WSL.
See the :doc:`common_errors` page for all known issues and fixes, including:

- ``swig`` not found (Box2D)
- ``nle`` build failure (NetHack)
- ``smac``/``smacv2`` protobuf version conflict with gRPC
- ``mpi4py`` MPI compiler not found
- Broken venv / wrong Python version
- Proto stub regeneration with ``tools/generate_protos.sh``

Verify Installation
-------------------

.. code-block:: bash

   # Inside WSL terminal with venv activated
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
       "cpu_cores_physical": 8,
       "cpu_cores_logical": 16,
       "cpu_freq_max_mhz": 0,
       "ram_total_gb": 15.5,
       "ram_available_gb": 13.3,
       "ram_used_gb": 2.2,
       "ram_percent_used": 14.0
     },
     "cuda": {
       "available": true,
       "device_count": 1,
       "current_device": 0,
       "device_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
       "memory_total_gb": 6.0,
       "memory_free_gb": 5.9,
       "memory_used_gb": 0.1
     },
     "display": {
       "DISPLAY": ":0",
       "WAYLAND_DISPLAY": "wayland-0",
       "XDG_RUNTIME_DIR": "/run/user/1000/",
       "QT_QPA_PLATFORM": "xcb",
       "wslg_available": true,
       "x11_socket_exists": true
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
       "mujoco": false,
       "atari": true,
       "vizdoom": false,
       "crafter": true,
       "nethack": false,
       "overcooked_ai": false,
       "smac": false,
       "smacv2": false,
       "rware": false,
       "ray_worker": true,
       "cleanrl_worker": false,
       "xuance_worker": false
     },
     "protobuf": {
       "compiled": true,
       "location": "gym_gui/services/trainer/proto",
       "pb2_files_count": 1,
       "grpc_files_count": 1
     }
   }
   [gym_gui] Qt platform plugin: xcb
