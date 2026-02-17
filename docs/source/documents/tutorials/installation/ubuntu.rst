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

If successful, the console will log detected optional dependencies:

.. code-block:: text

   optional_deps_detected  method=find_spec  found=12  total=16
