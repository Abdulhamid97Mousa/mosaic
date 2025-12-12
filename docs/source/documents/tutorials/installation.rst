Installation
============

This guide covers installing MOSAIC and its dependencies, explaining our modular
dependency architecture and how to choose what to install.

.. contents:: Table of Contents
   :local:
   :depth: 2

Why Modular Dependencies?
-------------------------

MOSAIC bridges many different frameworks: RL (CleanRL, RLlib, XuanCe), symbolic AI
(Jason BDI, SPADE), robotics (MuJoCo MPC), 3D simulation (Godot), and more. Each
framework has its own dependencies, and some have **conflicting requirements**.

Installing everything would:

- Cause **dependency conflicts** (e.g., different PyTorch versions)
- Waste **disk space** (10+ GB for all workers)
- Slow down installation unnecessarily

Our solution: **install only what you need**.

Dependency Architecture
-----------------------

MOSAIC uses two complementary systems:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Method
     - Use Case
     - Command
   * - **pyproject.toml**
     - Quick setup, optional extras
     - ``pip install -e ".[extra]"``
   * - **requirements/**
     - Full worker isolation, pinned versions
     - ``pip install -r requirements/xxx.txt``

pyproject.toml Structure
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   [project]
   dependencies = [...]           # Core GUI only (minimal)

   [project.optional-dependencies]
   # Environment Families
   gymnasium = [...]              # Base Gymnasium
   box2d = [...]                  # LunarLander, BipedalWalker
   mujoco = [...]                 # Ant, HalfCheetah, Humanoid
   atari = [...]                  # Breakout, Pong, SpaceInvaders
   minigrid = [...]               # Grid-world navigation
   pettingzoo = [...]             # Multi-agent (Chess, Go, MPE)
   vizdoom = [...]                # FPS environments

   # Workers (Training Backends)
   cleanrl = [...]                # CleanRL + PyTorch
   ray-rllib = [...]              # Ray/RLlib distributed
   xuance = [...]                 # XuanCe MARL
   spade-bdi = [...]              # SPADE BDI agents
   llm = [...]                    # Ollama/LangChain
   mujoco-mpc = [...]             # MuJoCo MPC

   # Convenience Bundles
   all-gymnasium = [...]          # All single-agent envs
   all-envs = [...]               # All environment families
   full = [...]                   # Everything + dev tools

requirements/ Directory
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   requirements/
   ├── base.txt              # Core GUI + shared libraries
   ├── cleanrl_worker.txt    # CleanRL training backend
   ├── ray_worker.txt        # RLlib distributed training
   ├── xuance_worker.txt     # XuanCe MARL algorithms
   ├── jason_worker.txt      # Jason BDI (includes Java setup)
   ├── spade_bdi_worker.txt  # SPADE BDI agents
   ├── llm_worker.txt        # Ollama/LangChain integration
   ├── mujoco_mpc_worker.txt # MuJoCo MPC controller
   ├── godot_worker.txt      # Godot game engine
   ├── vizdoom.txt           # ViZDoom environments
   ├── pettingzoo.txt        # PettingZoo multi-agent
   └── airsim_worker.txt     # AirSim drone simulation

Each worker file includes ``-r base.txt`` to pull in shared dependencies.

System Requirements
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Requirement
     - Details
   * - **Python**
     - 3.10, 3.11, or 3.12 (3.13 not yet supported)
   * - **Operating System**
     - Linux (recommended), macOS, Windows
   * - **GPU**
     - CUDA-capable GPU optional (for neural training)
   * - **RAM**
     - 8GB minimum, 16GB+ recommended
   * - **Disk Space**
     - ~2GB base, varies by worker selection

Quick Start Installation
------------------------

1. Clone and Setup Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
   cd MOSAIC

   # Create virtual environment (Python 3.10-3.12)
   python3.11 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or: .venv\Scripts\activate  # Windows

2. Install Core GUI (Minimal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This installs only what's needed to launch the GUI:

.. code-block:: bash

   pip install -e .

This gives you:

- PyQt6 visual interface
- gRPC infrastructure
- Telemetry and replay storage
- No training workers or environments

3. Add What You Need
^^^^^^^^^^^^^^^^^^^^

Choose your installation based on your use case:

.. tabs::

   .. tab:: CleanRL Training

      For single-agent RL training with CleanRL:

      .. code-block:: bash

         # Quick install via pyproject.toml
         pip install -e ".[cleanrl,box2d,atari]"

         # Or full isolation via requirements
         pip install -r requirements/cleanrl_worker.txt

   .. tab:: Multi-Agent (PettingZoo)

      For multi-agent environments like Chess, Go, MPE:

      .. code-block:: bash

         pip install -e ".[pettingzoo]"

         # Note: PettingZoo includes Stockfish for Chess
         # You may need to install stockfish binary separately on some systems

   .. tab:: MiniGrid

      For procedural grid-world environments:

      .. code-block:: bash

         pip install -e ".[minigrid]"

   .. tab:: ViZDoom

      For Doom-based visual RL:

      .. code-block:: bash

         pip install -e ".[vizdoom]"

         # Or via requirements (includes build dependencies)
         pip install -r requirements/vizdoom.txt

   .. tab:: Distributed Training (Ray/RLlib)

      For scalable distributed training:

      .. code-block:: bash

         pip install -e ".[ray-rllib]"

         # Or via requirements
         pip install -r requirements/ray_worker.txt

   .. tab:: Full Development

      Everything for development and testing:

      .. code-block:: bash

         pip install -e ".[full]"

         # This installs: all-envs + cleanrl + dev tools

Worker-Specific Installation
----------------------------

CleanRL Worker
^^^^^^^^^^^^^^

The CleanRL worker provides PPO, DQN, SAC, TD3 and other algorithms.

.. code-block:: bash

   # Via pyproject.toml
   pip install -e ".[cleanrl]"

   # Via requirements (recommended for production)
   pip install -r requirements/cleanrl_worker.txt

   # Verify
   python -c "import torch; print(f'PyTorch {torch.__version__}')"

XuanCe Worker (MARL)
^^^^^^^^^^^^^^^^^^^^

XuanCe provides multi-agent algorithms: MAPPO, QMIX, MADDPG.

.. code-block:: bash

   pip install -e ".[xuance]"

   # Or via requirements
   pip install -r requirements/xuance_worker.txt

.. warning::

   XuanCe requires ``mpi4py`` which needs MPI libraries:

   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install libopenmpi-dev

      # macOS
      brew install open-mpi

Jason BDI Worker
^^^^^^^^^^^^^^^^

Jason provides AgentSpeak-based BDI agents via Java/gRPC bridge.

.. code-block:: bash

   # Python dependencies
   pip install -r requirements/jason_worker.txt

   # Java setup (requires JDK 21)
   # Download Temurin JDK 21:
   curl -L -o temurin21.tar.gz \
     https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz
   tar -xzf temurin21.tar.gz
   rm temurin21.tar.gz

   # Build Jason components
   export JAVA_HOME="$(pwd)/jdk-21.0.5+11"
   export PATH="$JAVA_HOME/bin:$PATH"
   cd 3rd_party/jason_worker
   ./gradlew :jason-cli:build :jason-interpreter:build

See ``3rd_party/jason_worker/`` for detailed documentation.

SPADE BDI Worker
^^^^^^^^^^^^^^^^

SPADE provides Python-native BDI agents:

.. code-block:: bash

   pip install -e ".[spade-bdi]"

   # Or via requirements
   pip install -r requirements/spade_bdi_worker.txt

LLM Worker (Ollama)
^^^^^^^^^^^^^^^^^^^

For LLM-based agents using Ollama:

.. code-block:: bash

   pip install -e ".[llm]"

   # Install Ollama separately (see https://ollama.ai)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull a model
   ollama pull llama3.2

MuJoCo MPC Worker
^^^^^^^^^^^^^^^^^

For model predictive control with MuJoCo:

.. code-block:: bash

   pip install -e ".[mujoco-mpc]"

   # The MPC binary is pre-built in 3rd_party/mujoco_mpc_worker/bin/
   # Or build from source - see 3rd_party/mujoco_mpc_worker/README.md

Godot Worker
^^^^^^^^^^^^

For 3D game environment training:

.. code-block:: bash

   # The Godot binary is included in 3rd_party/godot_worker/bin/
   # No pip installation needed for basic usage

   # For Python RL bridge (optional):
   pip install godot-rl

Environment Families
--------------------

Environments are grouped by family. Install only what you need:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Family
     - Install Command
     - Environments
   * - **Box2D**
     - ``pip install -e ".[box2d]"``
     - LunarLander, BipedalWalker, CarRacing
   * - **MuJoCo**
     - ``pip install -e ".[mujoco]"``
     - Ant, HalfCheetah, Humanoid, Walker2d
   * - **Atari**
     - ``pip install -e ".[atari]"``
     - Breakout, Pong, SpaceInvaders, Asteroids
   * - **MiniGrid**
     - ``pip install -e ".[minigrid]"``
     - Empty, DoorKey, MultiRoom, RedBlueDoors
   * - **PettingZoo**
     - ``pip install -e ".[pettingzoo]"``
     - Chess, Go, Tic-Tac-Toe, MPE, Butterfly
   * - **ViZDoom**
     - ``pip install -e ".[vizdoom]"``
     - Basic, DeadlyCorridor, DefendTheCenter

Verifying Installation
----------------------

Launch the GUI
^^^^^^^^^^^^^^

.. code-block:: bash

   python -m gym_gui

If successful, you'll see the MOSAIC visual interface with the animated
space welcome screen.

Run Tests
^^^^^^^^^

.. code-block:: bash

   # Install dev dependencies
   pip install -e ".[dev]"

   # Run test suite
   pytest gym_gui/tests/

   # Run specific tests
   pytest gym_gui/tests/test_minigrid_empty_integration.py -v

Check Worker Availability
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from gym_gui.constants import (
       is_cleanrl_available,
       is_vizdoom_available,
       is_pettingzoo_available,
       is_torch_available,
   )

   print(f"CleanRL: {is_cleanrl_available()}")
   print(f"ViZDoom: {is_vizdoom_available()}")
   print(f"PettingZoo: {is_pettingzoo_available()}")
   print(f"PyTorch: {is_torch_available()}")

Troubleshooting
---------------

PyQt6 Display Issues
^^^^^^^^^^^^^^^^^^^^

On headless servers or WSL:

.. code-block:: bash

   # Install virtual display
   sudo apt-get install xvfb

   # Run with virtual display
   xvfb-run python -m gym_gui

CUDA/PyTorch Issues
^^^^^^^^^^^^^^^^^^^

If PyTorch doesn't detect your GPU:

.. code-block:: bash

   # Reinstall PyTorch with CUDA support
   pip install torch --index-url https://download.pytorch.org/whl/cu121

MPI Issues (XuanCe)
^^^^^^^^^^^^^^^^^^^

If ``mpi4py`` fails to install:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev

   # Then reinstall
   pip install mpi4py

Stockfish Issues (PettingZoo Chess)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If Chess environments fail:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install stockfish

   # macOS
   brew install stockfish

Dependency Conflicts
^^^^^^^^^^^^^^^^^^^^

If you encounter conflicts between workers:

.. code-block:: bash

   # Create separate environments for conflicting workers
   python -m venv .venv-cleanrl
   python -m venv .venv-rllib

   # Install each worker in its own environment
   source .venv-cleanrl/bin/activate
   pip install -r requirements/cleanrl_worker.txt

Next Steps
----------

After installation:

1. **Quick Start**: See :doc:`quickstart` to run your first experiment
2. **Basic Usage**: See :doc:`basic_usage` for GUI walkthrough
3. **Architecture**: See :doc:`../architecture/overview` to understand MOSAIC's design
