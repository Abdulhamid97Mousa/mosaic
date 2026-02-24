Common Errors
=============

This section documents errors commonly encountered when installing and running
MOSAIC. Errors shared across platforms are listed below. For platform-specific
issues, see the sub-pages:

.. toctree::
   :maxdepth: 2

   wsl
   ubuntu
   workers/index
   operators/index

-----

Virtual Environment
-------------------

``python -m env`` -- "No module named env"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   /usr/bin/python3: No module named env

**Cause:** The module is called ``venv`` (with a **v**), not ``env``.

**Fix:**

.. code-block:: bash

   python3.11 -m venv .venv

Wrong Python version (< 3.10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ERROR: Package 'mosaic' requires a different Python: 3.9.13 not in '>=3.10,<3.13'

**Cause:** MOSAIC requires Python 3.10, 3.11, or 3.12. Your system default
``python3`` may point to an older version.

**Fix:**

.. code-block:: bash

   # Check what you have
   python3 --version

   # Install Python 3.11 if needed (Ubuntu)
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

   # Create venv with the correct version
   python3.11 -m venv .venv

-----

System Dependencies (Build Failures)
-------------------------------------

Several MOSAIC optional packages compile native C/C++ code and require
system-level build tools. If you see ``Failed building wheel for ...``,
the fix is usually installing the missing system package.

**Install all build dependencies at once:**

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y \
       build-essential \
       cmake \
       swig \
       flex \
       bison \
       libbz2-dev \
       libopenmpi-dev

``box2d-py``: "command 'swig' failed"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   swig -python -c++ -IBox2D -small -O -includeall ...
   error: command 'swig' failed: No such file or directory
   ERROR: Failed building wheel for box2d-py

**Cause:** The ``box2d`` extra (LunarLander, BipedalWalker, CarRacing)
requires ``box2d-py``, which wraps C++ using SWIG. SWIG is not installed
by default.

**Fix:**

.. code-block:: bash

   sudo apt-get install -y swig build-essential
   pip install -e ".[box2d]"

``nle`` (NetHack): build failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ERROR: Failed building wheel for nle

**Cause:** The NetHack Learning Environment (``nle``) compiles the NetHack
game engine from C source. It requires ``cmake``, ``flex``, ``bison``, and
``libbz2-dev``.

**Fix:**

.. code-block:: bash

   sudo apt-get install -y build-essential cmake flex bison libbz2-dev
   pip install -e ".[nethack]"

``mpi4py``: "Cannot compile MPI programs"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   error: Cannot compile MPI programs. Check your configuration!

or:

.. code-block:: text

   mpicc: No such file or directory

**Cause:** The XuanCe worker depends on ``mpi4py``, which compiles against
MPI libraries. The MPI compiler wrapper ``mpicc`` is not installed by default.

**Fix:**

.. code-block:: bash

   # Ubuntu/Debian/WSL
   sudo apt-get install -y libopenmpi-dev

   # macOS
   brew install open-mpi

   # Then retry
   pip install mpi4py

.. tip::

   MOSAIC sets ``MPI4PY_RC_INITIALIZE=0`` in the ``.env`` file to prevent
   ``MPI_Init()`` from blocking when XuanCe is imported outside of ``mpirun``.
   Make sure your ``.env`` file includes this setting.

TensorBoard: "No module named 'pkg_resources'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ModuleNotFoundError: No module named 'pkg_resources'

This error may appear when launching TensorBoard (either from the command
line or via the embedded TensorBoard viewer in the Render View).

**Cause:** ``setuptools`` version 78 and later removed the ``pkg_resources``
package, which was previously bundled with ``setuptools``. TensorBoard
(and several other packages) still imports ``pkg_resources`` at startup.
If your environment has ``setuptools>=78``, the import fails.

**Fix:** Downgrade ``setuptools`` to a version that still includes
``pkg_resources``:

.. code-block:: bash

   pip install "setuptools<78"

   # Verify
   python -c "import pkg_resources; print('pkg_resources OK')"
   tensorboard --version

MOSAIC's ``requirements/base.txt`` now pins ``setuptools<78`` to prevent
this on fresh installs.

-----

Dependency Conflicts
--------------------

``smac`` / ``smacv2``:  protobuf version conflict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ERROR: Cannot install mosaic[smac] because these package versions have
   conflicting dependencies.

   The conflict is caused by:
       mosaic 0.1.0 depends on protobuf>=4.25.0
       smac 1.0.0 depends on protobuf<3.21

**Cause:** This is a fundamental version conflict between two parts of the
stack:

- **MOSAIC** uses gRPC (``grpcio``) for trainer <-> GUI communication. The
  gRPC protocol buffers are compiled using ``grpcio-tools`` and require
  ``protobuf>=4.25``. The generated stubs live in
  ``gym_gui/services/trainer/proto/`` and can be regenerated with:

  .. code-block:: bash

     bash tools/generate_protos.sh

- **SMAC / SMACv2** (StarCraft Multi-Agent Challenge) uses the StarCraft II
  client protocol (``s2clientprotocol``), which declares a dependency on
  ``protobuf<3.21``.

These two version ranges (``>=4.25`` vs ``<3.21``) are mutually exclusive,
so pip refuses to install both.

**Fix:** Install SMAC with ``--no-deps`` to bypass the declared constraint:

.. code-block:: bash

   # Install SMAC without its dependency constraints
   pip install --no-deps "smac @ git+https://github.com/oxwhirl/smac.git"
   pip install --no-deps "smacv2 @ git+https://github.com/oxwhirl/smacv2.git"

   # Manually install SMAC's other dependencies (excluding protobuf)
   pip install pysc2 s2clientprotocol pygame

.. note::

   **Why does this work at runtime?** The ``protobuf<3.21`` pin in SMAC is
   overly conservative. The ``s2clientprotocol`` package works with
   ``protobuf>=4.25`` in practice -- the protobuf library maintains backwards
   compatibility for compiled proto messages. Hamid's development machine
   runs both MOSAIC (gRPC) and SMAC successfully with ``protobuf>=4.25``.

**Proto regeneration:** If you modify ``trainer.proto`` or encounter stale
proto stubs, regenerate them:

.. code-block:: bash

   source .venv/bin/activate
   bash tools/generate_protos.sh

This runs ``grpc_tools.protoc`` on ``gym_gui/services/trainer/proto/trainer.proto``
and produces:

- ``trainer_pb2.py``:  message classes
- ``trainer_pb2_grpc.py``:  gRPC service stubs
- ``trainer_pb2.pyi``:  type stubs

``grpcio-tools``:  "grpcio-tools not installed"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``tools/generate_protos.sh``):**

.. code-block:: text

   (.venv) $ bash ./tools/generate_protos.sh
   [protos] Root: /path/to/mosaic
   [protos] grpcio-tools not installed. Run: pip install grpcio-tools

**Cause:** The proto generation script requires ``grpcio-tools`` to compile
``.proto`` files into Python stubs. This package is listed in MOSAIC's core
dependencies (``grpcio-tools>=1.60.0``), but may be missing if you installed
with ``--no-deps`` or if the initial ``pip install -e .`` was interrupted.

**Fix:**

.. code-block:: bash

   pip install grpcio-tools

Then re-run the proto generation:

.. code-block:: bash

   (.venv) $ bash ./tools/generate_protos.sh
   [protos] Root: /path/to/mosaic
   [protos] Generating trainer stubs
   [protos] Done

.. note::

   Proto stubs are **checked into the repository**, so you only need to
   regenerate them if you modify ``trainer.proto`` or if the stubs become
   stale after a ``protobuf`` version upgrade.

-----

Startup Errors
--------------

Trainer daemon failed to start: "VersionError: Detected incompatible Protobuf Gencode/Runtime versions"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``./run.sh``):**

.. code-block:: text

   Checking for existing trainer processes...
   Starting trainer daemon...
   Waiting for trainer daemon...
   ERROR: Trainer daemon failed to start. Check var/logs/trainer_daemon.log

**Check the log** (``var/logs/trainer_daemon.log``):

.. code-block:: text

   File ".../google/protobuf/runtime_version.py", line 100, in ValidateProtobufRuntimeVersion
     _ReportVersionError(
   File ".../google/protobuf/runtime_version.py", line 50, in _ReportVersionError
     raise VersionError(msg)
   google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf Gencode/Runtime versions when loading trainer.proto: gencode 6.31.1 runtime 5.29.6. Runtime version cannot be older than the linked gencode version.

**Cause:** The generated Protobuf files (``trainer_pb2.py``) checked into the repository were compiled with a newer version of Protobuf (e.g., 6.31.1) than the one installed in your Python environment (e.g., 5.29.6). Protobuf requires that the runtime library be at least as new as the compiler used to generate the code.

**Fix:** Regenerate the Protobuf files using your local environment's tools:

.. code-block:: bash

   source .venv/bin/activate
   bash tools/generate_protos.sh

Then restart the application:

.. code-block:: bash

   ./run.sh

Trainer daemon failed to start: "No module named 'dotenv'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``./run.sh``):**

.. code-block:: text

   Checking for existing trainer processes...
   Starting trainer daemon...
   Waiting for trainer daemon...
   ERROR: Trainer daemon failed to start. Check var/logs/trainer_daemon.log

**Check the log** (``var/logs/trainer_daemon.log``):

.. code-block:: text

   google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf
   Gencode/Runtime versions when loading trainer.proto: gencode 6.31.1 runtime 5.29.6.

**Cause:** The generated Protobuf files (``trainer_pb2.py``) checked into the
repository were compiled with a newer version of Protobuf than the one
installed in your environment.

**Fix:** Regenerate the Protobuf files using your local environment's tools:

.. code-block:: bash

   source .venv/bin/activate
   bash tools/generate_protos.sh
   ./run.sh

Trainer daemon: "No module named 'dotenv'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (in ``var/logs/trainer_daemon.log``):**

.. code-block:: text

   ModuleNotFoundError: No module named 'dotenv'

**Cause:** The core MOSAIC package was not installed.

**Fix:**

.. code-block:: bash

   source .venv/bin/activate
   pip install -e .
   ./run.sh

Missing ``.env`` file
^^^^^^^^^^^^^^^^^^^^^

**Symptom:** The GUI launches but uses incorrect defaults, or the trainer
daemon fails to connect with unexpected settings.

**Cause:** MOSAIC reads configuration from a ``.env`` file in the project
root. This file is **gitignored** (never committed) and must be created
locally from the provided template.

**Fix:**

.. code-block:: bash

   cp .env.example .env

The ``.env.example`` file is fully documented with all available settings.
Key variables you should review:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Purpose
   * - ``QT_API=PyQt6``
     - **Required.** Must be ``PyQt6`` with exact capitalization for qasync
   * - ``PLATFORM=ubuntu``
     - Platform identifier (``ubuntu`` for both native Ubuntu and WSL)
   * - ``MUJOCO_GL=egl``
     - MuJoCo rendering backend (``egl`` for headless, ``glfw`` for display)
   * - ``MPI4PY_RC_INITIALIZE=0``
     - **Required.** Prevents MPI deadlock when XuanCe is imported
   * - ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python``
     - **Required for SMAC.** Allows protobuf 5.x to read SMAC's 3.x pb2 files
   * - ``GRPC_VERBOSITY=ERROR``
     - gRPC log level (``ERROR`` recommended; ``DEBUG`` is very noisy)
   * - ``WANDB_API_KEY``
     - Weights & Biases API key (get from https://wandb.ai/authorize)
   * - ``OPENROUTER_API_KEY``
     - OpenRouter LLM API key (get from https://openrouter.ai/keys)
   * - ``HF_TOKEN``
     - HuggingFace token for gated models

.. warning::

   The ``.env`` file contains **API keys and secrets**. It is gitignored by
   default. Never commit it to version control.

Startup crash: "No module named 'torch'" / "No module named 'psutil'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ModuleNotFoundError: No module named 'torch'

Or:

.. code-block:: text

   ModuleNotFoundError: No module named 'psutil'

**Cause:** ``app.py`` imports ``torch``, ``psutil``, and ``pynvml``
unconditionally at startup for GPU detection and system info display.

**Fix:**

.. code-block:: bash

   pip install torch psutil pynvml

.. note::

   For CUDA GPU support, install PyTorch with the CUDA index:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cu121

Startup crash: "No module named 'stockfish'" (misleading Qt error)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   [gym_gui] Qt bindings not available. Install qtpy and a Qt backend
   (PyQt5/PyQt6/PySide2/PySide6): No module named 'stockfish'

**Cause:** The ``stockfish`` Python package was imported at module level.
The missing import was caught by a generic handler, producing a misleading
"Qt bindings not available" message.

**Fix:** This was a bug (fixed). The import was moved to a lazy location.
If you encounter this on an older version:

.. code-block:: bash

   pip install stockfish

Startup crash: "No module named 'syllabus'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ModuleNotFoundError: No module named 'syllabus'

**Cause:** The CleanRL worker uses `Syllabus-RL <https://github.com/RyanNavillus/Syllabus>`_
for curriculum learning. Syllabus is a local 3rd-party package.

**Fix:**

.. code-block:: bash

   # If the directory is empty, initialize the submodule
   git submodule update --init 3rd_party/Syllabus

   # Install in editable mode
   pip install -e 3rd_party/Syllabus

Worker load warnings (non-fatal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During startup, MOSAIC discovers workers via entry points. Missing optional
workers produce warnings but **do not prevent the GUI from launching**:

.. code-block:: text

   Failed to load worker 'cleanrl' from entry point: No module named 'scipy'
   Failed to load worker 'ray' from entry point: No module named 'tree'

These are informational -- install the relevant extras to enable each worker:

.. code-block:: bash

   # CleanRL worker
   pip install -e 3rd_party/Syllabus
   pip install matplotlib scipy

   # Ray/RLlib worker
   pip install -e ".[ray-rllib]"

-----

Runtime Errors
--------------

CleanRL training crash: "No module named 'tyro'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (in ``var/logs/trainer_daemon.log``):**

.. code-block:: text

   ModuleNotFoundError: No module named 'tyro'

**Cause:** The CleanRL worker requires ``tyro``, ``wandb``, ``tensorboard``,
``tenacity``, and ``moviepy``.

**Fix:** Install the full CleanRL extras:

.. code-block:: bash

   pip install -e ".[cleanrl]"

``pynvml`` deprecation warning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (at startup):**

.. code-block:: text

   FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.

**Fix:** Uninstall the deprecated wrapper:

.. code-block:: bash

   pip uninstall -y pynvml
   pip install nvidia-ml-py

gRPC: "Log level DEBUG is not suitable for production"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (when running ``./run.sh``):**

.. code-block:: text

   WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
   W0000 00:00:1771284312.764300  84092 log.cc:110] Log level DEBUG is not suitable
   for production.

**This is harmless.** Set ``GRPC_VERBOSITY=ERROR`` in ``.env`` to suppress it.

``resource_tracker`` -- "leaked shared_memory objects"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (on shutdown):**

.. code-block:: text

   resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown

**This is harmless.** The tracker cleans up the leaked segments automatically.

Syllabus / CleanRL: missing transitive dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ModuleNotFoundError: No module named 'matplotlib'

or:

.. code-block:: text

   Failed to load worker 'cleanrl' from entry point: No module named 'scipy'

**Fix:**

.. code-block:: bash

   pip install matplotlib scipy

Stockfish not found (Chess Engine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   stockfish.StockfishException: Stockfish binary not found

**Fix:**

.. code-block:: bash

   # Ubuntu/Debian/WSL
   sudo apt-get install -y stockfish

   # macOS
   brew install stockfish

CleanRL PPO crash: "mat1 and mat2 shapes cannot be multiplied" (Discrete obs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when training PPO on FrozenLake-v1, Taxi-v3, CliffWalking-v0, etc.):**

.. code-block:: text

   RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x4 and 1x64)

**Cause:** Environments with ``Discrete`` observation spaces (e.g.
``FrozenLake-v1`` has ``Discrete(16)``) return a single integer as the
observation. CleanRL's ``MLPAgent`` computes the input dimension as
``prod(obs_space.shape)``. For ``Discrete(n)`` the shape is ``()`` so
``prod(()) == 1``, causing a shape mismatch.

**Fix:** This is now handled automatically. The ``make_env`` factory detects
``Discrete`` observation spaces and wraps them with one-hot encoding. For
example, ``FrozenLake-v1``'s ``Discrete(16)`` becomes ``Box(0, 1, (16,),
float32)``.

**Affected environments:** Any environment with a ``Discrete`` observation
space, including ``FrozenLake-v1``, ``Taxi-v3``, and ``CliffWalking-v0``.
Environments with ``Box`` observation spaces (``CartPole-v1``,
``LunarLander-v2``, etc.) are unaffected.

-----

Quick Reference: All System Dependencies
-----------------------------------------

.. code-block:: bash

   # Install everything at once
   sudo apt-get update
   sudo apt-get install -y \
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
   :widths: 25 35 40
   :header-rows: 1

   * - System Package
     - Required By
     - Error Without It
   * - ``build-essential``
     - Most native extensions
     - ``gcc: command not found``
   * - ``cmake``
     - ``nle`` (NetHack)
     - ``Failed building wheel for nle``
   * - ``swig``
     - ``box2d-py``
     - ``command 'swig' failed``
   * - ``flex``, ``bison``
     - ``nle`` (NetHack)
     - ``Failed building wheel for nle``
   * - ``libbz2-dev``
     - ``nle`` (NetHack)
     - ``Failed building wheel for nle``
   * - ``libopenmpi-dev``
     - ``mpi4py`` (XuanCe)
     - ``Cannot compile MPI programs``
   * - ``libegl1``, ``libgl1``, ``libopengl0``
     - PyQt6 (OpenGL/EGL)
     - ``libGL.so.1: cannot open shared object file``
   * - ``libxkbcommon0``
     - PyQt6 (keyboard handling)
     - ``libxkbcommon.so.0: cannot open shared object file``
   * - ``libnss3``, ``libnspr4``
     - PyQt6-WebEngine (TensorBoard viewer)
     - ``libsmime3.so: cannot open shared object file``
   * - ``libasound2``
     - PyQt6-WebEngine (audio backend)
     - ``libasound.so.2: cannot open shared object file``
   * - ``libxcomposite1``
     - PyQt6-WebEngine (X11 compositing)
     - ``libXcomposite.so.1: cannot open shared object file``
   * - ``libxdamage1``
     - PyQt6-WebEngine (X11 damage tracking)
     - ``libXdamage.so.1: cannot open shared object file``
   * - ``libxkbfile1``
     - PyQt6-WebEngine (keyboard layout files)
     - ``libxkbfile.so.1: cannot open shared object file``
   * - ``libxcb-cursor0``
     - PyQt6 (X11 cursor support, required since Qt 6.5)
     - ``Could not load the Qt platform plugin "xcb"``
   * - ``fontconfig``
     - PyQt6 (font discovery)
     - GUI launches but all text is invisible
   * - ``fonts-dejavu-extra``, ``fonts-liberation``
     - System fonts for Qt rendering
     - GUI launches but all text is invisible
   * - ``stockfish``
     - Chess engine
     - ``Stockfish binary not found``
   * - ``xvfb``
     - Headless display (optional)
     - ``Could not find platform plugin "xcb"``
