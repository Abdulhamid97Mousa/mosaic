Common Errors
=============

This page documents errors commonly encountered when installing and running
MOSAIC. These apply to both **native Ubuntu** and **WSL** setups.

For platform-specific issues, see:

- :doc:`ubuntu` -- native Ubuntu setup
- :doc:`wsl` -- Windows Subsystem for Linux setup

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

Broken ``.venv`` (I/O errors on symlinks)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ls: cannot access '.venv/lib64': Input/output error
   ls: cannot access '.venv/bin/python3': Input/output error
   .venv/bin/python: No such file or directory

**Cause:** The venv was created with a Python interpreter that is no longer
available, or the symlinks inside ``.venv/bin/`` are broken. This is
especially common on **WSL** when the venv is accessed from Windows tools
(PowerShell, Explorer, Git Bash) -- Linux symlinks are not visible to Windows.

**Fix:** Delete and recreate the venv:

.. code-block:: bash

   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate

On **WSL**, always create and use the venv from a native WSL terminal
(not PowerShell or Git Bash). See :doc:`wsl` for details.

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

``box2d-py`` -- "command 'swig' failed"
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

``nle`` (NetHack) -- build failure
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

``mpi4py`` -- "Cannot compile MPI programs"
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

TensorBoard -- "No module named 'pkg_resources'"
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

``smac`` / ``smacv2`` -- protobuf version conflict
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

- **MOSAIC** uses gRPC (``grpcio``) for trainer ↔ GUI communication. The
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

- ``trainer_pb2.py`` -- message classes
- ``trainer_pb2_grpc.py`` -- gRPC service stubs
- ``trainer_pb2.pyi`` -- type stubs

``grpcio-tools`` -- "grpcio-tools not installed"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``tools/generate_protos.sh``):**

.. code-block:: text

   (.venv) $ bash ./tools/generate_protos.sh
   [protos] Root: /home/zahra/projects_hamid/GUI_BDI_RL
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
   [protos] Root: /home/zahra/projects_hamid/GUI_BDI_RL
   [protos] Generating trainer stubs
   [protos] Done

This compiles ``gym_gui/services/trainer/proto/trainer.proto`` and produces:

- ``trainer_pb2.py`` -- message classes
- ``trainer_pb2_grpc.py`` -- gRPC service stubs
- ``trainer_pb2.pyi`` -- type stubs

.. note::

   Proto stubs are **checked into the repository**, so you only need to
   regenerate them if you modify ``trainer.proto`` or if the stubs become
   stale after a ``protobuf`` version upgrade.

-----

Startup Errors
--------------

Trainer daemon failed to start -- "No module named 'dotenv'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``./run.sh``):**

.. code-block:: text

   Checking for existing trainer processes...
   Starting trainer daemon...
   Waiting for trainer daemon...
   ERROR: Trainer daemon failed to start. Check var/logs/trainer_daemon.log

**Check the log** (``var/logs/trainer_daemon.log``):

.. code-block:: text

   File "/home/.../GUI_BDI_RL/gym_gui/config/settings.py", line 11, in <module>
       from dotenv import load_dotenv
   ModuleNotFoundError: No module named 'dotenv'

**Cause:** The core MOSAIC package was not installed. The ``run.sh`` script
starts the trainer daemon (``gym_gui.services.trainer_daemon``) which imports
``python-dotenv`` and other core dependencies. If ``pip install -e .`` was
never run (or failed), these imports fail.

**Fix:**

.. code-block:: bash

   source .venv/bin/activate
   pip install -e .

   # Verify core dependencies are installed
   python -c "from dotenv import load_dotenv; print('dotenv OK')"
   python -c "import grpc; print('gRPC OK')"

Then retry:

.. code-block:: bash

   ./run.sh

Trainer daemon failed to start -- "libGL.so.1: cannot open shared object file"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``./run.sh``):**

.. code-block:: text

   Checking for existing trainer processes...
   Starting trainer daemon...
   Waiting for trainer daemon...
   ERROR: Trainer daemon failed to start. Check var/logs/trainer_daemon.log

**Check the log** (``var/logs/trainer_daemon.log``):

.. code-block:: text

   File "/home/.../gym_gui/validations/validations_ui.py", line 8, in <module>
       from PyQt6 import QtCore, QtWidgets
   ImportError: libGL.so.1: cannot open shared object file: No such file or directory

**Cause:** PyQt6 depends on OpenGL and EGL shared libraries at import time,
even for non-GUI components like the trainer daemon. The import chain is:

.. code-block:: text

   trainer_daemon → dispatcher → subprocess_validation → validations_ui → PyQt6

On native Ubuntu **desktop** these libraries come pre-installed, but on
**WSL** minimal installs and **headless servers** they are missing.

**Fix:**

.. code-block:: bash

   sudo apt-get install -y libegl1 libgl1 libopengl0 libxkbcommon0

Then retry:

.. code-block:: bash

   ./run.sh

.. tip::

   If you see similar errors for other ``lib*.so`` files, install the
   full Qt runtime dependencies:

   .. code-block:: bash

      sudo apt-get install -y libgl1 libegl1 libopengl0 \
          libxkbcommon0 libxkbcommon-x11-0 libdbus-1-3 \
          libfontconfig1 libxcb-icccm4 libxcb-image0 \
          libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
          libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 libxcb-cursor0

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
     - HuggingFace token for gated models (get from https://huggingface.co/settings/tokens)

.. warning::

   The ``.env`` file contains **API keys and secrets**. It is gitignored by
   default. Never commit it to version control.

Startup crash -- "No module named 'torch'" / "No module named 'psutil'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when running ``./run.sh``):**

.. code-block:: text

   Launching MOSAIC...
   File "/home/.../gym_gui/app.py", line 45, in _format_settings
       import torch
   ModuleNotFoundError: No module named 'torch'

Or:

.. code-block:: text

   File "/home/.../gym_gui/app.py", line 112, in _get_system_info
       import psutil
   ModuleNotFoundError: No module named 'psutil'

**Cause:** ``app.py`` imports ``torch``, ``psutil``, and ``pynvml``
unconditionally at startup for GPU detection and system info display. These
are core dependencies (listed in ``pyproject.toml`` and ``requirements/base.txt``)
but may be missing if you only ran ``pip install -e .`` before they were added.

**Fix:**

.. code-block:: bash

   pip install torch psutil pynvml

.. note::

   For CUDA GPU support, install PyTorch with the CUDA index:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cu121

Startup crash -- "No module named 'stockfish'" (misleading Qt error)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   [gym_gui] Qt bindings not available. Install qtpy and a Qt backend
   (PyQt5/PyQt6/PySide2/PySide6): No module named 'stockfish'

**Cause:** The ``stockfish`` Python package (chess engine bindings) was
imported at module level in ``stockfish_service.py``. When the module was
loaded during the ``MainWindow`` import chain, the missing ``stockfish``
package raised an ``ImportError`` that was caught by the generic handler in
``app.py``, producing a misleading "Qt bindings not available" message.

**Fix:** This was a bug (fixed). The ``from stockfish import Stockfish``
import was moved from the module top-level to the ``StockfishService.start()``
method, making it lazy. The ``stockfish`` package is now only needed when
actually starting a chess game against the AI.

If you encounter this error on an older version, install the package:

.. code-block:: bash

   pip install stockfish

Startup crash -- "NoneType takes no arguments" (QWebEnginePage)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   File "/home/.../gym_gui/ui/widgets/filtered_web_engine.py", line 34, in <module>
       class FilteredWebEnginePage(QWebEnginePage):
   TypeError: NoneType takes no arguments

**Cause:** The ``PyQt6-WebEngine`` package is not installed, or it fails to
import due to missing system libraries. MOSAIC uses ``QWebEnginePage`` for the
TensorBoard artifact viewer. When the import fails, ``QWebEnginePage`` resolves
to ``None`` and the class definition crashes.

The confusing error message ("NoneType takes no arguments") is because
``filtered_web_engine.py`` catches the ``ImportError`` silently:

.. code-block:: python

   try:
       from PyQt6.QtWebEngineCore import QWebEnginePage
   except ImportError:
       QWebEnginePage = None  # <-- this is what causes the confusing error

To see the **real** underlying error, run:

.. code-block:: bash

   python -c "from PyQt6.QtWebEngineCore import QWebEnginePage"

Common underlying errors:

- ``libXcomposite.so.1: cannot open shared object file`` → missing ``libxcomposite1``
- ``libXdamage.so.1: cannot open shared object file`` → missing ``libxdamage1``
- ``libxkbfile.so.1: cannot open shared object file`` → missing ``libxkbfile1``
- ``libsmime3.so: cannot open shared object file`` → missing ``libnss3``

**Fix:**

.. code-block:: bash

   # Install the Python package
   pip install PyQt6-WebEngine

   # Install ALL system libraries required by Qt6 WebEngine
   sudo apt-get install -y \
       libnss3 libnspr4 libasound2 \
       libxcomposite1 libxdamage1 libxkbfile1

   # Verify the import works
   python -c "from PyQt6.QtWebEngineCore import QWebEnginePage; print('OK')"

.. tip::

   If the verify step still fails with a different ``lib*.so`` error, use
   ``ldd`` to find all missing libraries at once:

   .. code-block:: bash

      ldd $(python -c "import PyQt6; print(PyQt6.__path__[0])")/Qt6/lib/libQt6WebEngineCore.so.6 | grep "not found"

Startup crash -- "No module named 'syllabus'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   File "/home/.../cleanrl_worker/wrappers/curriculum.py", line 47, in <module>
       from syllabus.core import ReinitTaskWrapper
   ModuleNotFoundError: No module named 'syllabus'

**Cause:** The CleanRL worker uses `Syllabus-RL <https://github.com/RyanNavillus/Syllabus>`_
for curriculum learning. Syllabus is a local 3rd-party package in
``3rd_party/Syllabus/`` (git submodule).

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
   pip install -e 3rd_party/Syllabus        # Curriculum learning
   pip install matplotlib scipy             # Transitive dependencies

   # Ray/RLlib worker
   pip install -e ".[ray-rllib]"

-----

Runtime Errors
--------------

CleanRL training crash -- "No module named 'tyro'"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (in ``var/logs/trainer_daemon.log``):**

.. code-block:: text

   ModuleNotFoundError: No module named 'tyro'

**Symptom:** You launch a training run from the CleanRL training form in the
GUI. The worker process starts, immediately crashes, and the GUI may close
or show an error.

**Cause:** The CleanRL worker requires several packages (``tyro``, ``wandb``,
``tensorboard``, ``tenacity``, ``moviepy``) that are defined in the ``cleanrl``
extras group in ``pyproject.toml``. If you installed CleanRL dependencies
manually (e.g. just ``pip install matplotlib scipy``), these training-specific
packages may be missing.

**Fix:** Install the full CleanRL extras:

.. code-block:: bash

   pip install -e ".[cleanrl]"

This installs ``tyro``, ``wandb``, ``tensorboard``, ``tenacity``, and
``moviepy`` together.

QRhiGles2 -- "Failed to create temporary context" (WSL only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (repeated on console, may crash the GUI):**

.. code-block:: text

   QRhiGles2: Failed to create temporary context
   QRhiGles2: Failed to create context
   Failed to create QRhi for QBackingStoreRhiSupport
   QQuickWidget: Failed to get a QRhi from the top-level widget's window
   QQuickWidget: Attempted to render scene with no rhi

**Cause:** Qt 6's QQuickWidget uses the RHI (Rendering Hardware Interface) for
GPU-accelerated rendering. On **WSLg**, hardware OpenGL contexts are not
available through the virtual GPU, so RHI fails to create an OpenGL ES 2
context. This affects QML-based UI components (FastLane video panel) and can
crash the GUI entirely.

**This issue only affects WSL.** Native Ubuntu with a real GPU and display
server does not have this problem.

**Fix:** Add these lines to your ``.env`` file:

.. code-block:: bash

   # Bypass RHI entirely -- use Qt's QPainter-based software renderer
   QT_QUICK_BACKEND=software

   # Fallback: if any non-Quick GL usage remains, force Mesa software GL
   QSG_RHI_BACKEND=gl
   LIBGL_ALWAYS_SOFTWARE=1

``QT_QUICK_BACKEND=software`` tells Qt Quick to skip the RHI layer and use
the QPainter-based software adaptation of the scene graph. No OpenGL context
is needed at all. The main application window (QWidget-based) is unaffected
-- only QML/Quick content (the FastLane video panel) uses this fallback.

.. note::

   ``QSG_RHI_BACKEND=software`` is **not** a valid value (Qt will print
   "Unknown key" and fall back to default). The correct env var for
   software rendering is ``QT_QUICK_BACKEND=software``.

**Native Ubuntu users** should keep the default ``.env.example`` values
(``QSG_RHI_BACKEND=gl``) or remove these lines entirely. Hardware-accelerated
rendering works correctly on native installs with NVIDIA/AMD drivers.

``pynvml`` deprecation warning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (at startup):**

.. code-block:: text

   FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
   If you did not install pynvml directly, please report this to the maintainers of the
   package that installed pynvml for you.

**Cause:** The deprecated ``pynvml`` wrapper package is installed alongside the
actual ``nvidia-ml-py`` package. PyTorch imports ``pynvml`` internally and the
deprecated wrapper emits this warning. Both packages expose the same
``pynvml`` module, but the wrapper adds a noisy deprecation notice.

**Fix:** Uninstall the deprecated wrapper. The underlying ``nvidia-ml-py``
provides the same ``pynvml`` module without the warning:

.. code-block:: bash

   pip uninstall -y pynvml
   pip install nvidia-ml-py

   # Verify: should print no warning
   python -c "import pynvml; print('OK')"

.. note::

   MOSAIC's ``pyproject.toml`` and ``requirements/base.txt`` now depend on
   ``nvidia-ml-py`` (not ``pynvml``) to prevent this warning on fresh installs.

Syllabus / CleanRL -- missing transitive dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When installing ``Syllabus`` (curriculum learning for CleanRL) via
``pip install -e 3rd_party/Syllabus``, some transitive dependencies may not
be pulled in automatically, causing import failures at worker discovery time.

**Error 1 -- matplotlib:**

.. code-block:: text

   File ".../Syllabus/syllabus/core/curriculum_base.py", line 4, in <module>
       import matplotlib.pyplot as plt
   ModuleNotFoundError: No module named 'matplotlib'

**Error 2 -- scipy:**

.. code-block:: text

   Failed to load worker 'cleanrl' from entry point: No module named 'scipy'

**Fix:** Install the missing packages:

.. code-block:: bash

   pip install matplotlib scipy

gRPC -- "Log level DEBUG is not suitable for production"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (when running ``./run.sh``):**

.. code-block:: text

   WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
   W0000 00:00:1771284312.764300  84092 log.cc:110] Log level DEBUG is not suitable
   for production. Prefer WARNING or ERROR. However if you see this message in a
   debug environment or test environment it is safe to ignore this message.

**Cause:** The ``.env`` file sets ``GRPC_VERBOSITY=debug`` for development
logging. The abseil library (used internally by gRPC) prints this warning
whenever verbosity is set below WARNING. The message appears once per gRPC
channel creation -- during startup, ``run.sh`` polls the trainer daemon up to
10 times, so you may see the warning repeated.

**This is harmless.** It is not an error. If the warnings bother you, you can
set ``GRPC_VERBOSITY=ERROR`` in ``.env``, but you will lose gRPC debug logs.

PyQt6 -- "Could not load the Qt platform plugin"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load
   the Qt xcb platform plugin.
   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though
   it was found.
   This application failed to start because no Qt platform plugin could be initialized.

**Cause:** PyQt6 >= 6.5 requires ``libxcb-cursor0`` for the X11 (xcb) platform
plugin. This library is not installed by default on minimal Ubuntu or WSL
distributions.

**Fix:**

.. code-block:: bash

   sudo apt-get install -y libxcb-cursor0

If the GUI still fails to launch after installing ``libxcb-cursor0``, check that
your display is configured:

.. code-block:: bash

   # Check DISPLAY is set (WSLg should set this automatically)
   echo $DISPLAY    # Should show :0

   # If empty, set it manually
   export DISPLAY=:0

   # For headless environments without a display, use a virtual framebuffer
   sudo apt-get install -y xvfb
   xvfb-run python -m gym_gui

GUI launches but text shows as squares or is invisible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** The MOSAIC window appears but all labels, buttons, and menus show
as squares (□□□□□) or are completely blank:

.. image:: /_static/figures/wsl_missing_fonts.jpg
   :alt: MOSAIC GUI with missing fonts — all text rendered as squares
   :width: 100%

**Cause:** This happens when either:

1. The ``fontconfig`` package is not installed — Qt uses fontconfig to discover
   system fonts, and without it, Qt cannot find any fonts at all.
2. The font cache is stale — fonts were installed but ``fc-cache`` was not run,
   so Qt still cannot discover them.

This is especially common on **WSL** minimal installs, which ship with very
few fonts and no fontconfig.

**Fix:**

.. code-block:: bash

   # Install fontconfig and common fonts
   sudo apt-get install -y fontconfig fonts-dejavu-extra fonts-liberation fonts-noto

   # Rebuild the font cache (required after installing new fonts)
   fc-cache -fv

Then restart the application. Text should render correctly:

.. image:: /_static/figures/wsl_fonts_fixed.jpg
   :alt: MOSAIC GUI with fonts working correctly after installing fontconfig
   :width: 100%

CUDA / GPU not detected
^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: python

   >>> import torch
   >>> torch.cuda.is_available()
   False

**Fix (native Ubuntu):**

.. code-block:: bash

   # Install PyTorch with CUDA
   pip install torch --index-url https://download.pytorch.org/whl/cu121

**Fix (WSL 2):** On WSL, CUDA uses the **Windows** NVIDIA driver. Do **not**
install the Linux NVIDIA driver inside WSL.

.. code-block:: bash

   # Verify the Windows driver is visible inside WSL
   nvidia-smi

   # If nvidia-smi works but torch doesn't detect CUDA, reinstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121

Stockfish not found (Chess Engine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   stockfish.StockfishException: Stockfish binary not found

**Cause:** `Stockfish <https://stockfishchess.org/>`_ is a standalone chess
engine 

**Fix:**

.. code-block:: bash

   # Ubuntu/Debian/WSL
   sudo apt-get install -y stockfish

   # macOS
   brew install stockfish

``resource_tracker`` -- "leaked shared_memory objects"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning (on shutdown):**

.. code-block:: text

   resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown

**Cause:** Some workers (particularly those using multiprocessing) allocate
shared memory segments that are not explicitly unlinked before the process
exits. Python's ``resource_tracker`` detects and cleans them up automatically.

**This is harmless.** The tracker cleans up the leaked segments. No action
is needed unless the warning count is very large (hundreds), which could
indicate a real memory leak in a custom worker.

Overcooked -- Python version incompatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   ERROR: overcooked-ai requires Python >=3.10,<3.11

**Cause:** Overcooked-AI is pinned to Python 3.10 only.

**Fix:** Create a separate venv with Python 3.10:

.. code-block:: bash

   python3.10 -m venv .venv-overcooked
   source .venv-overcooked/bin/activate
   pip install -e ".[overcooked]"
   pip install -e 3rd_party/overcooked_ai/

CleanRL PPO crash -- "mat1 and mat2 shapes cannot be multiplied" (Discrete obs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error (when training PPO on FrozenLake-v1, Taxi-v3, CliffWalking-v0, etc.):**

.. code-block:: text

   RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x4 and 1x64)

**Cause:** Environments with ``Discrete`` observation spaces (e.g.
``FrozenLake-v1`` has ``Discrete(16)``) return a single integer as the
observation. CleanRL's ``MLPAgent`` computes the input dimension as
``prod(obs_space.shape)``. For ``Discrete(n)`` the shape is ``()`` so
``prod(()) == 1``, causing the agent to create ``Linear(1, 64)`` instead of
the correct size. When the vectorized environment passes a batch of
observations, the tensor shape does not match the linear layer and PyTorch
raises a ``RuntimeError``.

**Fix:** This is now handled automatically. The ``make_env`` factory in
``cleanrl_worker/wrappers/minigrid.py`` detects ``Discrete`` observation
spaces and wraps the environment with a one-hot encoding wrapper. For
example, ``FrozenLake-v1``'s ``Discrete(16)`` becomes ``Box(0, 1, (16,),
float32)`` -- observation ``3`` is converted to a 16-element vector
``[0, 0, 0, 1, 0, ..., 0]``. The ``MLPAgent`` then correctly creates
``Linear(16, 64)``.

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
