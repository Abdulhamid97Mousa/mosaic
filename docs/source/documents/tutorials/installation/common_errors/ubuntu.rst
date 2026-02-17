Ubuntu-Specific Errors
======================

These errors are specific to running MOSAIC on **native Ubuntu** (bare-metal
or VM, not WSL).  For shared errors that apply to both platforms, see the
parent :doc:`index` page.

-----

CUDA / GPU not detected on native Ubuntu
-----------------------------------------

**Error:**

.. code-block:: python

   >>> import torch
   >>> torch.cuda.is_available()
   False

**Cause:** PyTorch was installed without CUDA support, or the NVIDIA driver
is missing.

**Fix:**

.. code-block:: bash

   # 1. Verify the NVIDIA driver is installed
   nvidia-smi
   # Should show your GPU (e.g., NVIDIA GeForce RTX 4090 Laptop GPU)

   # 2. If nvidia-smi is not found, install the driver
   sudo apt-get install -y nvidia-driver-535
   # Then reboot

   # 3. Reinstall PyTorch with CUDA support
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # 4. Verify
   python -c "import torch; print(torch.cuda.is_available())"  # True

NoneType takes no arguments (QWebEnginePage)
--------------------------------------------

**Error:**

.. code-block:: text

   File "/home/.../gym_gui/ui/widgets/filtered_web_engine.py", line 34, in <module>
       class FilteredWebEnginePage(QWebEnginePage):
   TypeError: NoneType takes no arguments

**Cause:** The ``PyQt6-WebEngine`` package is not installed, or it fails to
import due to missing system libraries. MOSAIC uses ``QWebEnginePage`` for the
TensorBoard artifact viewer. When the import fails, ``QWebEnginePage`` resolves
to ``None`` and the class definition crashes.

To see the **real** underlying error, run:

.. code-block:: bash

   python -c "from PyQt6.QtWebEngineCore import QWebEnginePage"

Common underlying errors:

- ``libXcomposite.so.1: cannot open shared object file`` -- missing ``libxcomposite1``
- ``libXdamage.so.1: cannot open shared object file`` -- missing ``libxdamage1``
- ``libxkbfile.so.1: cannot open shared object file`` -- missing ``libxkbfile1``
- ``libsmime3.so: cannot open shared object file`` -- missing ``libnss3``

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

Headless server -- no display available
---------------------------------------

**Error:**

.. code-block:: text

   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though
   it was found.
   This application failed to start because no Qt platform plugin could be initialized.

**Cause:** On headless servers (no monitor, no X11), Qt cannot find a display
to render to.

**Fix:**

.. code-block:: bash

   # Install virtual framebuffer
   sudo apt-get install -y xvfb

   # Run with virtual display
   xvfb-run python -m gym_gui

   # Or set DISPLAY manually if you have a remote X session
   export DISPLAY=:1

Overcooked -- Python version incompatibility
---------------------------------------------

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

MuJoCo rendering -- "GLEW initialization error"
------------------------------------------------

**Error:**

.. code-block:: text

   mujoco.FatalError: an OpenGL platform library has not been loaded into this process

**Cause:** MuJoCo needs a rendering backend. On native Ubuntu with a GPU, EGL
is the recommended backend.

**Fix:** Set the rendering backend in your ``.env`` file:

.. code-block:: bash

   MUJOCO_GL=egl

If you see ``libEGL.so.1: cannot open shared object file``:

.. code-block:: bash

   sudo apt-get install -y libegl1

-----

Quick Reference: Ubuntu System Dependencies
--------------------------------------------

.. code-block:: bash

   # Install all Ubuntu-specific dependencies at once
   sudo apt-get update
   sudo apt-get install -y \
       build-essential cmake swig flex bison libbz2-dev \
       libopenmpi-dev \
       libegl1 libgl1 libopengl0 libxkbcommon0 \
       libnss3 libnspr4 libasound2 \
       libxcomposite1 libxdamage1 libxkbfile1 \
       stockfish \
       xvfb
