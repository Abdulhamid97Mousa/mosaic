WSL-Specific Errors
===================

These errors are specific to running MOSAIC on **Windows Subsystem for Linux
(WSL 2)**.  For shared errors that apply to both platforms, see the parent
:doc:`index` page.

-----

Broken ``.venv`` (I/O errors on symlinks)
-----------------------------------------

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

Always create and use the venv from a **native WSL terminal** (not PowerShell
or Git Bash). See :doc:`../wsl` for details.

Trainer daemon -- "libGL.so.1: cannot open shared object file"
--------------------------------------------------------------

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

   trainer_daemon -> dispatcher -> subprocess_validation -> validations_ui -> PyQt6

On native Ubuntu **desktop** these libraries come pre-installed, but on
**WSL** minimal installs they are missing.

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

QRhiGles2 -- "Failed to create temporary context"
--------------------------------------------------

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

GUI launches but text shows as squares or is invisible
------------------------------------------------------

**Symptom:** The MOSAIC window appears but all labels, buttons, and menus show
as squares or are completely blank:

.. image:: /_static/figures/wsl_missing_fonts.jpg
   :alt: MOSAIC GUI with missing fonts -- all text rendered as squares
   :width: 100%

**Cause:** This happens when either:

1. The ``fontconfig`` package is not installed -- Qt uses fontconfig to discover
   system fonts, and without it, Qt cannot find any fonts at all.
2. The font cache is stale -- fonts were installed but ``fc-cache`` was not run,
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

PyQt6 -- "Could not load the Qt platform plugin"
-------------------------------------------------

**Error:**

.. code-block:: text

   qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load
   the Qt xcb platform plugin.
   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though
   it was found.
   This application failed to start because no Qt platform plugin could be initialized.

**Cause:** PyQt6 >= 6.5 requires ``libxcb-cursor0`` for the X11 (xcb) platform
plugin. This library is not installed by default on minimal WSL distributions.

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

CUDA / GPU not detected on WSL
------------------------------

**Error:**

.. code-block:: python

   >>> import torch
   >>> torch.cuda.is_available()
   False

**Cause:** On WSL, CUDA uses the **Windows** NVIDIA driver. Do **not**
install the Linux NVIDIA driver inside WSL.

**Fix:**

.. code-block:: bash

   # Verify the Windows driver is visible inside WSL
   nvidia-smi

   # If nvidia-smi works but torch doesn't detect CUDA, reinstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121

If ``nvidia-smi`` is not found, update your Windows NVIDIA driver from
`nvidia.com/drivers <https://www.nvidia.com/Download/index.aspx>`_.

-----

Quick Reference: WSL System Dependencies
-----------------------------------------

.. code-block:: bash

   # Install all WSL-specific dependencies at once
   sudo apt-get update
   sudo apt-get install -y \
       libegl1 libgl1 libopengl0 libxkbcommon0 \
       libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 \
       libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
       libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
       libxcb-xfixes0 libxcb-xinerama0 libxcb-cursor0 \
       libnss3 libnspr4 libasound2 \
       libxcomposite1 libxdamage1 libxkbfile1 \
       fontconfig fonts-dejavu-extra fonts-liberation fonts-noto
