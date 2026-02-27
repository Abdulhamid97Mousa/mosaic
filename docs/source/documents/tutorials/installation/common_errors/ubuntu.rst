Ubuntu-Specific Errors
======================

These errors are specific to running MOSAIC on **Ubuntu** (native Linux).
For shared errors that apply to both platforms, see the parent
:doc:`index` page.

-----

box2d-py -- "ModuleNotFoundError: No module named 'swig'"
----------------------------------------------------------

**Error (when installing box2d environments):**

.. code-block:: text

   Building wheel for box2d-py (pyproject.toml) did not run successfully.
   swigging Box2D/Box2D.i to Box2D/Box2D_wrap.cpp
   swig -python -c++ -IBox2D -small -O -includeall -ignoremissing -w201 ...
   Traceback (most recent call last):
     File "/home/hamid/.local/bin/swig", line 5, in <module>
       from swig import swig
   ModuleNotFoundError: No module named 'swig'
   error: command '/home/hamid/.local/bin/swig' failed with exit code 1

**Cause:** A broken SWIG Python script exists at ``~/.local/bin/swig`` that tries
to import a non-existent Python module called ``swig``. This script shadows the
proper system SWIG binary at ``/usr/bin/swig``.

The ``box2d-py`` package requires SWIG (Simplified Wrapper and Interface Generator)
to build its C++ extensions. When pip tries to build the wheel, it finds the broken
script first because ``~/.local/bin`` appears earlier in ``PATH`` than ``/usr/bin``.

**Fix:**

.. code-block:: bash

   # Remove or rename the broken SWIG script
   mv ~/.local/bin/swig ~/.local/bin/swig.broken

   # Verify system SWIG is now being used
   which swig    # Should show /usr/bin/swig or /bin/swig
   swig -version # Should show SWIG Version 4.x

   # Retry the installation
   pip install -e ".[box2d]"

If system SWIG is not installed, install it first:

.. code-block:: bash

   sudo apt-get install -y swig

.. note::

   This issue occurs when a Python package (possibly installed via pip) created
   a wrapper script at ``~/.local/bin/swig`` instead of using the system SWIG.
   The wrapper script is non-functional and must be removed to allow the proper
   SWIG binary to be used.
