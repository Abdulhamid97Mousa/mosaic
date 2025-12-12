Installation
============

This guide covers installing MOSAIC and its dependencies.

Requirements
------------

- Python 3.10 or higher
- PyQt6 for the visual interface
- PyTorch 2.0+ for neural network backends

Quick Install
-------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
   cd MOSAIC

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or: .venv\Scripts\activate  # Windows

   # Install base dependencies
   pip install -e .

Optional Dependencies
---------------------

Install specific workers based on your needs:

.. tabs::

   .. tab:: CleanRL

      .. code-block:: bash

         pip install -e ".[cleanrl]"

   .. tab:: RLlib

      .. code-block:: bash

         pip install -e ".[ray-rllib]"

   .. tab:: XuanCe

      .. code-block:: bash

         pip install -e ".[xuance]"

   .. tab:: Full

      .. code-block:: bash

         pip install -e ".[full]"

Verifying Installation
----------------------

.. code-block:: bash

   # Run the visual interface
   python -m gym_gui

   # Or run tests
   pytest gym_gui/tests/
