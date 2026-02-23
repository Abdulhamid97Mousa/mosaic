Contributing
============

We welcome contributions to MOSAIC! This guide will help you get started.

Development Setup
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Abdulhamid97Mousa/mosaic.git
   cd mosaic

   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Install core GUI + development dependencies
   pip install -e ".[dev]"

   # Run tests
   pytest gym_gui/tests/

Code Style
----------

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write docstrings in Google style
- Maximum line length: 100 characters

Pull Request Process
--------------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make your changes
4. Run tests: ``pytest``
5. Submit a pull request

Adding a New Worker
-------------------

See :doc:`architecture/workers/integrated_workers/index` for the worker
architecture and existing worker implementations.

1. Create ``3rd_party/myworker/`` directory with ``pyproject.toml``
2. Implement the worker with an ``InteractiveRuntime`` class
3. Add a CLI entry point (``myworker-worker``)
4. Support the JSON protocol: ``reset``, ``step``, ``stop`` commands
5. Add tests
6. Create documentation under ``docs/source/documents/architecture/workers/integrated_workers/``

Adding a New Environment
------------------------

1. Create an adapter in ``gym_gui/core/adapters/``
2. Add a game config in ``gym_gui/config/game_configs.py``
3. Add key combination resolver in ``gym_gui/controllers/``
4. Register in ``gym_gui/config/game_config_builder.py``
5. Add documentation under ``docs/source/documents/environments/``

Documentation
-------------

Build the docs locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   sphinx-build -b html source build/html

   # Serve locally
   cd build/html && python3 -m http.server 8765
