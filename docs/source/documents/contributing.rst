Contributing
============

We welcome contributions to MOSAIC! This guide will help you get started.

Development Setup
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
   cd MOSAIC

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate

   # Install development dependencies
   pip install -e ".[dev]"

   # Run tests
   pytest gym_gui/tests/

   # Run type checking
   pyright gym_gui/

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
5. Run type checking: ``pyright``
6. Submit a pull request

Adding a New Worker
-------------------

See :doc:`architecture/workers` for the worker architecture.

1. Create ``3rd_party/myworker/`` directory
2. Create ``requirements/myworker.txt``
3. Add to ``pyproject.toml`` optional dependencies
4. Implement the worker interface
5. Add tests

Documentation
-------------

Build the docs locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

   # Or live preview
   make livehtml
