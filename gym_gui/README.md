# Gym GUI Scaffold

Phase 0 of the Gymnasium Qt GUI brings in the core project skeleton, configuration
plumbing, and typed enums that future phases will build upon.

## Quick start

1. Create and activate the virtual environment if you have not already:

   ```bash
   source .venv/bin/activate
   ```

2. Install the foundational dependencies (we will add more as the project grows):

   ```bash
   pip install -U -r requirements.txt
   ```

   You can swap `pyqt6` for `pyside6`; because all imports go through
   [QtPy](https://github.com/spyder-ide/qtpy), the rest of the code does not need
   to change.

3. Copy the environment template and tweak values as needed:

   ```bash
   cp .env.example .env
   ```

4. Verify the scaffold loads configuration correctly:

   ```bash
   python -m gym_gui.app
   ```

   You should see the current settings printed to the console. If Qt bindings are
   installed, a stub window will also appear.

## Project layout

```
gym_gui/
   __init__.py
   README.md
   app.py                    # CLI entry-point for manual smoke tests
   agents/                   # Agent implementations and orchestration (future phases)
   cache/
      __init__.py
      memory.py               # In-memory memoization helpers
   config/
      __init__.py
      settings.py             # Loads configuration from .env and environment variables
   controllers/              # Qt controllers (future phases)
   core/
      __init__.py
      enums.py                # Shared enums for environments, control modes, etc.
      data_model/
         __init__.py           # Shared data schemas for observations and sessions
      factories/
         __init__.py           # Factory helpers for adapters, agents, renderers
   docs/
      learning_journal.md     # Notes captured as the project evolves
   core/
      adapters/
         __init__.py
         base.py               # Shared adapter contract
         toy_text.py           # FrozenLake/CliffWalking/Taxi adapters
         toy_text_demo.py      # CLI smoke harness
   env_adapters/
      __init__.py             # Re-export adapters for backwards compatibility
   logging_config/
      __init__.py
      logger.py               # Structured logging configuration
   rendering/                # Rendering helpers (future phases)
   runtime/
      cache/                  # Writable cache for derived assets
      data/                   # Episode exports, replays, and snapshots
         README.md
         toy_text/
            README.md         # Notes about generated toy-text snapshots
      log_output/             # Runtime log output (gitignored)
         README.md
   services/                 # Cross-cutting services (e.g., orchestrators)
   storage/
      __init__.py
   ui/                       # Qt Designer files and resources (future phases)
   utils/
      __init__.py
      qt.py                   # Qt-safe utility wrappers
```

## Next steps

- Implement the `EnvironmentAdapter` base class and toy-text adapters (Phase 1).
- Generate the first Qt Designer layout and wire a selection panel (Phase 2).
- Layer in the human, agent, and hybrid control orchestrators (Phase 3+).

Document your discoveries in `docs/learning_journal.md` as you progress so the project
remains a living tutorial as well as a functional application.
