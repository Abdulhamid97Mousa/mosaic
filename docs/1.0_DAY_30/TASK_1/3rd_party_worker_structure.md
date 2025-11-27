# 3rd Party Worker Structure Guide

This document explains the pattern used for integrating third-party libraries (like CleanRL) into the GUI_BDI_RL project as modular workers.

## Directory Structure

```
GUI_BDI_RL/
├── 3rd_party/
│   └── cleanrl_worker/              # Worker wrapper directory
│       ├── cleanrl/                 # Git submodule (VENDORED - DO NOT MODIFY)
│       ├── cleanrl_worker/          # Our modified/extended code
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── config.py
│       │   ├── algorithms/
│       │   └── eval/
│       └── pyproject.toml           # Package definition
├── requirements/
│   ├── base.txt                     # Core GUI + shared infrastructure
│   ├── cleanrl_worker.txt           # CleanRL worker dependencies
│   └── jason_worker.txt             # Jason worker dependencies
├── requirements.txt                 # Root entry point
├── pyproject.toml                   # Main project package definition
└── .gitmodules                      # Git submodule references
```

## The Pattern

### Step 1: Create Worker Directory
```bash
mkdir -p 3rd_party/<library>_worker/
```

### Step 2: Add Vendored Library as Git Submodule
```bash
cd 3rd_party/<library>_worker/
git submodule add https://github.com/org/library.git library
```

Update `.gitmodules`:
```ini
[submodule "library"]
  path = 3rd_party/<library>_worker/library
  url = https://github.com/org/library.git
```

### Step 3: Create Your Modified Code Directory
```bash
mkdir -p 3rd_party/<library>_worker/<library>_worker/
```

Copy and modify code from the vendored library into `<library>_worker/`.

### Step 4: Create `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mosaic-<library>"
version = "0.1.0"
description = "<Library> integration for MOSAIC BDI-RL framework"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = []  # Managed via root requirements.txt

[project.scripts]
<library>-worker = "<library>_worker.cli:main"

[tool.setuptools.packages]
find = {}

[tool.setuptools.package-dir]
# Expose vendored library packages
"<library>" = "<library>/<library>"
# Expose our wrapper
"<library>_worker" = "<library>_worker"
```

### Step 5: Add Requirements File
Create `requirements/<library>_worker.txt`:
```txt
# Include base requirements
-r base.txt

# Library-specific dependencies
torch>=2.0.0
tensorboard>=2.11.0
# ... other deps
```

## Why Both `pyproject.toml` AND `requirements.txt`?

They serve **different purposes**:

| Aspect | `pyproject.toml` | `requirements/<lib>_worker.txt` |
|--------|------------------|--------------------------------|
| **Purpose** | Package identity & structure | Dependency list |
| **Editable install** | Enables `pip install -e .` | Cannot make code importable |
| **Entry points** | Defines CLI commands | Cannot define CLI |
| **Package mapping** | Maps import paths to directories | N/A |
| **Submodule exposure** | Exposes vendored code as importable | N/A |

### The Key Insight

```
requirements/<lib>_worker.txt  →  "What external libraries do I need?"
pyproject.toml                 →  "How do I package MY code as a library?"
```

### Without `pyproject.toml`
```python
# FAILS - Python doesn't know where the package is
from cleanrl_worker.cli import main
from cleanrl_worker.algorithms.ppo_with_save import train
```

### With `pyproject.toml` + editable install
```python
# WORKS - setuptools registers the package
from cleanrl_worker.cli import main
from cleanrl.ppo import Agent  # Vendored cleanrl also works!
```

## Installation Flow

```bash
# 1. Install external dependencies
pip install -r requirements/cleanrl_worker.txt

# 2. Install worker package in editable mode
pip install -e 3rd_party/cleanrl_worker

# Or all at once from root:
pip install -r requirements.txt
pip install -e .
pip install -e 3rd_party/cleanrl_worker
```

## Benefits of This Pattern

1. **Vendored code is untouched** - Easy to update via `git submodule update`
2. **Clear separation** - Original vs modified code is obvious
3. **Centralized dependencies** - Root `requirements.txt` manages everything
4. **Modular installation** - Install only the workers you need
5. **Editable development** - Changes to `<lib>_worker/` reflect immediately
6. **CLI entry points** - Workers can expose command-line tools

## Example: CleanRL Worker

```
3rd_party/cleanrl_worker/
├── cleanrl/                    # Submodule: github.com/vwxyzjn/cleanrl
│   ├── cleanrl/
│   │   ├── ppo.py
│   │   ├── dqn.py
│   │   └── ...
│   └── cleanrl_utils/
├── cleanrl_worker/             # Our code
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # Configuration management
│   ├── runtime.py              # Runtime integration
│   ├── algorithms/
│   │   └── ppo_with_save.py    # Modified PPO with checkpointing
│   └── eval/
│       └── ppo.py              # Evaluation harness
├── pyproject.toml
└── tests/
```

### Import Examples
```python
# From vendored cleanrl (untouched)
from cleanrl.ppo import Agent, make_env

# From our wrapper
from cleanrl_worker.cli import main
from cleanrl_worker.algorithms.ppo_with_save import train
from cleanrl_worker.config import CleanRLConfig
```

### CLI Usage
```bash
# After: pip install -e 3rd_party/cleanrl_worker
cleanrl-worker --help
cleanrl-worker train --env CartPole-v1 --algo ppo
```

## Adding a New Worker

To add a new third-party library (e.g., `stable-baselines3`):

```bash
# 1. Create structure
mkdir -p 3rd_party/sb3_worker/sb3_worker

# 2. Add submodule
cd 3rd_party/sb3_worker
git submodule add https://github.com/DLR-RM/stable-baselines3.git sb3

# 3. Create pyproject.toml (see template above)

# 4. Create requirements/sb3_worker.txt
echo "-r base.txt" > ../../requirements/sb3_worker.txt
echo "stable-baselines3>=2.0.0" >> ../../requirements/sb3_worker.txt

# 5. Add to root requirements.txt (optional)
echo "-r requirements/sb3_worker.txt" >> ../../requirements.txt

# 6. Install
pip install -e 3rd_party/sb3_worker
```
