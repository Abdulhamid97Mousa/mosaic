# ViZDoom Requirements Structure

## Status: IMPLEMENTED

This document details the requirements management approach for ViZDoom in gym_gui.

## Requirements File Created

### `requirements/vizdoom.txt`

```txt
# =============================================================================
# ViZDoom Environment Dependencies
# =============================================================================
# ViZDoom enables Doom-based reinforcement learning with visual observations.
#
# USAGE:
#   pip install -r requirements/vizdoom.txt
#
# SYSTEM REQUIREMENTS:
#   Linux:   sudo apt install libopenal-dev
#   macOS:   brew install openal-soft
#   Windows: OpenAL included in ViZDoom wheel
# =============================================================================

# Include base dependencies
-r base.txt

# ViZDoom - Doom-based RL platform
vizdoom>=1.2.0,<2.0.0
```

## Installation Methods

### Method 1: Separate Requirements File

```bash
pip install -r requirements/vizdoom.txt
```

### Method 2: Optional Dependency (pyproject.toml)

```bash
pip install -e .[vizdoom]
```

### Method 3: Full Stack

```bash
pip install -r requirements.txt
pip install -r requirements/vizdoom.txt
```

## System Dependencies

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install libopenal-dev
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install openal-soft-devel
```

### macOS
```bash
brew install openal-soft
```

### Windows
OpenAL is bundled with the ViZDoom wheel. No additional installation needed.

## Version Considerations

| Version | Python Support | Notes |
|---------|---------------|-------|
| 1.2.x | 3.8-3.11 | Current stable, Gymnasium support |
| 1.3.x | 3.8-3.13 | Development, NumPy 2.x support |

**Pinned version**: `vizdoom>=1.2.0,<2.0.0`
- Stable Gymnasium integration
- Broad Python support
- Excludes potentially breaking 2.0 changes

## Verification

After installing ViZDoom, verify with:

```bash
python -c "import vizdoom; print(vizdoom.__version__)"
```

## Error Handling

When ViZDoom is not installed but user tries to use it:

```python
# In gym_gui/core/adapters/vizdoom.py
def _ensure_vizdoom():
    """Lazy import with helpful error message."""
    global vizdoom
    if vizdoom is None:
        try:
            import vizdoom as vzd
            vizdoom = vzd
        except ImportError as e:
            raise ImportError(
                "ViZDoom is not installed. Install with:\n"
                "  pip install -r requirements/vizdoom.txt\n"
                "or:\n"
                "  pip install vizdoom\n\n"
                "Linux users also need OpenAL:\n"
                "  sudo apt install libopenal-dev"
            ) from e
    return vizdoom
```

## Why Optional?

ViZDoom is kept as an optional dependency because:
- Large dependency (~50MB+ with game assets)
- Requires system-level OpenAL library
- Not all users need Doom environments
- Keeps base install lightweight

## Current Requirements Structure

```
requirements/
├── base.txt                 # Core GUI + shared infrastructure
├── cleanrl_worker.txt       # CleanRL worker dependencies
├── jason_worker.txt         # Jason BDI bridge dependencies
├── spade_bdi_worker.txt     # SPADE multi-agent framework
├── mujoco_mpc_worker.txt    # MuJoCo MPC dependencies
└── vizdoom.txt              # ViZDoom environment dependencies
```

## ViZDoom Vendor Location

The ViZDoom source/assets are located at:
```
3rd_party/vizdoom_worker/ViZDoom/
```

This is configured as a git submodule in `.gitmodules`.
