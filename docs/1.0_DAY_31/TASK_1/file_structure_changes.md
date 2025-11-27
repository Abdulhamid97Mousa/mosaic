# ViZDoom Integration - File Structure Changes

## Status: IMPLEMENTED

This document details all files created and modified for the ViZDoom integration.

## New Files Created

### 1. Requirements

```
requirements/
└── vizdoom.txt                     # ViZDoom dependencies
```

### 2. Core Adapter

```
gym_gui/core/adapters/
└── vizdoom.py                      # ViZDoom adapter implementation
                                    # - ViZDoomConfig dataclass
                                    # - ViZDoomAdapter base class
                                    # - 10 scenario-specific adapters
                                    # - Delta button support (360° mouse)
                                    # - NOOP action handling
```

### 3. Game Documentation

```
gym_gui/game_docs/ViZDoom/
├── __init__.py                     # Aggregator exports
├── basic.py                        # Basic, Predict Position, Take Cover docs
├── combat.py                       # Defend Center, Defend Line, Deadly Corridor docs
├── navigation.py                   # Health Gathering, My Way Home docs
├── multiplayer.py                  # Deathmatch documentation
└── controls.py                     # Shared mouse capture instructions
```

### 4. UI Configuration

```
gym_gui/ui/environments/single_agent_env/vizdoom/
├── __init__.py                     # Exports VIZDOOM_GAME_IDS, build_vizdoom_controls
└── config_panel.py                 # Configuration UI builders
```

### 5. Interaction Controller

```
gym_gui/controllers/
└── interaction.py                  # Contains ViZDoomInteractionController
                                    # (also AleInteractionController, Box2D, TurnBased)
```

## Files Modified

### 1. `gym_gui/core/enums.py`

**Changes:**
- Added `EnvironmentFamily.VIZDOOM`
- Added 10 `GameId` entries for ViZDoom scenarios
- Updated `ENVIRONMENT_FAMILY_BY_GAME` mapping
- Updated `DEFAULT_RENDER_MODES` (all → RGB_ARRAY)
- Updated `DEFAULT_CONTROL_MODES`

### 2. `gym_gui/core/adapters/__init__.py`

**Changes:**
- Added conditional import for ViZDoom adapters
- Exports `ViZDoomAdapter`, `ViZDoomConfig`, `VIZDOOM_ADAPTERS`

### 3. `gym_gui/core/factories/adapters.py`

**Changes:**
- Imports `VIZDOOM_ADAPTERS` (conditional)
- Registers ViZDoom adapters in `_registry()`
- Handles `ViZDoomConfig` in `create_adapter()`

### 4. `gym_gui/config/game_config_builder.py`

**Changes:**
- Added ViZDoom import (conditional)
- Added `_VIZDOOM_GAME_IDS` tuple
- Added ViZDoom config building in `build_config()`

### 5. `gym_gui/game_docs/__init__.py`

**Changes:**
- Conditional import for ViZDoom documentation
- Updated `GAME_INFO` dict with ViZDoom entries

### 6. `gym_gui/ui/environments/single_agent_env/__init__.py`

**Changes:**
- Conditional import for ViZDoom UI components
- Exports `VIZDOOM_GAME_IDS`, `build_vizdoom_controls`

### 7. `gym_gui/ui/widgets/control_panel.py`

**Changes:**
- Added ViZDoom imports
- Added ViZDoom conditional in `_refresh_game_config_ui()`
- Added `_on_vizdoom_config_changed()` callback
- Fixed config refresh on game combo rebuild

### 8. `gym_gui/ui/main_window.py`

**Changes:**
- Added delta mode detection for mouse capture
- Wires `apply_mouse_delta()` callback for ViZDoom

### 9. `gym_gui/rendering/strategies/rgb.py`

**Changes:**
- Added 2D mouse delta support (`set_mouse_delta_callback`)
- Added delta scale configuration
- Updated `mouseMoveEvent()` for X+Y delta calculation
- Added cursor grabbing and re-centering

### 10. `gym_gui/ui/widgets/render_tabs.py`

**Changes:**
- Extended `configure_mouse_capture()` for delta callback
- Added `delta_callback` and `delta_scale` parameters

### 11. `gym_gui/controllers/session.py`

**Changes:**
- Import `ViZDoomInteractionController`
- Register in `_create_interaction_controller()`
- Updated `_idle_step()` to bypass awaiting_human for ViZDoom

### 12. `gym_gui/controllers/human_input.py`

**Changes:**
- Added `get_vizdoom_mouse_turn_actions()` function
- ViZDoom-specific keyboard shortcut mappings

## Directory Structure After Implementation

```
gym_gui/
├── config/
│   └── game_config_builder.py      # [MODIFIED] +ViZDoom config building
├── core/
│   ├── enums.py                    # [MODIFIED] +VIZDOOM family, +GameIds
│   ├── adapters/
│   │   ├── __init__.py             # [MODIFIED] +ViZDoom exports
│   │   ├── ale.py                  # (existing)
│   │   ├── minigrid.py             # (existing)
│   │   └── vizdoom.py              # [NEW] Full adapter implementation
│   └── factories/
│       └── adapters.py             # [MODIFIED] +VIZDOOM_ADAPTERS
├── controllers/
│   ├── session.py                  # [MODIFIED] +ViZDoomInteractionController
│   ├── human_input.py              # [MODIFIED] +ViZDoom shortcuts
│   └── interaction.py              # [MODIFIED] +ViZDoomInteractionController
├── game_docs/
│   ├── __init__.py                 # [MODIFIED] +ViZDoom docs
│   ├── ALE/                        # (existing)
│   ├── MiniGrid/                   # (existing)
│   └── ViZDoom/                    # [NEW DIRECTORY]
│       ├── __init__.py
│       ├── basic.py
│       ├── combat.py
│       ├── controls.py
│       ├── navigation.py
│       └── multiplayer.py
├── rendering/
│   └── strategies/
│       └── rgb.py                  # [MODIFIED] +360° mouse capture
└── ui/
    ├── main_window.py              # [MODIFIED] +delta mode wiring
    ├── widgets/
    │   ├── control_panel.py        # [MODIFIED] +ViZDoom config panel
    │   └── render_tabs.py          # [MODIFIED] +delta callback
    └── environments/
        └── single_agent_env/
            ├── __init__.py         # [MODIFIED] +ViZDoom exports
            ├── ale/                # (existing)
            ├── minigrid/           # (existing)
            └── vizdoom/            # [NEW DIRECTORY]
                ├── __init__.py
                └── config_panel.py

requirements/
├── base.txt                        # (no changes)
├── cleanrl_worker.txt              # (no changes)
├── jason_worker.txt                # (no changes)
└── vizdoom.txt                     # [NEW]
```

## Import Dependency Graph

```
vizdoom (pip package)
    ↓
gym_gui/core/adapters/vizdoom.py
    ↓
gym_gui/core/adapters/__init__.py
    ↓
gym_gui/core/factories/adapters.py ← gym_gui/config/game_config_builder.py
    ↓
gym_gui/controllers/session.py
    ↓
gym_gui/ui/main_window.py

gym_gui/game_docs/ViZDoom/*
    ↓
gym_gui/game_docs/__init__.py
    ↓
(UI components that display game info)

gym_gui/ui/environments/single_agent_env/vizdoom/*
    ↓
gym_gui/ui/environments/single_agent_env/__init__.py
    ↓
gym_gui/ui/widgets/control_panel.py
```

## Graceful Degradation

All ViZDoom imports are conditional with try/except:
- If vizdoom not installed → empty dictionaries returned
- No crashes when ViZDoom unavailable
- ViZDoom games simply don't appear in UI
- Clear error message when user tries to load unavailable game:
  ```
  ImportError: ViZDoom is not installed. Install with:
    pip install -r requirements/vizdoom.txt
  ```

## ViZDoom Vendor Location

The ViZDoom source repository is located at:
```
3rd_party/vizdoom_worker/ViZDoom/
```

This is configured as a git submodule in `.gitmodules`:
```ini
[submodule "ViZDoom"]
  path = 3rd_party/vizdoom_worker/ViZDoom
  url = https://github.com/Farama-Foundation/ViZDoom.git
```

The `_vizdoom.ini` configuration file at project root references this path for WAD and asset lookup.
