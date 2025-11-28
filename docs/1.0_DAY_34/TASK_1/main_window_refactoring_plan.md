# MainWindow Refactoring Plan

## Status: PROPOSED

This document outlines a refactoring plan for `/gym_gui/ui/main_window.py` (2533 lines) to improve maintainability, testability, and separation of concerns.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Identified Method Groups](#identified-method-groups)
3. [Proposed Extraction Targets](#proposed-extraction-targets)
4. [Refactoring Phases](#refactoring-phases)
5. [File Structure After Refactoring](#file-structure-after-refactoring)
6. [Migration Strategy](#migration-strategy)

---

## Current State Analysis

### File Statistics

| Metric | Value |
|--------|-------|
| Total Lines | 2,533 |
| Methods | ~75 |
| Imports | 83 lines |
| Class Attributes | 5 |
| Instance Variables | ~40 |

### Current Responsibilities (Too Many!)

The `MainWindow` class currently handles:

1. **UI Construction** - Building menus, toolbars, panels, layouts
2. **Game Configuration** - FrozenLake, Taxi, LunarLander, MiniGrid configs
3. **Session Management** - Start/stop/pause/continue game
4. **Training Submission** - Building and submitting training configs
5. **Training Monitoring** - Polling, watching, subscribing to runs
6. **Tab Management** - Creating/removing dynamic agent tabs
7. **MuJoCo MPC** - Launching/stopping MPC instances
8. **Logging** - Log record filtering and display
9. **Mouse Capture** - Configuring FPS-style mouse for ViZDoom
10. **Metadata Resolution** - Resolving run metadata from disk
11. **FastLane Tab Logic** - Determining when to open FastLane tabs

---

## Identified Method Groups

### Group 1: Game Configuration Handlers (~250 lines)

**Lines:** 613-848
**Methods:**
- `_on_slippery_toggled()`
- `_on_taxi_config_changed()`
- `_on_frozen_v2_config_changed()`
- `_on_cliff_config_changed()`
- `_on_lunar_config_changed()`
- `_on_car_config_changed()`
- `_on_bipedal_config_changed()`

**Target:** `gym_gui/ui/handlers/game_config_handlers.py`

---

### Group 2: MuJoCo MPC Handlers (~200 lines)

**Lines:** 1174-1348
**Methods:**
- `_on_mpc_launch_requested()`
- `_create_external_mpc_tab()`
- `_create_embedded_mpc_tab()`
- `_on_mpc_stop_instance()`
- `_on_mpc_stop_all_requested()`

**Target:** `gym_gui/ui/handlers/mpc_handlers.py`

---

### Group 3: Training Submission & Monitoring (~400 lines)

**Lines:** 1349-1700, 2205-2420
**Methods:**
- `_build_policy_evaluation_config()`
- `_submit_training_config()`
- `_on_training_submitted()`
- `_on_training_submit_failed()`
- `_poll_for_new_runs()`
- `_auto_subscribe_run()`
- `_auto_subscribe_run_main_thread()`
- `_start_run_watch()`
- `_shutdown_run_watch()`

**Target:** `gym_gui/ui/handlers/training_handlers.py`

---

### Group 4: Agent Tab Management (~200 lines)

**Lines:** 1643-1922
**Methods:**
- `_on_live_telemetry_tab_requested()`
- `_create_agent_tabs_for()`
- `_resolve_run_metadata()`
- `_maybe_open_fastlane_tab()`
- `_canonical_agent_id()`
- `_metadata_run_mode()`
- `_metadata_supports_fastlane()`
- `_backfill_run_metadata_from_disk()`

**Target:** `gym_gui/ui/handlers/agent_tab_handlers.py`

---

### Group 5: Logging Handlers (~100 lines)

**Lines:** 2437-2500
**Methods:**
- `_append_log_record()`
- `_on_log_filter_changed()`
- `_passes_filter()`
- `_format_log()`
- `_refresh_log_console()`

**Target:** `gym_gui/ui/handlers/log_handlers.py`

---

### Group 6: Mouse Capture Logic (~60 lines)

**Lines:** 896-952
**Methods:**
- `_configure_mouse_capture()`

**Target:** Move to `gym_gui/ui/widgets/render_tabs.py` or create `gym_gui/ui/handlers/input_handlers.py`

---

### Group 7: UI Constants

**Lines:** 108-129
**Attributes:**
- `LOG_SEVERITY_OPTIONS`
- `CONTROL_MODE_LABELS`
- `_HUMAN_INPUT_MODES`

**Target:** `gym_gui/ui/constants.py`

---

## Proposed Extraction Targets

### New Files to Create

```
gym_gui/ui/
├── constants.py                          # [NEW] UI constants
├── handlers/                             # [NEW DIRECTORY]
│   ├── __init__.py
│   ├── game_config_handlers.py           # Game config change handlers
│   ├── mpc_handlers.py                   # MuJoCo MPC handlers
│   ├── training_handlers.py              # Training submission/monitoring
│   ├── agent_tab_handlers.py             # Agent tab creation/management
│   ├── log_handlers.py                   # Log filtering and display
│   └── input_handlers.py                 # Mouse capture configuration
```

### Handler Pattern

Each handler module follows this pattern:

```python
# gym_gui/ui/handlers/mpc_handlers.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy import QtWidgets, QtCore

from gym_gui.logging_config.helpers import LogConstantMixin

if TYPE_CHECKING:
    from gym_gui.ui.main_window import MainWindow

_LOGGER = logging.getLogger(__name__)


class MPCHandlerMixin(LogConstantMixin):
    """Mixin providing MuJoCo MPC handling methods for MainWindow."""

    # Type hints for attributes used from MainWindow
    _mjpc_launcher: object
    _mpc_tabs: dict
    _render_tabs: object
    _control_panel: object
    _status_bar: object

    def _on_mpc_launch_requested(self: "MainWindow", display_mode: str) -> None:
        """Handle launch request for MuJoCo MPC."""
        ...

    def _create_external_mpc_tab(self: "MainWindow", process, instance_id: int, tab_name: str) -> None:
        ...

    # ... etc
```

### MainWindow After Refactoring

```python
# gym_gui/ui/main_window.py

from gym_gui.ui.handlers.game_config_handlers import GameConfigHandlerMixin
from gym_gui.ui.handlers.mpc_handlers import MPCHandlerMixin
from gym_gui.ui.handlers.training_handlers import TrainingHandlerMixin
from gym_gui.ui.handlers.agent_tab_handlers import AgentTabHandlerMixin
from gym_gui.ui.handlers.log_handlers import LogHandlerMixin


class MainWindow(
    QtWidgets.QMainWindow,
    LogConstantMixin,
    GameConfigHandlerMixin,
    MPCHandlerMixin,
    TrainingHandlerMixin,
    AgentTabHandlerMixin,
    LogHandlerMixin,
):
    """Primary window that orchestrates the Gym session."""

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # ... initialization code ...

    def _build_ui(self) -> None:
        # ... UI construction ...

    def _connect_signals(self) -> None:
        # ... signal connections ...

    # Core lifecycle methods remain here
```

---

## Refactoring Phases

### Phase 1: Extract UI Constants

**Files Changed:**
- Create `gym_gui/ui/constants.py`
- Update `main_window.py` to import from constants

**Estimated Reduction:** ~25 lines

```python
# gym_gui/ui/constants.py

from typing import Dict
from gym_gui.core.enums import ControlMode

LOG_SEVERITY_OPTIONS: Dict[str, str | None] = {
    "All": None,
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
}

CONTROL_MODE_LABELS: Dict[ControlMode, str] = {
    ControlMode.HUMAN_ONLY: "Human Only",
    ControlMode.AGENT_ONLY: "Agent Only",
    ControlMode.HYBRID_TURN_BASED: "Hybrid (Turn-Based)",
    ControlMode.HYBRID_HUMAN_AGENT: "Hybrid (Human + Agent)",
    ControlMode.MULTI_AGENT_COOP: "Multi-Agent (Cooperation)",
    ControlMode.MULTI_AGENT_COMPETITIVE: "Multi-Agent (Competition)",
}

HUMAN_INPUT_MODES = {
    ControlMode.HUMAN_ONLY,
    ControlMode.HYBRID_TURN_BASED,
    ControlMode.HYBRID_HUMAN_AGENT,
}
```

---

### Phase 2: Extract Game Config Handlers

**Files Changed:**
- Create `gym_gui/ui/handlers/__init__.py`
- Create `gym_gui/ui/handlers/game_config_handlers.py`
- Update `main_window.py`

**Estimated Reduction:** ~250 lines

---

### Phase 3: Extract MPC Handlers

**Files Changed:**
- Create `gym_gui/ui/handlers/mpc_handlers.py`
- Update `main_window.py`

**Estimated Reduction:** ~200 lines

---

### Phase 4: Extract Training Handlers

**Files Changed:**
- Create `gym_gui/ui/handlers/training_handlers.py`
- Update `main_window.py`

**Estimated Reduction:** ~400 lines

---

### Phase 5: Extract Agent Tab Handlers

**Files Changed:**
- Create `gym_gui/ui/handlers/agent_tab_handlers.py`
- Update `main_window.py`

**Estimated Reduction:** ~200 lines

---

### Phase 6: Extract Log Handlers

**Files Changed:**
- Create `gym_gui/ui/handlers/log_handlers.py`
- Update `main_window.py`

**Estimated Reduction:** ~100 lines

---

## File Structure After Refactoring

```
gym_gui/ui/
├── __init__.py
├── constants.py                          # [NEW] ~50 lines
├── main_window.py                        # REDUCED from 2533 to ~1200 lines
├── handlers/                             # [NEW DIRECTORY]
│   ├── __init__.py                       # Exports all handler mixins
│   ├── game_config_handlers.py           # ~250 lines
│   ├── mpc_handlers.py                   # ~200 lines
│   ├── training_handlers.py              # ~400 lines
│   ├── agent_tab_handlers.py             # ~200 lines
│   ├── log_handlers.py                   # ~100 lines
│   └── input_handlers.py                 # ~60 lines
├── forms/
├── indicators/
├── panels/
├── presenters/
├── qml/
├── renderers/
├── widgets/
└── workers/
```

### Expected Line Counts After Refactoring

| File | Before | After | Change |
|------|--------|-------|--------|
| `main_window.py` | 2,533 | ~1,200 | -1,333 |
| `constants.py` | 0 | ~50 | +50 |
| `game_config_handlers.py` | 0 | ~250 | +250 |
| `mpc_handlers.py` | 0 | ~200 | +200 |
| `training_handlers.py` | 0 | ~400 | +400 |
| `agent_tab_handlers.py` | 0 | ~200 | +200 |
| `log_handlers.py` | 0 | ~100 | +100 |
| `input_handlers.py` | 0 | ~60 | +60 |
| **Total** | 2,533 | ~2,460 | -73 (+ better organization) |

---

## Migration Strategy

### Step 1: Create Handler Directory

```bash
mkdir -p gym_gui/ui/handlers
touch gym_gui/ui/handlers/__init__.py
```

### Step 2: Extract One Handler at a Time

For each handler group:

1. Create new handler file with mixin class
2. Move methods from `main_window.py` to handler
3. Add mixin to `MainWindow` class inheritance
4. Remove methods from `main_window.py`
5. Run tests to verify functionality
6. Commit changes

### Step 3: Verify No Circular Imports

The handler mixins should only import:
- Standard library
- Third-party packages (Qt)
- Logging infrastructure
- Type hints (using `TYPE_CHECKING`)

They should NOT import from `main_window.py` (only use type hints).

### Step 4: Update Tests

Any tests that directly reference moved methods need updates to import from the new locations.

---

## Alternative Approach: Composition Over Mixins

If mixins prove problematic, consider composition:

```python
# gym_gui/ui/handlers/mpc_handler.py

class MPCHandler:
    def __init__(self, window: "MainWindow"):
        self._window = window
        self._mjpc_launcher = get_mjpc_launcher()
        self._mpc_tabs: dict[int, QtWidgets.QWidget] = {}

    def on_launch_requested(self, display_mode: str) -> None:
        ...


# main_window.py
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ...):
        ...
        self._mpc_handler = MPCHandler(self)

    # Delegate to handler
    def _on_mpc_launch_requested(self, display_mode: str) -> None:
        self._mpc_handler.on_launch_requested(display_mode)
```

This approach provides:
- Better testability (handlers can be unit tested)
- Clearer dependencies
- No mixin complexity

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Circular imports | Medium | High | Use TYPE_CHECKING, careful import ordering |
| Broken signals | Low | High | Test after each extraction |
| State inconsistency | Low | Medium | Keep all state in MainWindow, handlers access via self |
| Mixin method resolution | Low | Medium | Document MRO, test thoroughly |

---

## Recommended Order

1. **Phase 1: Constants** - Low risk, immediate benefit
2. **Phase 3: MPC Handlers** - Self-contained, good test case
3. **Phase 5: Log Handlers** - Self-contained
4. **Phase 2: Game Config Handlers** - Many similar methods
5. **Phase 4: Training Handlers** - Complex, but well-bounded
6. **Phase 5: Agent Tab Handlers** - Depends on training handlers

---

## Next Steps

1. Review and approve this plan
2. Create `gym_gui/ui/handlers/` directory structure
3. Begin with Phase 1 (constants extraction)
4. Proceed through phases with testing between each
