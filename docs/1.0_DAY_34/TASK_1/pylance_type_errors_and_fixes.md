# TASK_1: Pylance Type Errors and Fixes

## Status: IN PROGRESS

This document catalogs Pylance type errors in `gym_gui/ui/main_window.py` and related files, with root cause analysis and recommended fixes.

---

## Table of Contents

1. [Issue 1: mujoco_mpc_worker Import Error](#issue-1-mujoco_mpc_worker-import-error)
2. [Issue 2: EnvironmentAdapter Missing Mouse Delta Methods](#issue-2-environmentadapter-missing-mouse-delta-methods)
3. [Issue 3: _submit_training_config Type Error](#issue-3-_submit_training_config-type-error)
4. [Issue 4: LOG_SEVERITY_OPTIONS Redundancy Question](#issue-4-log_severity_options-redundancy-question)
5. [Recommended Fixes](#recommended-fixes)

---

## Issue 1: mujoco_mpc_worker Import Error

### Error Message

```
Import "mujoco_mpc_worker" could not be resolved
Pylance(reportMissingImports)
```

### Location

**File:** `/gym_gui/ui/main_window.py:83`

```python
from mujoco_mpc_worker import get_launcher as get_mjpc_launcher
```

### Root Cause

The `mujoco_mpc_worker` package is located in `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/` but:

1. **Not installed in editable mode** - The package has a `pyproject.toml` but may not be installed via `pip install -e 3rd_party/mujoco_mpc_worker`
2. **Not in PYTHONPATH** - The `3rd_party/mujoco_mpc_worker` directory is not in Python's module search path
3. **Pylance cannot resolve** - Even if it works at runtime (via PYTHONPATH manipulation), Pylance's static analysis cannot find it

### Package Structure

```
3rd_party/mujoco_mpc_worker/
├── pyproject.toml                 # Defines "mosaic-mujoco-mpc" package
├── mujoco_mpc_worker/
│   ├── __init__.py                # Exports get_launcher, MuJoCoMPCConfig
│   ├── config.py
│   ├── launcher.py
│   └── cli.py
└── mujoco_mpc/                    # Vendored submodule
    └── python/mujoco_mpc/
```

### Resolution Options

#### Option A: Install in Editable Mode (Recommended)

```bash
pip install -e 3rd_party/mujoco_mpc_worker
```

This installs the package so both Python runtime AND Pylance can find it.

#### Option B: Conditional Import with Fallback

```python
# gym_gui/ui/main_window.py

try:
    from mujoco_mpc_worker import get_launcher as get_mjpc_launcher
    _MUJOCO_MPC_AVAILABLE = True
except ImportError:
    get_mjpc_launcher = None  # type: ignore[assignment,misc]
    _MUJOCO_MPC_AVAILABLE = False
```

#### Option C: Add to pyrightconfig.json / pyproject.toml

```toml
# pyproject.toml
[tool.pyright]
extraPaths = ["3rd_party/mujoco_mpc_worker"]
```

---

## Issue 2: EnvironmentAdapter Missing Mouse Delta Methods

### Error Messages

```
Cannot access attribute "has_mouse_delta_support" for class "EnvironmentAdapter[Unknown, Unknown]"
  Attribute "has_mouse_delta_support" is unknown
Pylance(reportAttributeAccessIssue)

Cannot access attribute "apply_mouse_delta" for class "EnvironmentAdapter[Unknown, Unknown]"
  Attribute "apply_mouse_delta" is unknown
Pylance(reportAttributeAccessIssue)
```

### Location

**File:** `/gym_gui/ui/main_window.py:921-930`

```python
adapter = self._session._adapter
has_delta_support = (
    adapter is not None
    and hasattr(adapter, "has_mouse_delta_support")
    and adapter.has_mouse_delta_support()  # <-- ERROR
)

if has_delta_support:
    def mouse_delta_callback(delta_x: float, delta_y: float) -> None:
        if adapter is not None and hasattr(adapter, "apply_mouse_delta"):
            adapter.apply_mouse_delta(delta_x, delta_y)  # <-- ERROR
```

### Root Cause

1. **EnvironmentAdapter base class** (`gym_gui/core/adapters/base.py`) does NOT define `has_mouse_delta_support()` or `apply_mouse_delta()`
2. **ViZDoomAdapter** (`gym_gui/core/adapters/vizdoom.py:216-234`) DOES define these methods
3. **Type narrowing fails** - Even with `hasattr()` checks, Pylance cannot narrow the type because `EnvironmentAdapter` doesn't declare these methods

### Current Implementation in ViZDoomAdapter

```python
# gym_gui/core/adapters/vizdoom.py:216-234

def apply_mouse_delta(self, delta_x: float, delta_y: float) -> None:
    """Queue mouse movement to be applied on the next step."""
    scaled_x = delta_x * self._config.mouse_sensitivity_x
    scaled_y = delta_y * self._config.mouse_sensitivity_y
    current_x, current_y = self._pending_mouse_delta
    self._pending_mouse_delta = (current_x + scaled_x, current_y + scaled_y)

def has_mouse_delta_support(self) -> bool:
    """Return True if delta buttons are configured for mouse control."""
    return self._turn_delta_index is not None
```

### Resolution Options

#### Option A: Add Optional Methods to Base Class (Recommended)

Add these methods to `EnvironmentAdapter` base class as no-op defaults:

```python
# gym_gui/core/adapters/base.py

class EnvironmentAdapter(ABC, Generic[ObservationT, ActionT], LogConstantMixin):
    # ... existing code ...

    def has_mouse_delta_support(self) -> bool:
        """Return True if the adapter supports FPS-style mouse delta control.

        Override in subclasses that support continuous mouse movement (e.g., ViZDoom).
        """
        return False

    def apply_mouse_delta(self, delta_x: float, delta_y: float) -> None:
        """Apply mouse movement delta for FPS-style control.

        Override in subclasses that support continuous mouse movement.

        Args:
            delta_x: Horizontal movement (positive = right, negative = left).
            delta_y: Vertical movement (positive = down, negative = up).
        """
        pass  # No-op by default
```

#### Option B: Protocol-Based Type Narrowing

Define a Protocol for mouse-delta-capable adapters:

```python
# gym_gui/core/adapters/protocols.py

from typing import Protocol

class MouseDeltaCapable(Protocol):
    def has_mouse_delta_support(self) -> bool: ...
    def apply_mouse_delta(self, delta_x: float, delta_y: float) -> None: ...


# Usage in main_window.py
from gym_gui.core.adapters.protocols import MouseDeltaCapable

adapter = self._session._adapter
if isinstance(adapter, MouseDeltaCapable) and adapter.has_mouse_delta_support():
    ...
```

#### Option C: Type Cast with Comment (Quick Fix)

```python
from typing import cast, Any

adapter = self._session._adapter
if adapter is not None and hasattr(adapter, "has_mouse_delta_support"):
    # ViZDoom adapter has mouse delta support
    delta_adapter = cast(Any, adapter)
    if delta_adapter.has_mouse_delta_support():
        ...
```

---

## Issue 3: _submit_training_config Type Error

### Error Message

```
Argument of type "object" cannot be assigned to parameter "config" of type "dict[Unknown, Unknown]"
  "object" is not assignable to "dict[Unknown, Unknown]"
Pylance(reportArgumentType)
```

### Location

**File:** `/gym_gui/ui/main_window.py:1079, 1104`

```python
self._submit_training_config(config)  # config is typed as 'object'
```

### Current Signature

```python
# Line 1391
def _submit_training_config(self, config: dict) -> None:
    """Submit a training configuration to the trainer daemon."""
```

### Root Cause

The `config` variable being passed comes from a dialog or form that returns `object` (not `dict`). This could be:

1. A form's `.build_config()` method returns `object` instead of `dict`
2. The variable is untyped or has an overly broad type annotation
3. The caller doesn't properly type the config before passing

### Resolution Options

#### Option A: Fix Return Type at Source

Ensure the config builder returns `dict[str, Any]`:

```python
# In the form/dialog class
def build_config(self) -> dict[str, Any]:
    return {...}
```

#### Option B: Add Type Assertion at Call Site

```python
if isinstance(config, dict):
    self._submit_training_config(config)
```

#### Option C: Improve Signature with TypedDict

```python
from typing import TypedDict

class TrainingConfig(TypedDict):
    run_id: str
    algo: str
    env_id: str
    # ... other fields

def _submit_training_config(self, config: TrainingConfig) -> None:
    ...
```

---

## Issue 4: LOG_SEVERITY_OPTIONS Redundancy Question

### Question

> Why do we have `LOG_SEVERITY_OPTIONS` when it comes from `LogConstantMixin` and `_LOGGER = logging.getLogger(__name__)`?

### Answer: No Redundancy - Different Purposes

These serve **completely different purposes**:

| Component | Purpose | Location |
|-----------|---------|----------|
| `_LOGGER` | Runtime logging via Python's logging module | `main_window.py:99` |
| `LogConstantMixin` | Structured logging with log constants (codes, tags) | `helpers.py:49-62` |
| `LOG_SEVERITY_OPTIONS` | **UI filter dropdown** for log viewer widget | `main_window.py:108-114` |

### LOG_SEVERITY_OPTIONS Definition

```python
# main_window.py:108-114
LOG_SEVERITY_OPTIONS: Dict[str, str | None] = {
    "All": None,
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
}
```

### Usage Context

```python
# main_window.py:351-356
severity_label = QtWidgets.QLabel("Severity:")
self._log_severity_filter = QtWidgets.QComboBox()
self._log_severity_filter.addItems(self.LOG_SEVERITY_OPTIONS.keys())
```

### Explanation

1. **`_LOGGER`**: Used to emit log messages to Python's logging system
2. **`LogConstantMixin.log_constant()`**: Used to emit structured logs with codes like `LOG_UI_MAINWINDOW_INFO`
3. **`LOG_SEVERITY_OPTIONS`**: A UI constant for populating a **QComboBox filter** in the log viewer panel

The `LOG_SEVERITY_OPTIONS` dict is **not for logging output** - it's for the **log viewer UI filter** that lets users filter displayed logs by severity level.

### Is This Good Design?

**Potential Improvement**: Move `LOG_SEVERITY_OPTIONS` to a central location:

```python
# gym_gui/logging_config/constants.py

SEVERITY_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

LOG_SEVERITY_OPTIONS: dict[str, str | None] = {
    "All": None,
    **{level: level for level in SEVERITY_LEVELS},
}
```

Then import in `main_window.py` and `info_log_container.py` (the two files that use it).

---

## Recommended Fixes

### Priority 1: Install mujoco_mpc_worker Package

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
pip install -e 3rd_party/mujoco_mpc_worker
```

### Priority 2: Add Mouse Delta Methods to Base Adapter

Edit `/gym_gui/core/adapters/base.py`:

```python
class EnvironmentAdapter(ABC, Generic[ObservationT, ActionT], LogConstantMixin):
    # ... existing code ...

    # ------------------------------------------------------------------
    # Mouse delta support (for FPS-style control)
    # ------------------------------------------------------------------

    def has_mouse_delta_support(self) -> bool:
        """Return True if the adapter supports FPS-style mouse delta control.

        Override in subclasses that support continuous mouse movement.
        Default implementation returns False.
        """
        return False

    def apply_mouse_delta(self, delta_x: float, delta_y: float) -> None:
        """Apply mouse movement delta for FPS-style control.

        Override in subclasses that support continuous mouse movement.
        Default implementation is a no-op.

        Args:
            delta_x: Horizontal movement (positive = right).
            delta_y: Vertical movement (positive = down).
        """
        pass
```

### Priority 3: Fix Config Type Annotation

Find where `config` is built and ensure it returns `dict[str, Any]`:

```python
# Example fix in form class
def build_config(self) -> dict[str, Any]:
    return {
        "run_id": self.run_id,
        "algo": self.algo,
        ...
    }
```

### Priority 4: Optional - Centralize LOG_SEVERITY_OPTIONS

```python
# gym_gui/ui/constants.py (new file or add to existing)

LOG_SEVERITY_OPTIONS: dict[str, str | None] = {
    "All": None,
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `gym_gui/core/adapters/base.py` | Add `has_mouse_delta_support()` and `apply_mouse_delta()` methods |
| `3rd_party/mujoco_mpc_worker/` | Run `pip install -e .` to install package |
| `gym_gui/ui/main_window.py` | Optional: wrap mujoco import in try/except |
| Form classes | Fix return type annotations to `dict[str, Any]` |

---

## Verification

After fixes, run:

```bash
# Check Pylance errors
cd /home/hamid/Desktop/Projects/GUI_BDI_RL
python -m pyright gym_gui/ui/main_window.py

# Or use VS Code's Pylance extension to see resolved errors
```
