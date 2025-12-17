# Optional Dependencies Refactor Report

## Problem Description

The GUI application had hardcoded error messages and dependency checks for optional components like Godot and MuJoCo MPC within the UI handlers (`godot.py` and `mpc.py`). This approach had several drawbacks:

1.  **Tight Coupling**: The UI logic was tightly coupled with the specific error strings and installation instructions.
2.  **Maintenance Difficulty**: Updating installation instructions or error messages required modifying the core logic files.
3.  **Inconsistency**: There was no centralized place to manage user-facing messages, leading to potential inconsistencies in tone and format.
4.  **Hardcoded Strings**: Hardcoded strings in the code make internationalization (i18n) and localization (l10n) difficult.

## Solution Implemented

To address these issues, we have refactored the code to centralize these messages into a dedicated constants file.

### 1. Created `gym_gui/validations/log_constants.py`

A new file `gym_gui/validations/log_constants.py` was created to store all user-facing log messages and error notifications related to optional dependencies.

```python
# Godot Worker Messages
GODOT_NOT_INSTALLED_TITLE = "Godot Not Available"
GODOT_NOT_INSTALLED_MSG = ...

# MuJoCo MPC Worker Messages
MJPC_NOT_INSTALLED_TITLE = "MJPC Not Available"
MJPC_NOT_INSTALLED_MSG = ...
```

### 2. Refactored UI Handlers

The `GodotHandler` in `gym_gui/ui/handlers/features/godot.py` and `MPCHandler` in `gym_gui/ui/handlers/features/mpc.py` were updated to import and use these constants instead of hardcoded strings.

**Before:**
```python
QtWidgets.QMessageBox.warning(
    None,
    "Godot Not Available",
    "Godot worker is not installed.\n\n..."
)
```

**After:**
```python
from gym_gui.validations.log_constants import GODOT_NOT_INSTALLED_TITLE, GODOT_NOT_INSTALLED_MSG

QtWidgets.QMessageBox.warning(
    None,
    GODOT_NOT_INSTALLED_TITLE,
    GODOT_NOT_INSTALLED_MSG,
)
```

## Benefits

*   **Separation of Concerns**: UI logic is now separated from the content of the messages.
*   **Easier Maintenance**: Messages can be updated in a single file without touching the logic.
*   **Cleaner Code**: The handler files are now cleaner and focused on the logic of handling the features.
*   **Reusability**: These constants can be reused in other parts of the application if needed.
