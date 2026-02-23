# Qt Configuration Directory

This directory stores Qt configuration files for the GUI_BDI_RL application.

## Purpose

Configuration files in this directory are managed by `gym_gui.config.qt_config.ProjectQtConfig` and contain persistent settings for various GUI components, such as:

- Window geometry and state
- User preferences
- Component-specific settings

## Files

Configuration files follow the naming convention: `{Organization}_{Application}.ini`

Example:
- `GymGUI_ControlPanelWidget.ini` - Control panel widget settings

## Management

Configuration files are automatically created and managed by the application. Users should not manually edit these files.

To reset all configuration:
```python
from gym_gui.config.qt_config import ProjectQtConfig
ProjectQtConfig.reset_all()
```

## Why This Directory?

By default, Qt stores configuration in the user's home directory (`~/.config/`), which:
- Pollutes the user's home directory
- Makes the application less portable
- Creates privacy concerns

This project centralizes all Qt configuration in the `var/config/` directory to:
- Keep the project self-contained
- Improve portability
- Maintain clean separation of concerns
- Make testing easier

## Gitignore

All `.ini` files in this directory are ignored by git (see `.gitignore`), ensuring that user-specific configuration is not committed to the repository.

