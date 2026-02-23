"""Grid environment configuration dialogs for Operators.

This package provides extensible UI components for configuring custom
starting states in grid environments (MiniGrid, BabyAI, MultiGrid, etc.).

Architecture:
- Strategy Pattern: Game-specific editors implement common interfaces
- Factory Pattern: GridConfigDialogFactory creates appropriate dialogs
- Template Method: Base classes handle common UI, subclasses customize

Usage:
    from gym_gui.ui.widgets.operators_grid_config_form import GridConfigDialogFactory

    # Check if configuration is supported for a game
    if GridConfigDialogFactory.supports("MiniGrid-Empty-5x5-v0"):
        dialog = GridConfigDialogFactory.create(
            env_id="MiniGrid-Empty-5x5-v0",
            initial_state=current_state,  # Optional
            parent=parent_widget
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            custom_state = dialog.get_state()
            # Store custom_state in operator config

Extensibility:
    To add support for a new grid environment:

    1. Create a new editor module (e.g., crafter_editor.py):
        class CrafterGridState(GridState): ...
        class CrafterGridEditor(GridEditorWidget): ...
        class CrafterConfigDialog(GridConfigDialog): ...

    2. Register with the factory:
        GridConfigDialogFactory.register("Crafter", CrafterConfigDialog)

Supported Environments:
    - MiniGrid-* : Gymnasium MiniGrid environments
    - BabyAI-* : BabyAI language-grounded environments (all 6 colors, mission text)
    - MultiGrid-* : Multi-agent grid environments (Soccer, Collect)
    - MeltingPot-* : Multi-agent social scenarios

Planned:
    - Crafter : Crafting/survival grid environment
"""

from .base import (
    GridConfigDialog,
    GridEditorWidget,
    GridObjectTray,
    GridState,
    GridCell,
    GridObject,
)
from .factory import GridConfigDialogFactory
from .babyai_editor import BabyAIConfigDialog, BabyAIState

__all__ = [
    # Base classes (for extending)
    "GridConfigDialog",
    "GridEditorWidget",
    "GridObjectTray",
    "GridState",
    "GridCell",
    "GridObject",
    # Factory (main entry point)
    "GridConfigDialogFactory",
    # BabyAI editor
    "BabyAIConfigDialog",
    "BabyAIState",
]
