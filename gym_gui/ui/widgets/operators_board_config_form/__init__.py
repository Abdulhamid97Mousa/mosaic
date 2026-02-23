"""Board game configuration dialogs.

This package provides extensible UI components for configuring custom
starting positions in board games (Chess, Go, Checkers, etc.).

Architecture:
- Strategy Pattern: Game-specific editors implement common interfaces
- Factory Pattern: BoardConfigDialogFactory creates appropriate dialogs
- Template Method: Base classes handle common UI, subclasses customize

Usage:
    from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory

    # Check if configuration is supported for a game
    if BoardConfigDialogFactory.supports("chess_v6"):
        dialog = BoardConfigDialogFactory.create(
            game_id="chess_v6",
            initial_state=current_fen,  # Optional
            parent=parent_widget
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            custom_state = dialog.get_state()
            # Store custom_state in operator config

Extensibility:
    To add support for a new game:

    1. Create a new editor module (e.g., checkers_editor.py):
        class CheckersBoardState(BoardState): ...
        class EditableCheckersBoard(EditableBoardWidget): ...
        class CheckersPieceTray(PieceTrayWidget): ...
        class CheckersConfigDialog(BoardConfigDialog): ...

    2. Register with the factory:
        BoardConfigDialogFactory.register("checkers", CheckersConfigDialog)

Supported Games:
    - chess_v6: PettingZoo Chess (FEN notation)
    - open_spiel/checkers: OpenSpiel Checkers 8x8 (American rules)
    - draughts/american_checkers: American Checkers 8x8
    - draughts/russian_checkers: Russian Checkers 8x8 (flying kings)
    - draughts/international_draughts: International Draughts 10x10

Planned:
    - go_v5: Go (SGF notation)
    - connect_four_v3: Connect Four
    - tictactoe_v3: Tic-Tac-Toe
"""

from .base import (
    BoardConfigDialog,
    EditableBoardWidget,
    PieceTrayWidget,
    BoardState,
    GamePiece,
)
from .factory import BoardConfigDialogFactory

# Game-specific editors depend on optional libraries (e.g. python-chess).
# Import them conditionally so the package loads even when those libs are absent.
# The factory already handles lazy registration with try/except.

try:
    from .chess_editor import ChessConfigDialog, ChessBoardState
except ImportError:
    pass

try:
    from .base_checkers_editor import (
        CheckersBoardState,
        EditableCheckersBoard,
        CheckersPieceTray,
        BaseCheckersConfigDialog,
    )
except ImportError:
    pass

try:
    from .checkers_editor import OpenSpielCheckersConfigDialog
except ImportError:
    pass

try:
    from .american_checkers_editor import AmericanCheckersConfigDialog
except ImportError:
    pass

try:
    from .russian_checkers_editor import RussianCheckersConfigDialog
except ImportError:
    pass

try:
    from .international_draughts_editor import InternationalDraughtsConfigDialog
except ImportError:
    pass

__all__ = [
    # Base classes (for extending)
    "BoardConfigDialog",
    "EditableBoardWidget",
    "PieceTrayWidget",
    "BoardState",
    "GamePiece",
    # Factory (main entry point)
    "BoardConfigDialogFactory",
    # Chess implementation (optional: requires python-chess)
    "ChessConfigDialog",
    "ChessBoardState",
    # Checkers/Draughts base classes (optional)
    "CheckersBoardState",
    "EditableCheckersBoard",
    "CheckersPieceTray",
    "BaseCheckersConfigDialog",
    # Checkers/Draughts variants (optional)
    "OpenSpielCheckersConfigDialog",
    "AmericanCheckersConfigDialog",
    "RussianCheckersConfigDialog",
    "InternationalDraughtsConfigDialog",
]
