"""Russian Checkers (8x8) board configuration dialog.

This editor is for the Russian Checkers variant (Shashki):
- 8x8 board
- Men CAN capture backward
- Kings are "flying" (can move multiple squares diagonally)

Game ID: draughts/russian_checkers
"""

from typing import List, Tuple, Optional

from PyQt6 import QtWidgets

from .base_checkers_editor import BaseCheckersConfigDialog


# Standard 8x8 checkers positions (32 dark squares)
STANDARD_8X8 = "bbbbbbbbbbbb........wwwwwwwwwwww"
EMPTY_8X8 = "." * 32

# Interesting positions for Russian Checkers
# Flying king demonstration
FLYING_KING = "b.............B..............w.W"
# Backward capture setup
BACKWARD_CAPTURE = "....b.......w.b................."
# Complex multi-capture
MULTI_CAPTURE = "..b.b.b.....w..................w"


class RussianCheckersConfigDialog(BaseCheckersConfigDialog):
    """Configuration dialog for Russian Checkers (8x8).

    Russian Checkers (Shashki) rules:
    - 8x8 board with 32 playable (dark) squares
    - 12 pieces per side at start
    - Men CAN capture backward (unlike American)
    - Kings are "flying" - can move any number of squares diagonally
    - Kings can land on any square after a capture
    - Mandatory captures with maximum capture rule
    - Promotion happens immediately during a capture sequence
    """

    def __init__(
        self,
        initial_state: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(
            initial_state=initial_state,
            parent=parent,
            board_size=8,
            variant_name="Russian Checkers"
        )

    def _get_presets(self) -> List[Tuple[str, str]]:
        return [
            ("Standard", STANDARD_8X8),
            ("Empty", EMPTY_8X8),
            ("Flying King", FLYING_KING),
            ("Backward Capture", BACKWARD_CAPTURE),
            ("Multi-Capture", MULTI_CAPTURE),
        ]


__all__ = ["RussianCheckersConfigDialog"]
