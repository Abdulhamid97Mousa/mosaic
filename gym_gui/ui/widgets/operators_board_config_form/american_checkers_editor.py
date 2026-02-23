"""American Checkers (8x8) board configuration dialog.

This editor is for the American Checkers variant (also known as English Draughts):
- 8x8 board
- Men cannot capture backward
- Kings move one square diagonally (non-flying)

Game ID: draughts/american_checkers
"""

from typing import List, Tuple, Optional

from PyQt6 import QtWidgets

from .base_checkers_editor import BaseCheckersConfigDialog


# Standard 8x8 checkers positions (32 dark squares)
STANDARD_8X8 = "bbbbbbbbbbbb........wwwwwwwwwwww"
EMPTY_8X8 = "." * 32

# Interesting positions for American Checkers
# First move advantage position
FIRST_MOVE = "bbbbbbbbbbb.........wwwwwwwwwwww"
# Classic endgame: King chase
KING_CHASE = "...........B..................W."
# Double corner position
DOUBLE_CORNER = "..........BB..................WW"


class AmericanCheckersConfigDialog(BaseCheckersConfigDialog):
    """Configuration dialog for American Checkers (8x8).

    American Checkers (English Draughts) rules:
    - 8x8 board with 32 playable (dark) squares
    - 12 pieces per side at start
    - Men can only move and capture forward (diagonally)
    - Kings can move one square in any diagonal direction
    - Mandatory captures (must take if able)
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
            variant_name="American Checkers"
        )

    def _get_presets(self) -> List[Tuple[str, str]]:
        return [
            ("Standard", STANDARD_8X8),
            ("Empty", EMPTY_8X8),
            ("First Move", FIRST_MOVE),
            ("King Chase", KING_CHASE),
            ("Double Corner", DOUBLE_CORNER),
        ]


__all__ = ["AmericanCheckersConfigDialog"]
