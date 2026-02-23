"""International Draughts (10x10) board configuration dialog.

This editor is for the International Draughts variant:
- 10x10 board (larger than American/Russian)
- Men CAN capture backward
- Kings are "flying" (can move multiple squares diagonally)
- Used in World Draughts Federation (FMJD) tournaments

Game ID: draughts/international_draughts
"""

from typing import List, Tuple, Optional

from PyQt6 import QtWidgets

from .base_checkers_editor import BaseCheckersConfigDialog


# Standard 10x10 draughts positions (50 dark squares)
# 20 pieces per side, 10 empty squares in middle
STANDARD_10X10 = "bbbbbbbbbbbbbbbbbbbb..........wwwwwwwwwwwwwwwwwwww"
EMPTY_10X10 = "." * 50

# Interesting positions for International Draughts
# Flying king on large board
FLYING_KING_10X10 = "b................B................................w..W"
# Opening position with one move made
FIRST_MOVE_10X10 = "bbbbbbbbbbbbbbbbbbb.b.........wwwwwwwwwwwwwwwwwwww"
# Endgame: kings only
ENDGAME_KINGS_10X10 = ".........B.B..............................W.W......."


class InternationalDraughtsConfigDialog(BaseCheckersConfigDialog):
    """Configuration dialog for International Draughts (10x10).

    International Draughts rules:
    - 10x10 board with 50 playable (dark) squares
    - 20 pieces per side at start (4 rows each)
    - Men CAN capture backward
    - Kings are "flying" - can move any number of squares diagonally
    - Kings can land on any square after a capture
    - Mandatory captures with maximum capture rule
    - "Turkish strike" rule: captured pieces are removed after the sequence
    """

    def __init__(
        self,
        initial_state: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(
            initial_state=initial_state,
            parent=parent,
            board_size=10,
            variant_name="International Draughts"
        )

    def _get_presets(self) -> List[Tuple[str, str]]:
        return [
            ("Standard", STANDARD_10X10),
            ("Empty", EMPTY_10X10),
            ("Flying King", FLYING_KING_10X10),
            ("First Move", FIRST_MOVE_10X10),
            ("Endgame Kings", ENDGAME_KINGS_10X10),
        ]


__all__ = ["InternationalDraughtsConfigDialog"]
