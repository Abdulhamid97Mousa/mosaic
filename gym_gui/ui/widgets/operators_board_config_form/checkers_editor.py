"""OpenSpiel Checkers (8x8) board configuration dialog.

This editor is for the standard OpenSpiel checkers implementation
which uses American Checkers rules on an 8x8 board.

Game ID: open_spiel/checkers
"""

from typing import List, Tuple, Optional

from PyQt6 import QtWidgets

from .base_checkers_editor import BaseCheckersConfigDialog


# Standard 8x8 checkers positions (32 dark squares)
STANDARD_8X8 = "bbbbbbbbbbbb........wwwwwwwwwwww"
EMPTY_8X8 = "." * 32

# Some interesting endgame positions
ENDGAME_2V1 = "..........B..........w.......w."  # 2 white vs 1 black king
ENDGAME_KINGS = ".........B.B.........W.W......."  # 2 kings each


class OpenSpielCheckersConfigDialog(BaseCheckersConfigDialog):
    """Configuration dialog for OpenSpiel Checkers (8x8).

    Standard American checkers rules:
    - 8x8 board with 32 playable (dark) squares
    - 12 pieces per side at start
    - Men cannot capture backward
    - Kings move one square diagonally
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
            variant_name="Checkers (OpenSpiel)"
        )

    def _get_presets(self) -> List[Tuple[str, str]]:
        return [
            ("Standard", STANDARD_8X8),
            ("Empty", EMPTY_8X8),
            ("Endgame 2v1", ENDGAME_2V1),
            ("Kings Battle", ENDGAME_KINGS),
        ]


__all__ = ["OpenSpielCheckersConfigDialog"]
