"""Qt controller classes for the Gym GUI."""

import sys

from gym_gui.controllers.chess_controller import ChessGameController, PlayerType
from gym_gui.controllers.connect_four_controller import ConnectFourGameController
from gym_gui.controllers.go_controller import GoGameController
from gym_gui.controllers.tictactoe_controller import TicTacToeGameController

__all__ = [
    "ChessGameController",
    "ConnectFourGameController",
    "GoGameController",
    "TicTacToeGameController",
    "PlayerType",
]

# Conditionally export evdev modules (Linux only)
if sys.platform.startswith('linux'):
    try:
        from gym_gui.controllers.evdev_keyboard_monitor import (
            EvdevKeyboardMonitor,
            KeyboardDevice,
        )
        from gym_gui.controllers.keycode_translation import (
            linux_keycode_to_qt_key,
            qt_key_to_linux_keycode,
            get_keycode_name,
        )
        __all__.extend([
            "EvdevKeyboardMonitor",
            "KeyboardDevice",
            "linux_keycode_to_qt_key",
            "qt_key_to_linux_keycode",
            "get_keycode_name",
        ])
    except ImportError:
        pass
