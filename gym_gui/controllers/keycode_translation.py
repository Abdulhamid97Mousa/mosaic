#!/usr/bin/env python3
"""
Linux keycode to Qt key translation table.

Maps Linux input event keycodes (from /usr/include/linux/input-event-codes.h)
to PyQt6 Qt.Key values for use with evdev keyboard monitoring.
"""
from PyQt6.QtCore import Qt
from typing import Dict

# Linux input keycodes to Qt Key mapping
# Source: /usr/include/linux/input-event-codes.h
LINUX_TO_QT_KEYCODE: Dict[int, Qt.Key] = {
    # Numbers
    2: Qt.Key.Key_1,
    3: Qt.Key.Key_2,
    4: Qt.Key.Key_3,
    5: Qt.Key.Key_4,
    6: Qt.Key.Key_5,
    7: Qt.Key.Key_6,
    8: Qt.Key.Key_7,
    9: Qt.Key.Key_8,
    10: Qt.Key.Key_9,
    11: Qt.Key.Key_0,

    # Letters (QWERTY layout)
    16: Qt.Key.Key_Q,
    17: Qt.Key.Key_W,
    18: Qt.Key.Key_E,
    19: Qt.Key.Key_R,
    20: Qt.Key.Key_T,
    21: Qt.Key.Key_Y,
    22: Qt.Key.Key_U,
    23: Qt.Key.Key_I,
    24: Qt.Key.Key_O,
    25: Qt.Key.Key_P,

    30: Qt.Key.Key_A,
    31: Qt.Key.Key_S,
    32: Qt.Key.Key_D,
    33: Qt.Key.Key_F,
    34: Qt.Key.Key_G,
    35: Qt.Key.Key_H,
    36: Qt.Key.Key_J,
    37: Qt.Key.Key_K,
    38: Qt.Key.Key_L,

    44: Qt.Key.Key_Z,
    45: Qt.Key.Key_X,
    46: Qt.Key.Key_C,
    47: Qt.Key.Key_V,
    48: Qt.Key.Key_B,
    49: Qt.Key.Key_N,
    50: Qt.Key.Key_M,

    # Function keys
    59: Qt.Key.Key_F1,
    60: Qt.Key.Key_F2,
    61: Qt.Key.Key_F3,
    62: Qt.Key.Key_F4,
    63: Qt.Key.Key_F5,
    64: Qt.Key.Key_F6,
    65: Qt.Key.Key_F7,
    66: Qt.Key.Key_F8,
    67: Qt.Key.Key_F9,
    68: Qt.Key.Key_F10,
    87: Qt.Key.Key_F11,
    88: Qt.Key.Key_F12,

    # Special keys
    1: Qt.Key.Key_Escape,
    14: Qt.Key.Key_Backspace,
    15: Qt.Key.Key_Tab,
    28: Qt.Key.Key_Return,  # Enter
    29: Qt.Key.Key_Control,
    42: Qt.Key.Key_Shift,
    54: Qt.Key.Key_Shift,  # Right shift
    56: Qt.Key.Key_Alt,
    57: Qt.Key.Key_Space,
    58: Qt.Key.Key_CapsLock,

    # Arrow keys
    103: Qt.Key.Key_Up,
    105: Qt.Key.Key_Left,
    106: Qt.Key.Key_Right,
    108: Qt.Key.Key_Down,

    # Numpad
    69: Qt.Key.Key_NumLock,
    71: Qt.Key.Key_7,  # Numpad 7
    72: Qt.Key.Key_8,  # Numpad 8
    73: Qt.Key.Key_9,  # Numpad 9
    74: Qt.Key.Key_Minus,  # Numpad minus
    75: Qt.Key.Key_4,  # Numpad 4
    76: Qt.Key.Key_5,  # Numpad 5
    77: Qt.Key.Key_6,  # Numpad 6
    78: Qt.Key.Key_Plus,  # Numpad plus
    79: Qt.Key.Key_1,  # Numpad 1
    80: Qt.Key.Key_2,  # Numpad 2
    81: Qt.Key.Key_3,  # Numpad 3
    82: Qt.Key.Key_0,  # Numpad 0
    83: Qt.Key.Key_Period,  # Numpad period
    96: Qt.Key.Key_Enter,  # Numpad enter
    98: Qt.Key.Key_Slash,  # Numpad slash
    55: Qt.Key.Key_Asterisk,  # Numpad asterisk

    # Navigation keys
    102: Qt.Key.Key_Home,
    104: Qt.Key.Key_PageUp,
    107: Qt.Key.Key_End,
    109: Qt.Key.Key_PageDown,
    110: Qt.Key.Key_Insert,
    111: Qt.Key.Key_Delete,

    # Punctuation
    12: Qt.Key.Key_Minus,
    13: Qt.Key.Key_Equal,
    26: Qt.Key.Key_BracketLeft,
    27: Qt.Key.Key_BracketRight,
    39: Qt.Key.Key_Semicolon,
    40: Qt.Key.Key_Apostrophe,
    41: Qt.Key.Key_QuoteLeft,  # Backtick
    43: Qt.Key.Key_Backslash,
    51: Qt.Key.Key_Comma,
    52: Qt.Key.Key_Period,
    53: Qt.Key.Key_Slash,

    # Additional modifier keys
    97: Qt.Key.Key_Control,  # Right Ctrl
    100: Qt.Key.Key_AltGr,  # Right Alt
    125: Qt.Key.Key_Meta,  # Left Windows/Super key
    126: Qt.Key.Key_Meta,  # Right Windows/Super key
    127: Qt.Key.Key_Menu,  # Menu key
}


def linux_keycode_to_qt_key(keycode: int) -> Qt.Key:
    """Convert Linux input keycode to Qt.Key.

    Args:
        keycode: Linux input event keycode

    Returns:
        Corresponding Qt.Key value, or Qt.Key.Key_unknown if not found
    """
    return LINUX_TO_QT_KEYCODE.get(keycode, Qt.Key.Key_unknown)


def qt_key_to_linux_keycode(qt_key: Qt.Key) -> int:
    """Convert Qt.Key to Linux input keycode.

    Args:
        qt_key: Qt.Key value

    Returns:
        Corresponding Linux keycode, or 0 if not found
    """
    for linux_code, key in LINUX_TO_QT_KEYCODE.items():
        if key == qt_key:
            return linux_code
    return 0


# Common key name mapping for display purposes
KEYCODE_NAMES: Dict[int, str] = {
    # WASD (common game controls)
    17: "W",
    30: "A",
    31: "S",
    32: "D",

    # Arrow keys
    103: "Up",
    105: "Left",
    106: "Right",
    108: "Down",

    # Common actions
    57: "Space",
    29: "Ctrl",
    42: "Shift",
    56: "Alt",
    28: "Enter",
    1: "Esc",

    # Numbers
    2: "1", 3: "2", 4: "3", 5: "4", 6: "5",
    7: "6", 8: "7", 9: "8", 10: "9", 11: "0",
}


def get_keycode_name(keycode: int) -> str:
    """Get human-readable name for a keycode.

    Args:
        keycode: Linux input event keycode

    Returns:
        Human-readable key name or hex representation
    """
    if keycode in KEYCODE_NAMES:
        return KEYCODE_NAMES[keycode]

    # Try to get Qt key name
    qt_key = linux_keycode_to_qt_key(keycode)
    if qt_key != Qt.Key.Key_unknown:
        # Extract key name from Qt.Key enum
        key_str = str(qt_key)
        if "Key_" in key_str:
            return key_str.split("Key_")[1]

    # Return hex representation as fallback
    return f"Key_{keycode:#x}"
