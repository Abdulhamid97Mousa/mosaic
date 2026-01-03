from __future__ import annotations

"""Keyboard shortcut management for human control within the Qt shell.

This module provides two input modes:
1. **Shortcut-based** (QShortcut): For turn-based games where single key presses trigger actions
2. **State-based** (key tracking): For real-time games requiring simultaneous key combinations

The state-based mode tracks all currently pressed keys and computes combined actions
(e.g., Up+Right → diagonal movement) on each game tick.
"""

from dataclasses import dataclass
import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import gymnasium.spaces as spaces
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent, QKeySequence, QShortcut

from gym_gui.core.enums import ControlMode, EnvironmentFamily, GameId, InputMode
from gym_gui.controllers.session import SessionController
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import LOG_INPUT_CONTROLLER_ERROR


_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Key Constants for State-Based Tracking
# =============================================================================
def _get_qt_key(name: str) -> int:
    """Get Qt key constant by name, handling Qt5/Qt6 differences."""
    key_enum = getattr(Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return int(getattr(key_enum, name))
    legacy = getattr(Qt, name, None)
    if legacy is not None:
        return int(legacy)
    raise AttributeError(f"Qt key '{name}' not available")


# Direction keys (both arrow keys and WASD)
_KEY_UP = _get_qt_key("Key_Up")
_KEY_DOWN = _get_qt_key("Key_Down")
_KEY_LEFT = _get_qt_key("Key_Left")
_KEY_RIGHT = _get_qt_key("Key_Right")
_KEY_W = _get_qt_key("Key_W")
_KEY_A = _get_qt_key("Key_A")
_KEY_S = _get_qt_key("Key_S")
_KEY_D = _get_qt_key("Key_D")
_KEY_SPACE = _get_qt_key("Key_Space")
_KEY_Q = _get_qt_key("Key_Q")
_KEY_E = _get_qt_key("Key_E")
_KEY_Z = _get_qt_key("Key_Z")
_KEY_C = _get_qt_key("Key_C")
_KEY_1 = _get_qt_key("Key_1")
_KEY_2 = _get_qt_key("Key_2")

# Sets for direction detection
_KEYS_UP = {_KEY_UP, _KEY_W}
_KEYS_DOWN = {_KEY_DOWN, _KEY_S}
_KEYS_LEFT = {_KEY_LEFT, _KEY_A}
_KEYS_RIGHT = {_KEY_RIGHT, _KEY_D}


# =============================================================================
# Key Combination Resolvers for Different Game Families
# =============================================================================
class KeyCombinationResolver:
    """Base class for resolving key combinations to game actions."""

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        """Resolve currently pressed keys to a single game action.

        Args:
            pressed_keys: Set of currently pressed Qt key codes.

        Returns:
            Action index, or None if no recognized action.
        """
        raise NotImplementedError


class ProcgenKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Procgen environments.

    Procgen action space (15 actions):
    0: down_left, 1: left, 2: up_left, 3: down, 4: noop, 5: up,
    6: down_right, 7: right, 8: up_right,
    9: action_d (fire), 10: action_a, 11: action_w, 12: action_s,
    13: action_q, 14: action_e
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        # Check directions
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # Cancel out opposing directions
        if up and down:
            up = down = False
        if left and right:
            left = right = False

        # Diagonal combinations (check first - more specific)
        if up and right:
            return 8  # up_right
        if up and left:
            return 2  # up_left
        if down and right:
            return 6  # down_right
        if down and left:
            return 0  # down_left

        # Cardinal directions
        if up:
            return 5  # up
        if down:
            return 3  # down
        if left:
            return 1  # left
        if right:
            return 7  # right

        # Action buttons (D is primary fire/interact)
        if _KEY_SPACE in pressed_keys or _KEY_D in pressed_keys:
            return 9  # action_d (fire)

        # Secondary action buttons
        if _KEY_1 in pressed_keys:
            return 13  # action_q
        if _KEY_2 in pressed_keys:
            return 14  # action_e

        return None  # No action - will use NOOP (4) from idle tick


class AleKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for ALE/Atari environments.

    Standard ALE action space (18 actions):
    0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN,
    6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
    10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE,
    14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        fire = _KEY_SPACE in pressed_keys

        # Cancel opposing directions
        if up and down:
            up = down = False
        if left and right:
            left = right = False

        # Diagonal + fire combinations
        if fire:
            if up and right:
                return 14  # UPRIGHTFIRE
            if up and left:
                return 15  # UPLEFTFIRE
            if down and right:
                return 16  # DOWNRIGHTFIRE
            if down and left:
                return 17  # DOWNLEFTFIRE
            if up:
                return 10  # UPFIRE
            if right:
                return 11  # RIGHTFIRE
            if left:
                return 12  # LEFTFIRE
            if down:
                return 13  # DOWNFIRE
            return 1  # FIRE only

        # Diagonal combinations (no fire)
        if up and right:
            return 6  # UPRIGHT
        if up and left:
            return 7  # UPLEFT
        if down and right:
            return 8  # DOWNRIGHT
        if down and left:
            return 9  # DOWNLEFT

        # Cardinal directions
        if up:
            return 2  # UP
        if right:
            return 3  # RIGHT
        if left:
            return 4  # LEFT
        if down:
            return 5  # DOWN

        return None  # NOOP


class LunarLanderKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for LunarLander environment."""

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        # LunarLander: 0=idle, 1=left engine, 2=main engine, 3=right engine
        up = bool(pressed_keys & _KEYS_UP)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # Priority: main engine > side engines
        if up:
            return 2  # Fire main engine
        if left:
            return 1  # Fire left engine
        if right:
            return 3  # Fire right engine

        return None  # Idle


# Backwards compatibility alias
Box2DKeyCombinationResolver = LunarLanderKeyCombinationResolver


class CarRacingKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for CarRacing environment.

    CarRacing action mapping (discrete indices to continuous presets):
    0: idle/coast (Space)
    1: steer right (Right/D)
    2: steer left (Left/A)
    3: accelerate (Up/W)
    4: brake (Down/S)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        space = _KEY_SPACE in pressed_keys

        # Priority: acceleration/brake > steering > idle
        if up:
            return 3  # Accelerate
        if down:
            return 4  # Brake
        if right:
            return 1  # Steer right
        if left:
            return 2  # Steer left
        if space:
            return 0  # Idle/coast

        return None  # No action


class BipedalWalkerKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for BipedalWalker environment.

    BipedalWalker action mapping:
    0: neutral stance (Space)
    1: lean forward/step (Right/D)
    2: lean backward/step back (Left/A)
    3: crouch/prepare jump (Up/W)
    4: extend legs/hop (Down/S)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        space = _KEY_SPACE in pressed_keys

        if right:
            return 1  # Lean forward
        if left:
            return 2  # Lean backward
        if up:
            return 3  # Crouch
        if down:
            return 4  # Extend legs
        if space:
            return 0  # Neutral

        return None


class ViZDoomKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for ViZDoom environments.

    Note: ViZDoom action spaces vary by scenario. This resolver handles common patterns.
    """

    def __init__(self, game_id: GameId):
        self._game_id = game_id

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        # Basic actions for most scenarios
        fire = _KEY_SPACE in pressed_keys
        up = bool(pressed_keys & _KEYS_UP)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # For now, return single actions (ViZDoom handles combos differently)
        if fire:
            return 0  # ATTACK (common index)
        if up:
            return 3 if self._game_id == GameId.VIZDOOM_DEADLY_CORRIDOR else 0  # MOVE_FORWARD
        if left:
            return 1  # TURN_LEFT / MOVE_LEFT
        if right:
            return 2  # TURN_RIGHT / MOVE_RIGHT

        return None


# Map environment families to their resolvers
def get_key_combination_resolver(game_id: GameId) -> Optional[KeyCombinationResolver]:
    """Get the appropriate key combination resolver for a game.

    Returns a resolver that maps key combinations to game actions, or None
    if no resolver is available for this game type.

    The resolver is determined by:
    1. Checking for game-specific resolvers (e.g., CarRacing, LunarLander)
    2. Looking up the game's family in ENVIRONMENT_FAMILY_BY_GAME
    3. Falling back to checking game ID prefixes/patterns
    """
    from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME

    # Game-specific resolvers for Box2D (each has unique action space)
    if game_id == GameId.CAR_RACING:
        return CarRacingKeyCombinationResolver()
    if game_id == GameId.LUNAR_LANDER:
        return LunarLanderKeyCombinationResolver()
    if game_id == GameId.BIPEDAL_WALKER:
        return BipedalWalkerKeyCombinationResolver()

    # First, try to get the family from the mapping
    family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)

    if family == EnvironmentFamily.PROCGEN:
        return ProcgenKeyCombinationResolver()
    if family in (EnvironmentFamily.ALE, EnvironmentFamily.ATARI):
        return AleKeyCombinationResolver()
    if family == EnvironmentFamily.VIZDOOM:
        return ViZDoomKeyCombinationResolver(game_id)

    # Fallback: check by game ID prefix/name for games not in the mapping
    game_name = game_id.value if hasattr(game_id, 'value') else str(game_id)

    if game_name.startswith("procgen:") or game_name.startswith("procgen/"):
        return ProcgenKeyCombinationResolver()
    if game_name.startswith("ALE/") or game_name.endswith("-v4") or game_name.endswith("-v5"):
        return AleKeyCombinationResolver()
    if game_name.startswith("ViZDoom"):
        return ViZDoomKeyCombinationResolver(game_id)

    return None


@dataclass(frozen=True)
class ShortcutMapping:
    key_sequences: Tuple[QKeySequence, ...]
    action: int


def _qt_key(name: str) -> int:
    key_enum = getattr(QtCore.Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return getattr(key_enum, name)
    legacy = getattr(QtCore.Qt, name, None)
    if legacy is None:  # pragma: no cover - defensive
        raise AttributeError(f"Qt key '{name}' not available")
    return legacy


def _mapping(names: Iterable[str], action: int) -> ShortcutMapping:
    sequences = tuple(QKeySequence(_qt_key(name)) for name in names)
    return ShortcutMapping(sequences, action)


_TOY_TEXT_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.FROZEN_LAKE: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Down", "Key_S"), 1),
        _mapping(("Key_Right", "Key_D"), 2),
        _mapping(("Key_Up", "Key_W"), 3),
    ),
    GameId.CLIFF_WALKING: (
        _mapping(("Key_Up", "Key_W"), 0),     # UP
        _mapping(("Key_Right", "Key_D"), 1),  # RIGHT
        _mapping(("Key_Down", "Key_S"), 2),   # DOWN
        _mapping(("Key_Left", "Key_A"), 3),   # LEFT
    ),
    GameId.TAXI: (
        _mapping(("Key_Down", "Key_S"), 0),   # SOUTH
        _mapping(("Key_Up", "Key_W"), 1),     # NORTH
        _mapping(("Key_Right", "Key_D"), 2),  # EAST
        _mapping(("Key_Left", "Key_A"), 3),   # WEST
        _mapping(("Key_Space",), 4),            # PICKUP
        _mapping(("Key_E",), 5),                # DROPOFF
    ),
    GameId.BLACKJACK: (
        _mapping(("Key_1", "Key_Q"), 0),      # STICK (stop taking cards) - 1 or Q
        _mapping(("Key_2", "Key_E"), 1),      # HIT (take another card) - 2 or E
    ),
}

_MINIG_GRID_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.MINIGRID_EMPTY_5x5: (
        _mapping(("Key_Left", "Key_A"), 0),    # turn left
        _mapping(("Key_Right", "Key_D"), 1),   # turn right
        _mapping(("Key_Up", "Key_W"), 2),      # move forward
        _mapping(("Key_G", "Key_Space"), 3),   # pick up
        _mapping(("Key_H",), 4),                # drop (rarely used)
        _mapping(("Key_E", "Key_Return"), 5),  # toggle / use
        _mapping(("Key_Q",), 6),                # done / no-op
    ),
    GameId.MINIGRID_EMPTY_RANDOM_5x5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_EMPTY_6x6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_EMPTY_RANDOM_6x6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_EMPTY_8x8: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_EMPTY_16x16: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_DOORKEY_5x5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_DOORKEY_6x6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_DOORKEY_8x8: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_DOORKEY_16x16: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_LAVAGAP_S5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_LAVAGAP_S6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_LAVAGAP_S7: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    # LavaCrossing environments - 3 actions only (turn left, turn right, move forward)
    GameId.MINIGRID_LAVA_CROSSING_S9N1: (
        _mapping(("Key_Left", "Key_A"), 0),    # turn left
        _mapping(("Key_Right", "Key_D"), 1),   # turn right
        _mapping(("Key_Up", "Key_W"), 2),      # move forward
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N2: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N3: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_LAVA_CROSSING_S11N5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    # SimpleCrossing environments - 3 actions only (turn left, turn right, move forward)
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: (
        _mapping(("Key_Left", "Key_A"), 0),    # turn left
        _mapping(("Key_Right", "Key_D"), 1),   # turn right
        _mapping(("Key_Up", "Key_W"), 2),      # move forward
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
    ),
    GameId.MINIGRID_BLOCKED_UNLOCK_PICKUP: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_MULTIROOM_N2_S4: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_MULTIROOM_N4_S5: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_MULTIROOM_N6: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
    GameId.MINIGRID_OBSTRUCTED_MAZE_FULL: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Right", "Key_D"), 1),
        _mapping(("Key_Up", "Key_W"), 2),
        _mapping(("Key_G", "Key_Space"), 3),
        _mapping(("Key_H",), 4),
        _mapping(("Key_E", "Key_Return"), 5),
        _mapping(("Key_Q",), 6),
    ),
}

_BOX_2D_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.LUNAR_LANDER: (
        _mapping(("Key_Space", "Key_S"), 0),  # Idle / cut thrust
        _mapping(("Key_Left", "Key_A"), 1),   # Fire left engine
        _mapping(("Key_Up", "Key_W"), 2),     # Fire main engine
        _mapping(("Key_Right", "Key_D"), 3),  # Fire right engine
    ),
    GameId.CAR_RACING: (
        _mapping(("Key_Space",), 0),           # Neutral / coast
        _mapping(("Key_Right", "Key_D"), 1), # Steer right
        _mapping(("Key_Left", "Key_A"), 2),  # Steer left
        _mapping(("Key_Up", "Key_W"), 3),    # Accelerate
        _mapping(("Key_Down", "Key_S"), 4),  # Brake
    ),
    GameId.BIPEDAL_WALKER: (
        _mapping(("Key_Space",), 0),            # Neutral stance
        _mapping(("Key_Right", "Key_D"), 1),  # Lean forward / step
        _mapping(("Key_Left", "Key_A"), 2),   # Lean backward / step back
        _mapping(("Key_Up", "Key_W"), 3),     # Crouch / prepare jump
        _mapping(("Key_Down", "Key_S"), 4),   # Extend legs / hop
    ),
}

# ALE (Atari) Adventure mappings (Discrete(18))
# ViZDoom mappings - button indices match _available_buttons order in adapter
# Each scenario has different available buttons, so mappings match scenario-specific order
# ViZDoom mouse turn action indices: (turn_left_action, turn_right_action)
# Maps each scenario to the button indices used for turning left/right
# Used for FPS-style mouse capture control
_VIZDOOM_MOUSE_TURN_ACTIONS: Dict[GameId, Tuple[int, int]] = {
    # Basic has no turn - uses MOVE_LEFT(1), MOVE_RIGHT(2) for lateral movement
    GameId.VIZDOOM_BASIC: (1, 2),  # MOVE_LEFT, MOVE_RIGHT (no true turn)
    # DeadlyCorridor: TURN_LEFT(4), TURN_RIGHT(5)
    GameId.VIZDOOM_DEADLY_CORRIDOR: (4, 5),
    # DefendTheCenter: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_CENTER: (1, 2),
    # DefendTheLine: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_LINE: (1, 2),
    # HealthGathering: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING: (1, 2),
    # HealthGatheringSupreme: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: (1, 2),
    # MyWayHome: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_MY_WAY_HOME: (1, 2),
    # PredictPosition: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_PREDICT_POSITION: (1, 2),
    # TakeCover has no turn - uses MOVE_LEFT(0), MOVE_RIGHT(1) for lateral movement
    GameId.VIZDOOM_TAKE_COVER: (0, 1),  # MOVE_LEFT, MOVE_RIGHT (no true turn)
    # Deathmatch: TURN_LEFT(6), TURN_RIGHT(7)
    GameId.VIZDOOM_DEATHMATCH: (6, 7),
}


def get_vizdoom_mouse_turn_actions(game_id: GameId) -> Tuple[int, int] | None:
    """Return (turn_left_action, turn_right_action) for a ViZDoom game, or None if not ViZDoom."""
    return _VIZDOOM_MOUSE_TURN_ACTIONS.get(game_id)


_VIZDOOM_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Basic: ATTACK(0), MOVE_LEFT(1), MOVE_RIGHT(2)
    GameId.VIZDOOM_BASIC: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # MOVE_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # MOVE_RIGHT
    ),
    # DeadlyCorridor: ATTACK(0), MOVE_LEFT(1), MOVE_RIGHT(2), MOVE_FORWARD(3), TURN_LEFT(4), TURN_RIGHT(5)
    GameId.VIZDOOM_DEADLY_CORRIDOR: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_A",), 1),                     # MOVE_LEFT (strafe)
        _mapping(("Key_D",), 2),                     # MOVE_RIGHT (strafe)
        _mapping(("Key_Up", "Key_W"), 3),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_Q"), 4),          # TURN_LEFT
        _mapping(("Key_Right", "Key_E"), 5),         # TURN_RIGHT
    ),
    # DefendTheCenter: ATTACK(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_CENTER: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # DefendTheLine: ATTACK(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_LINE: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # HealthGathering: MOVE_FORWARD(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING: (
        _mapping(("Key_Up", "Key_W"), 0),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # HealthGatheringSupreme: MOVE_FORWARD(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: (
        _mapping(("Key_Up", "Key_W"), 0),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # MyWayHome: MOVE_FORWARD(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_MY_WAY_HOME: (
        _mapping(("Key_Up", "Key_W"), 0),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # PredictPosition: ATTACK(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_PREDICT_POSITION: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # TakeCover: MOVE_LEFT(0), MOVE_RIGHT(1)
    GameId.VIZDOOM_TAKE_COVER: (
        _mapping(("Key_Left", "Key_A"), 0),          # MOVE_LEFT
        _mapping(("Key_Right", "Key_D"), 1),         # MOVE_RIGHT
    ),
    # Deathmatch: ATTACK(0), USE(1), MOVE_FORWARD(2), MOVE_BACKWARD(3), MOVE_LEFT(4), MOVE_RIGHT(5), TURN_LEFT(6), TURN_RIGHT(7)
    GameId.VIZDOOM_DEATHMATCH: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_E", "Key_Return"), 1),        # USE
        _mapping(("Key_Up", "Key_W"), 2),            # MOVE_FORWARD
        _mapping(("Key_Down", "Key_S"), 3),          # MOVE_BACKWARD
        _mapping(("Key_A",), 4),                     # MOVE_LEFT (strafe)
        _mapping(("Key_D",), 5),                     # MOVE_RIGHT (strafe)
        _mapping(("Key_Left", "Key_Q"), 6),          # TURN_LEFT
        _mapping(("Key_Right",), 7),                 # TURN_RIGHT
    ),
}

# ===========================================================================
# MiniHack Mappings (roguelike vi-keys + WASD alternatives)
# NLE action indices: 0-7 = 8 compass directions (N, E, S, W, NE, SE, SW, NW)
# Many MiniHack envs use Discrete(8) for basic navigation
# ===========================================================================
def _minihack_nav_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 8-direction navigation for MiniHack environments."""
    return (
        _mapping(("Key_K", "Key_Up", "Key_W"), 0),     # North (up)
        _mapping(("Key_L", "Key_Right", "Key_D"), 1),  # East (right)
        _mapping(("Key_J", "Key_Down", "Key_S"), 2),   # South (down)
        _mapping(("Key_H", "Key_Left", "Key_A"), 3),   # West (left)
        _mapping(("Key_U",), 4),                        # Northeast (diag)
        _mapping(("Key_N",), 5),                        # Southeast (diag)
        _mapping(("Key_B",), 6),                        # Southwest (diag)
        _mapping(("Key_Y",), 7),                        # Northwest (diag)
    )


_MINIHACK_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Navigation environments (8 directions)
    GameId.MINIHACK_ROOM_5X5: _minihack_nav_mappings(),
    GameId.MINIHACK_ROOM_15X15: _minihack_nav_mappings(),
    GameId.MINIHACK_CORRIDOR_R2: _minihack_nav_mappings(),
    GameId.MINIHACK_CORRIDOR_R3: _minihack_nav_mappings(),
    GameId.MINIHACK_CORRIDOR_R5: _minihack_nav_mappings(),
    GameId.MINIHACK_MAZEWALK_9X9: _minihack_nav_mappings(),
    GameId.MINIHACK_MAZEWALK_15X15: _minihack_nav_mappings(),
    GameId.MINIHACK_MAZEWALK_45X19: _minihack_nav_mappings(),
    GameId.MINIHACK_RIVER: _minihack_nav_mappings(),
    GameId.MINIHACK_RIVER_NARROW: _minihack_nav_mappings(),
    # Exploration environments
    GameId.MINIHACK_EXPLOREMAZE_EASY: _minihack_nav_mappings(),
    GameId.MINIHACK_EXPLOREMAZE_HARD: _minihack_nav_mappings(),
    GameId.MINIHACK_HIDENSEEK: _minihack_nav_mappings(),
    GameId.MINIHACK_MEMENTO_F2: _minihack_nav_mappings(),
    GameId.MINIHACK_MEMENTO_F4: _minihack_nav_mappings(),
    # Skill environments (use fallback for extended actions)
    GameId.MINIHACK_EAT: _minihack_nav_mappings(),
    GameId.MINIHACK_WEAR: _minihack_nav_mappings(),
    GameId.MINIHACK_WIELD: _minihack_nav_mappings(),
    GameId.MINIHACK_ZAP: _minihack_nav_mappings(),
    GameId.MINIHACK_READ: _minihack_nav_mappings(),
    GameId.MINIHACK_QUAFF: _minihack_nav_mappings(),
    GameId.MINIHACK_PUTON: _minihack_nav_mappings(),
    GameId.MINIHACK_LAVACROSS: _minihack_nav_mappings(),
    GameId.MINIHACK_WOD_EASY: _minihack_nav_mappings(),
    GameId.MINIHACK_WOD_MEDIUM: _minihack_nav_mappings(),
    GameId.MINIHACK_WOD_HARD: _minihack_nav_mappings(),
}

# ===========================================================================
# NetHack Mappings (full game via NLE)
# NLE has ~113 actions; mapping core navigation here
# ===========================================================================
_NETHACK_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # NetHack full game uses same 8-direction navigation as base
    GameId.NETHACK_FULL: _minihack_nav_mappings(),
    GameId.NETHACK_SCORE: _minihack_nav_mappings(),
    GameId.NETHACK_STAIRCASE: _minihack_nav_mappings(),
    GameId.NETHACK_STAIRCASE_PET: _minihack_nav_mappings(),
    GameId.NETHACK_ORACLE: _minihack_nav_mappings(),
    GameId.NETHACK_GOLD: _minihack_nav_mappings(),
    GameId.NETHACK_EAT: _minihack_nav_mappings(),
    GameId.NETHACK_SCOUT: _minihack_nav_mappings(),
}

# ===========================================================================
# Crafter Mappings (open-world survival benchmark)
# 17 discrete actions from crafter/data.yaml:
# 0:noop, 1:move_left, 2:move_right, 3:move_up, 4:move_down, 5:do, 6:sleep,
# 7:place_stone, 8:place_table, 9:place_furnace, 10:place_plant,
# 11:make_wood_pickaxe, 12:make_stone_pickaxe, 13:make_iron_pickaxe,
# 14:make_wood_sword, 15:make_stone_sword, 16:make_iron_sword
# ===========================================================================
def _crafter_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 17-action mapping for Crafter environments."""
    return (
        # Note: action 0 (noop) has no key - happens on timeout or idle
        _mapping(("Key_Left", "Key_A"), 1),       # move_left
        _mapping(("Key_Right", "Key_D"), 2),      # move_right
        _mapping(("Key_Up", "Key_W"), 3),         # move_up
        _mapping(("Key_Down", "Key_S"), 4),       # move_down
        _mapping(("Key_Space",), 5),               # do (interact)
        _mapping(("Key_R",), 6),                   # sleep
        _mapping(("Key_1",), 7),                   # place_stone
        _mapping(("Key_2",), 8),                   # place_table
        _mapping(("Key_3",), 9),                   # place_furnace
        _mapping(("Key_4",), 10),                  # place_plant
        _mapping(("Key_Q",), 11),                  # make_wood_pickaxe
        _mapping(("Key_E",), 12),                  # make_stone_pickaxe
        _mapping(("Key_F",), 13),                  # make_iron_pickaxe
        _mapping(("Key_Z",), 14),                  # make_wood_sword
        _mapping(("Key_X",), 15),                  # make_stone_sword
        _mapping(("Key_C",), 16),                  # make_iron_sword
    )


_CRAFTER_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.CRAFTER_REWARD: _crafter_mappings(),
    GameId.CRAFTER_NO_REWARD: _crafter_mappings(),
}

# ===========================================================================
# Procgen Mappings (procedurally generated benchmark - 16 environments)
# 15 discrete actions (button combinations):
# 0:down_left, 1:left, 2:up_left, 3:down, 4:noop, 5:up,
# 6:down_right, 7:right, 8:up_right,
# 9:action_d (fire), 10:action_a, 11:action_w, 12:action_s, 13:action_q, 14:action_e
# ===========================================================================
def _procgen_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 15-action mapping for Procgen environments."""
    return (
        # Diagonal: down-left (action 0) - no common key, use numpad 1 or Z
        _mapping(("Key_Z",), 0),                    # down_left
        # Cardinal directions using arrow keys
        _mapping(("Key_Left",), 1),                 # left
        # Diagonal: up-left (action 2) - use Q
        _mapping(("Key_Q",), 2),                    # up_left
        _mapping(("Key_Down",), 3),                 # down
        # Noop (action 4) - space or 0
        _mapping(("Key_0",), 4),                    # noop
        _mapping(("Key_Up",), 5),                   # up
        # Diagonal: down-right (action 6) - use C
        _mapping(("Key_C",), 6),                    # down_right
        _mapping(("Key_Right",), 7),                # right
        # Diagonal: up-right (action 8) - use E
        _mapping(("Key_E",), 8),                    # up_right
        # Game-specific actions (fire/interact buttons)
        _mapping(("Key_Space", "Key_D"), 9),        # action_d (primary fire/interact)
        _mapping(("Key_A",), 10),                   # action_a (secondary)
        _mapping(("Key_W",), 11),                   # action_w (tertiary)
        _mapping(("Key_S",), 12),                   # action_s (quaternary)
        _mapping(("Key_1",), 13),                   # action_q (special 1)
        _mapping(("Key_2",), 14),                   # action_e (special 2)
    )


_PROCGEN_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.PROCGEN_BIGFISH: _procgen_mappings(),
    GameId.PROCGEN_BOSSFIGHT: _procgen_mappings(),
    GameId.PROCGEN_CAVEFLYER: _procgen_mappings(),
    GameId.PROCGEN_CHASER: _procgen_mappings(),
    GameId.PROCGEN_CLIMBER: _procgen_mappings(),
    GameId.PROCGEN_COINRUN: _procgen_mappings(),
    GameId.PROCGEN_DODGEBALL: _procgen_mappings(),
    GameId.PROCGEN_FRUITBOT: _procgen_mappings(),
    GameId.PROCGEN_HEIST: _procgen_mappings(),
    GameId.PROCGEN_JUMPER: _procgen_mappings(),
    GameId.PROCGEN_LEAPER: _procgen_mappings(),
    GameId.PROCGEN_MAZE: _procgen_mappings(),
    GameId.PROCGEN_MINER: _procgen_mappings(),
    GameId.PROCGEN_NINJA: _procgen_mappings(),
    GameId.PROCGEN_PLUNDER: _procgen_mappings(),
    GameId.PROCGEN_STARPILOT: _procgen_mappings(),
}

_ALE_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Minimal but explicit mapping for core and diagonal moves, plus fire variants via letter keys.
    GameId.ADVENTURE_V4: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_Down", "Key_S"), 5),  # DOWN
        _mapping(("Key_E",), 6),               # UPRIGHT
        _mapping(("Key_Q",), 7),               # UPLEFT
        _mapping(("Key_C",), 8),               # DOWNRIGHT
        _mapping(("Key_Z",), 9),               # DOWNLEFT
        _mapping(("Key_I",), 10),              # UPFIRE
        _mapping(("Key_L",), 11),              # RIGHTFIRE
        _mapping(("Key_J",), 12),              # LEFTFIRE
        _mapping(("Key_K",), 13),              # DOWNFIRE
        _mapping(("Key_O",), 14),              # UPRIGHTFIRE
        _mapping(("Key_U",), 15),              # UPLEFTFIRE
        _mapping(("Key_M",), 16),              # DOWNRIGHTFIRE
        _mapping(("Key_N",), 17),              # DOWNLEFTFIRE
    ),
    GameId.ALE_ADVENTURE_V5: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_Down", "Key_S"), 5),  # DOWN
        _mapping(("Key_E",), 6),               # UPRIGHT
        _mapping(("Key_Q",), 7),               # UPLEFT
        _mapping(("Key_C",), 8),               # DOWNRIGHT
        _mapping(("Key_Z",), 9),               # DOWNLEFT
        _mapping(("Key_I",), 10),              # UPFIRE
        _mapping(("Key_L",), 11),              # RIGHTFIRE
        _mapping(("Key_J",), 12),              # LEFTFIRE
        _mapping(("Key_K",), 13),              # DOWNFIRE
        _mapping(("Key_O",), 14),              # UPRIGHTFIRE
        _mapping(("Key_U",), 15),              # UPLEFTFIRE
        _mapping(("Key_M",), 16),              # DOWNRIGHTFIRE
        _mapping(("Key_N",), 17),              # DOWNLEFTFIRE
    ),
    # AirRaid uses Discrete(6) by default; map core actions and fire variants
    GameId.AIR_RAID_V4: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Right", "Key_D"), 2), # RIGHT
        _mapping(("Key_Left", "Key_A"), 3),  # LEFT
        _mapping(("Key_L",), 4),              # RIGHTFIRE
        _mapping(("Key_J",), 5),              # LEFTFIRE
    ),
    GameId.ALE_AIR_RAID_V5: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Right", "Key_D"), 2), # RIGHT
        _mapping(("Key_Left", "Key_A"), 3),  # LEFT
        _mapping(("Key_L",), 4),              # RIGHTFIRE
        _mapping(("Key_J",), 5),              # LEFTFIRE
    ),
    # Assault uses Discrete(7): NOOP, FIRE, UP, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    GameId.ASSAULT_V4: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_L",), 5),              # RIGHTFIRE
        _mapping(("Key_J",), 6),              # LEFTFIRE
    ),
    GameId.ALE_ASSAULT_V5: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_L",), 5),              # RIGHTFIRE
        _mapping(("Key_J",), 6),              # LEFTFIRE
    ),
}


# Jumanji Logic Puzzle Environments
# Game2048: 4 actions (up=0, down=1, left=2, right=3)
# Minesweeper: Cell indices (use mouse click, not keyboard for this one)
# RubiksCube: 12 actions (6 faces x 2 directions)
# SlidingPuzzle: 4 actions (move blank up/down/left/right)
# Sudoku: 81 cells x 9 digits (complex, may need mouse)
# GraphColoring: node x color combinations (complex, may need mouse)

def _game2048_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for 2048 tile sliding."""
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # up
        _mapping(("Key_Down", "Key_S"), 1),    # down
        _mapping(("Key_Left", "Key_A"), 2),    # left
        _mapping(("Key_Right", "Key_D"), 3),   # right
    )


def _sliding_puzzle_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for sliding tile puzzle."""
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # move blank up
        _mapping(("Key_Down", "Key_S"), 1),    # move blank down
        _mapping(("Key_Left", "Key_A"), 2),    # move blank left
        _mapping(("Key_Right", "Key_D"), 3),   # move blank right
    )


def _rubiks_cube_mappings() -> Tuple[ShortcutMapping, ...]:
    """Letter keys for Rubik's Cube face rotations.

    Standard cube notation:
    R=right, L=left, U=up, D=down, F=front, B=back
    Shift+key for counter-clockwise (prime)
    """
    return (
        _mapping(("Key_R",), 0),   # R (right clockwise)
        _mapping(("Key_T",), 1),   # R' (right counter-clockwise, use T for shift-free)
        _mapping(("Key_L",), 2),   # L (left clockwise)
        _mapping(("Key_K",), 3),   # L' (left counter-clockwise)
        _mapping(("Key_U",), 4),   # U (up clockwise)
        _mapping(("Key_Y",), 5),   # U' (up counter-clockwise)
        _mapping(("Key_D",), 6),   # D (down clockwise)
        _mapping(("Key_E",), 7),   # D' (down counter-clockwise)
        _mapping(("Key_F",), 8),   # F (front clockwise)
        _mapping(("Key_G",), 9),   # F' (front counter-clockwise)
        _mapping(("Key_B",), 10),  # B (back clockwise)
        _mapping(("Key_N",), 11),  # B' (back counter-clockwise)
    )


def _pacman_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for PacMan movement.

    Jumanji PacMan action space: Discrete(5)
    - 0: no-op (stay)
    - 1: up
    - 2: right
    - 3: down
    - 4: left
    """
    return (
        _mapping(("Key_Up", "Key_W"), 1),      # up
        _mapping(("Key_Right", "Key_D"), 2),   # right
        _mapping(("Key_Down", "Key_S"), 3),    # down
        _mapping(("Key_Left", "Key_A"), 4),    # left
    )


def _snake_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for Snake movement.

    Jumanji Snake action space: Discrete(4)
    - 0: up
    - 1: right
    - 2: down
    - 3: left
    """
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # up
        _mapping(("Key_Right", "Key_D"), 1),   # right
        _mapping(("Key_Down", "Key_S"), 2),    # down
        _mapping(("Key_Left", "Key_A"), 3),    # left
    )


_JUMANJI_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.JUMANJI_GAME2048: _game2048_mappings(),
    GameId.JUMANJI_SLIDING_PUZZLE: _sliding_puzzle_mappings(),
    GameId.JUMANJI_RUBIKS_CUBE: _rubiks_cube_mappings(),
    GameId.JUMANJI_PACMAN: _pacman_mappings(),
    GameId.JUMANJI_SNAKE: _snake_mappings(),
    # Minesweeper, Sudoku, and GraphColoring use complex action spaces
    # that are better suited for mouse-based interaction
    # Tetris uses MultiDiscrete which needs special handling
}


class HumanInputController(QtCore.QObject, LogConstantMixin):
    """Registers keyboard shortcuts and forwards them to the session controller.

    This controller supports two input modes:

    1. **Shortcut-based** (turn-based games): Uses Qt's QShortcut mechanism for
       single-key actions. Each key press immediately triggers an action.

    2. **State-based** (real-time games): Tracks all currently pressed keys and
       computes combined actions (e.g., Up+Right → diagonal movement) on each
       game tick. This is essential for games requiring simultaneous key presses.

    The mode is automatically selected based on the environment family.
    """

    # Signal emitted when a key combination is detected in state-based mode
    # The session controller listens to this to apply actions during idle ticks
    key_action_available = QtCore.Signal(int)  # action index

    def __init__(self, widget: QtWidgets.QWidget, session: SessionController) -> None:
        super().__init__(widget)
        self._logger = _LOGGER
        self._widget = widget
        self._session = session
        self._shortcuts: List[QShortcut] = []
        self._mode_allows_input = True
        self._requested_enabled = True

        # State-based key tracking
        self._pressed_keys: Set[int] = set()
        self._use_state_based_input = False
        self._key_resolver: Optional[KeyCombinationResolver] = None
        self._current_game_id: Optional[GameId] = None

        # Install event filter for state-based input
        self._widget.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filter key events for state-based input tracking.

        This captures keyPressEvent and keyReleaseEvent to maintain a set of
        currently pressed keys, enabling simultaneous key combination detection.
        """
        if not self._use_state_based_input:
            return False  # Let shortcuts handle it

        if not self._mode_allows_input or not self._requested_enabled:
            return False

        event_type = event.type()

        if event_type == QtCore.QEvent.Type.KeyPress:
            key_event = event  # type: QKeyEvent
            if hasattr(key_event, 'isAutoRepeat') and key_event.isAutoRepeat():
                return False  # Ignore OS auto-repeat
            key = key_event.key()
            if key not in self._pressed_keys:
                self._pressed_keys.add(key)
                _LOGGER.debug("Key pressed: %s, pressed_keys=%s", key, self._pressed_keys)
                # Emit immediate action for responsive feel
                self._emit_current_action()
            return True  # Consume the event

        if event_type == QtCore.QEvent.Type.KeyRelease:
            key_event = event  # type: QKeyEvent
            if hasattr(key_event, 'isAutoRepeat') and key_event.isAutoRepeat():
                return False  # Ignore OS auto-repeat
            key = key_event.key()
            self._pressed_keys.discard(key)
            _LOGGER.debug("Key released: %s, pressed_keys=%s", key, self._pressed_keys)
            return True  # Consume the event

        # Handle focus loss - clear all keys
        if event_type == QtCore.QEvent.Type.FocusOut:
            if self._pressed_keys:
                _LOGGER.debug("Focus lost, clearing pressed keys")
                self._pressed_keys.clear()

        return False

    def _emit_current_action(self) -> None:
        """Compute and emit action from currently pressed keys."""
        action = self.get_current_action()
        if action is not None:
            self._session.perform_human_action(action, key_label="combo")

    def get_current_action(self) -> Optional[int]:
        """Get the action corresponding to currently pressed keys.

        Returns:
            Action index if keys are pressed and recognized, None otherwise.
        """
        if not self._pressed_keys or self._key_resolver is None:
            return None
        return self._key_resolver.resolve(self._pressed_keys)

    def has_keys_pressed(self) -> bool:
        """Return True if any tracked keys are currently pressed."""
        return bool(self._pressed_keys)

    def clear_pressed_keys(self) -> None:
        """Clear all tracked pressed keys (e.g., on game pause/stop)."""
        self._pressed_keys.clear()

    def configure(
        self,
        game_id: GameId | None,
        action_space: object | None,
        *,
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        """Configure input handling for a specific game.

        Args:
            game_id: The game to configure for, or None to disable.
            action_space: The game's action space.
            overrides: Optional dict of game config overrides, may include 'input_mode'.

        The input mode is determined by the 'input_mode' key in overrides:
        - 'state_based': Track all pressed keys, compute combined actions (for diagonals)
        - 'shortcut_based' (default): Each key triggers immediate action
        """
        self._clear_shortcuts()
        self._pressed_keys.clear()
        self._current_game_id = game_id
        self._key_resolver = None
        self._use_state_based_input = False

        if game_id is None or action_space is None:
            return

        # Get input mode from game configuration (user's choice)
        # Default to shortcut_based for backward compatibility
        input_mode = InputMode.SHORTCUT_BASED.value
        if overrides:
            input_mode = overrides.get("input_mode", InputMode.SHORTCUT_BASED.value)

        use_state_based = (input_mode == InputMode.STATE_BASED.value)

        _LOGGER.info(
            "Configuring input for %s: mode=%s",
            game_id.value if hasattr(game_id, 'value') else game_id,
            input_mode,
        )

        if use_state_based:
            # Try to get a key combination resolver for this game
            self._key_resolver = get_key_combination_resolver(game_id)
            if self._key_resolver is not None:
                self._use_state_based_input = True
                _LOGGER.info(
                    "Using state-based input with %s resolver",
                    type(self._key_resolver).__name__,
                )
                return  # Don't set up shortcuts for state-based games
            else:
                _LOGGER.warning(
                    "No key combination resolver available for %s, falling back to shortcuts",
                    game_id.value if hasattr(game_id, 'value') else game_id,
                )

        # Fall back to shortcut-based input for turn-based games
        try:
            mappings = _TOY_TEXT_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _MINIG_GRID_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _BOX_2D_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _ALE_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _VIZDOOM_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _MINIHACK_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _NETHACK_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _CRAFTER_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _PROCGEN_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _JUMANJI_MAPPINGS.get(game_id)
            if mappings is None and isinstance(action_space, spaces.Discrete):
                mappings = self._fallback_mappings(action_space)

            if not mappings:
                return

            for mapping in mappings:
                for sequence in mapping.key_sequences:
                    shortcut = QShortcut(sequence, self._widget)
                    shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
                    shortcut_label = sequence.toString() or repr(sequence)
                    shortcut.activated.connect(self._make_activation(mapping.action, shortcut_label))
                    self._shortcuts.append(shortcut)
            self._update_shortcuts_enabled()
        except Exception as exc:  # pragma: no cover - defensive
            self.log_constant(
                LOG_INPUT_CONTROLLER_ERROR,
                exc_info=exc,
                extra={
                    "game_id": game_id.value if game_id else "unknown",
                    "space_type": type(action_space).__name__ if action_space is not None else "None",
                },
            )
            self._clear_shortcuts()

    def is_state_based(self) -> bool:
        """Return True if using state-based input for current game."""
        return self._use_state_based_input

    def set_enabled(self, enabled: bool) -> None:
        self._requested_enabled = enabled
        self._update_shortcuts_enabled()
        if not enabled:
            self._pressed_keys.clear()

    def _make_activation(self, action: int, shortcut_label: str) -> Callable[[], None]:
        def trigger() -> None:
            _LOGGER.debug("Shortcut activated key='%s' action=%s", shortcut_label, action)
            self._session.perform_human_action(action, key_label=shortcut_label)

        return trigger

    def _clear_shortcuts(self) -> None:
        while self._shortcuts:
            shortcut = self._shortcuts.pop()
            shortcut.deleteLater()

    @staticmethod
    def _fallback_mappings(action_space: spaces.Discrete) -> Tuple[ShortcutMapping, ...]:
        sequences: List[ShortcutMapping] = []
        base_mappings: List[Tuple[Iterable[str], int]] = [
            (("Key_Left", "Key_A"), 0),
            (("Key_Down", "Key_S"), 1),
            (("Key_Right", "Key_D"), 2),
            (("Key_Up", "Key_W"), 3),
            (("Key_Space",), 4),
            (("Key_E",), 5),
            (("Key_Q",), 6),
        ]
        for keys, action in base_mappings:
            if action >= action_space.n:
                continue
            sequences.append(_mapping(keys, action))
        return tuple(sequences)

    def update_for_mode(self, mode: ControlMode) -> None:
        self._mode_allows_input = mode in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }
        self._update_shortcuts_enabled()
        if not self._mode_allows_input:
            self._pressed_keys.clear()

    def _update_shortcuts_enabled(self) -> None:
        """Enable or disable all shortcuts based on current state."""
        enabled = self._mode_allows_input and self._requested_enabled
        for shortcut in self._shortcuts:
            shortcut.setEnabled(enabled)
        _LOGGER.debug("Shortcuts enabled=%s (mode_allows=%s, requested=%s)",
                          enabled, self._mode_allows_input, self._requested_enabled)


__all__ = [
    "HumanInputController",
    "get_vizdoom_mouse_turn_actions",
    "get_key_combination_resolver",
    "KeyCombinationResolver",
    "ProcgenKeyCombinationResolver",
    "AleKeyCombinationResolver",
    "LunarLanderKeyCombinationResolver",
    "CarRacingKeyCombinationResolver",
    "BipedalWalkerKeyCombinationResolver",
    "Box2DKeyCombinationResolver",  # Backwards compatibility alias
]
