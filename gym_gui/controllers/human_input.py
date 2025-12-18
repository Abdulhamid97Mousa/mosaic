from __future__ import annotations

"""Keyboard shortcut management for human control within the Qt shell."""

from dataclasses import dataclass
import logging
from typing import Callable, Dict, Iterable, List, Tuple

import gymnasium.spaces as spaces
from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QKeySequence, QShortcut

from gym_gui.core.enums import ControlMode, GameId
from gym_gui.controllers.session import SessionController
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import LOG_INPUT_CONTROLLER_ERROR


_LOGGER = logging.getLogger(__name__)


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


class HumanInputController(QtCore.QObject, LogConstantMixin):
    """Registers keyboard shortcuts and forwards them to the session controller."""

    def __init__(self, widget: QtWidgets.QWidget, session: SessionController) -> None:
        super().__init__(widget)
        self._logger = _LOGGER
        self._widget = widget
        self._session = session
        self._shortcuts: List[QShortcut] = []
        self._mode_allows_input = True
        self._requested_enabled = True

    def configure(self, game_id: GameId | None, action_space: object | None) -> None:
        self._clear_shortcuts()
        if game_id is None or action_space is None:
            return

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

    def set_enabled(self, enabled: bool) -> None:
        self._requested_enabled = enabled
        self._update_shortcuts_enabled()

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

    def _update_shortcuts_enabled(self) -> None:
        """Enable or disable all shortcuts based on current state."""
        enabled = self._mode_allows_input and self._requested_enabled
        for shortcut in self._shortcuts:
            shortcut.setEnabled(enabled)
        _LOGGER.debug("Shortcuts enabled=%s (mode_allows=%s, requested=%s)", 
                          enabled, self._mode_allows_input, self._requested_enabled)


__all__ = ["HumanInputController", "get_vizdoom_mouse_turn_actions"]
