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
                mappings = _BOX_2D_MAPPINGS.get(game_id)
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


__all__ = ["HumanInputController"]
