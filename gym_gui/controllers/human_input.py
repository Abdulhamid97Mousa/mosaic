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


def _key_sequences(*names: str) -> Tuple[QKeySequence, ...]:
    return tuple(QKeySequence(_qt_key(name)) for name in names)


_TOY_TEXT_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.FROZEN_LAKE: (
        ShortcutMapping(_key_sequences("Key_Left", "Key_A"), 0),
        ShortcutMapping(_key_sequences("Key_Down", "Key_S"), 1),
        ShortcutMapping(_key_sequences("Key_Right", "Key_D"), 2),
        ShortcutMapping(_key_sequences("Key_Up", "Key_W"), 3),
    ),
    GameId.CLIFF_WALKING: (
        ShortcutMapping(_key_sequences("Key_Left", "Key_A"), 0),
        ShortcutMapping(_key_sequences("Key_Down", "Key_S"), 1),
        ShortcutMapping(_key_sequences("Key_Right", "Key_D"), 2),
        ShortcutMapping(_key_sequences("Key_Up", "Key_W"), 3),
    ),
    GameId.TAXI: (
        ShortcutMapping(_key_sequences("Key_Left", "Key_A"), 0),
        ShortcutMapping(_key_sequences("Key_Down", "Key_S"), 1),
        ShortcutMapping(_key_sequences("Key_Right", "Key_D"), 2),
        ShortcutMapping(_key_sequences("Key_Up", "Key_W"), 3),
        ShortcutMapping(_key_sequences("Key_Space"), 4),
        ShortcutMapping(_key_sequences("Key_E"), 5),
    ),
}


class HumanInputController(QtCore.QObject):
    """Registers keyboard shortcuts and forwards them to the session controller."""

    def __init__(self, widget: QtWidgets.QWidget, session: SessionController) -> None:
        super().__init__(widget)
        self._widget = widget
        self._session = session
        self._shortcuts: List[QShortcut] = []
        self._enabled = True
        self._mode_allows_input = True
        self._requested_enabled = True
        self._logger = logging.getLogger("gym_gui.controllers.human_input")

    def configure(self, game_id: GameId | None, action_space: object | None) -> None:
        self._clear_shortcuts()
        if game_id is None or action_space is None:
            return

        mappings = _TOY_TEXT_MAPPINGS.get(game_id)
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

    def set_enabled(self, enabled: bool) -> None:
        self._requested_enabled = enabled
        self._enabled = self._mode_allows_input and self._requested_enabled

    def _make_activation(self, action: int, shortcut_label: str) -> Callable[[], None]:
        def trigger() -> None:
            if not self._enabled:
                return
            self._logger.debug("Shortcut activated key='%s' action=%s", shortcut_label, action)
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
        ]
        for keys, action in base_mappings:
            if action >= action_space.n:
                continue
            sequences.append(ShortcutMapping(_key_sequences(*keys), action))
        return tuple(sequences)

    def update_for_mode(self, mode: ControlMode) -> None:
        self._mode_allows_input = mode in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }
        self._enabled = self._mode_allows_input and self._requested_enabled


__all__ = ["HumanInputController"]
