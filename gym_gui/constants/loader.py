"""Lightweight accessors for canonical Gym GUI constants.

This module intentionally keeps dependencies minimal so that the SPADE worker
can import shared defaults without dragging in heavy UI modules.
"""

from __future__ import annotations

from typing import Mapping, Iterable

from gym_gui.core.enums import GameId
from .constants_game import ToyTextDefaults, TOY_TEXT_DEFAULTS


def get_toy_text_defaults(game_id: GameId | str) -> ToyTextDefaults:
    """Return canonical toy-text defaults for the given game id."""
    try:
        normalized = game_id if isinstance(game_id, GameId) else GameId(game_id)
    except ValueError as exc:
        raise KeyError(f"{game_id!r} is not a recognised toy-text GameId") from exc
    try:
        return TOY_TEXT_DEFAULTS[normalized]
    except KeyError as exc:
        raise KeyError(f"No toy-text defaults registered for {normalized}") from exc


def list_toy_text_games() -> Iterable[GameId]:
    """Enumerate the registered toy-text GameIds."""
    return TOY_TEXT_DEFAULTS.keys()


__all__ = ["get_toy_text_defaults", "list_toy_text_games", "ToyTextDefaults"]
