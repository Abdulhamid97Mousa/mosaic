"""Factory helpers for environment adapters."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, TypeVar

from gym_gui.core.adapters.base import AdapterContext, EnvironmentAdapter
from gym_gui.core.adapters.toy_text import TOY_TEXT_ADAPTERS, ToyTextAdapter
from gym_gui.core.enums import GameId

AdapterT = TypeVar("AdapterT", bound=EnvironmentAdapter)


class AdapterFactoryError(KeyError):
    """Raised when an adapter cannot be created for the requested game."""


@lru_cache(maxsize=None)
def _registry() -> Mapping[GameId, type[EnvironmentAdapter]]:
    """Build the adapter registry on first use.

    Toy-text adapters are the initial entries; future phases can extend this by
    importing additional modules and updating the mapping.
    """

    return {
        **TOY_TEXT_ADAPTERS,
    }


def available_games() -> Iterable[GameId]:
    """Return the set of :class:`GameId` values with registered adapters."""

    return _registry().keys()


def get_adapter_cls(game_id: GameId) -> type[EnvironmentAdapter]:
    """Look up the adapter class registered for ``game_id``."""

    try:
        return _registry()[game_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AdapterFactoryError(f"No adapter registered for '{game_id.value}'") from exc


def create_adapter(game_id: GameId, context: AdapterContext | None = None) -> EnvironmentAdapter:
    """Instantiate the adapter bound to the optional ``context``."""

    adapter_cls = get_adapter_cls(game_id)
    adapter = adapter_cls(context)
    if context is not None:
        adapter.ensure_control_mode(context.control_mode)
    return adapter


__all__ = [
    "AdapterFactoryError",
    "available_games",
    "create_adapter",
    "get_adapter_cls",
]
