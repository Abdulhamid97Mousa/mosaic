"""Renderer strategy interfaces for mapping environment payloads to Qt widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from qtpy import QtWidgets

from gym_gui.core.enums import GameId, RenderMode


@dataclass(slots=True)
class RendererContext:
    """Contextual metadata made available to renderer strategies."""

    game_id: GameId | None = None
    square_size: int | None = None  # For board games (Chess, Go, etc.)


class RendererStrategy(Protocol):
    """Contract implemented by rendering strategies."""

    mode: RenderMode

    @property
    def widget(self) -> QtWidgets.QWidget:
        """Return the Qt widget hosting the rendered output."""
        ...

    def render(self, payload: Mapping[str, object], *, context: RendererContext | None = None) -> None:
        """Render ``payload`` into the strategy's widget."""
        ...

    def supports(self, payload: Mapping[str, object]) -> bool:
        """Return whether the payload is compatible with this strategy."""
        ...

    def reset(self) -> None:
        """Clear the widget when no payload is available."""
        ...


__all__ = ["RendererContext", "RendererStrategy"]
