from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for type checking only to satisfy Pylance without runtime deps
    from gym_gui.core.adapters.base import EnvironmentAdapter
    from gym_gui.rendering.interfaces import RendererStrategy
    from gym_gui.controllers.interaction import InteractionController


@dataclass(frozen=True)
class SessionComponents:
    adapter: "EnvironmentAdapter"
    renderer: "RendererStrategy"
    interaction: "InteractionController"
