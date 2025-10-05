"""Toy-text environment adapters for the Gym GUI."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, List, Tuple, Type

import gymnasium as gym

from gym_gui.core.adapters.base import EnvironmentAdapter
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_TOY_TEXT_DATA_DIR = Path(__file__).resolve().parents[2] / "runtime" / "data" / "toy_text"
_TOY_TEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[(?P<codes>[0-9;]*)m")


def _ansi_to_grid(ansi: str) -> Tuple[List[List[str]], Tuple[int, int] | None]:
    rows: List[List[str]] = []
    agent_pos: Tuple[int, int] | None = None

    for raw_line in ansi.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("("):
            continue

        row_chars: List[str] = []
        i = 0
        current_bg: int | None = None
        while i < len(raw_line):
            char = raw_line[i]
            if char == "\x1b":
                match = _ANSI_ESCAPE_RE.match(raw_line, i)
                if match:
                    codes = match.group("codes") or ""
                    if not codes:
                        current_bg = None
                    else:
                        for code in codes.split(";"):
                            if not code:
                                continue
                            if code == "0":
                                current_bg = None
                                continue
                            try:
                                value = int(code)
                            except ValueError:
                                continue
                            if 40 <= value <= 47 or 100 <= value <= 107:
                                current_bg = value
                    i = match.end()
                    continue
            if char != " ":
                row_chars.append(char)
                if current_bg is not None:
                    agent_pos = (len(rows), len(row_chars) - 1)
            i += 1
        if row_chars:
            rows.append(row_chars)

    return rows, agent_pos


class ToyTextAdapter(EnvironmentAdapter[int, int]):
    """Base adapter for Gymnasium toy-text environments."""

    default_render_mode = RenderMode.GRID
    _gym_render_mode = "ansi"

    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    def load(self) -> None:
        kwargs = self.gym_kwargs()
        env = gym.make(self.id, render_mode=self._gym_render_mode, **kwargs)
        self.logger.debug("Loaded toy-text env '%s' with kwargs=%s", self.id, kwargs)
        self._set_env(env)

    def render(self) -> dict[str, Any]:
        env = self._require_env()
        ansi_raw = env.render()
        ansi = str(ansi_raw)
        grid, agent_pos = _ansi_to_grid(ansi)
        snapshot_path = _TOY_TEXT_DATA_DIR / f"{self.id}_latest.txt"
        snapshot_path.write_text(ansi, encoding="utf-8")
        return {
            "mode": RenderMode.GRID.value,
            "grid": grid,
            "ansi": ansi,
            "snapshot_path": str(snapshot_path),
            "agent_position": agent_pos,
        }

    def gym_kwargs(self) -> dict[str, Any]:
        return {}


class FrozenLakeAdapter(ToyTextAdapter):
    id = GameId.FROZEN_LAKE.value

    def gym_kwargs(self) -> dict[str, Any]:
        settings = self.settings
        slippery = True
        map_name = "4x4"
        if settings is not None and hasattr(settings, "frozen_lake_is_slippery"):
            slippery = bool(getattr(settings, "frozen_lake_is_slippery"))
        if settings is not None and hasattr(settings, "frozen_lake_map_name"):
            map_name = getattr(settings, "frozen_lake_map_name")
        return {"is_slippery": slippery, "map_name": map_name}


class CliffWalkingAdapter(ToyTextAdapter):
    id = GameId.CLIFF_WALKING.value


class TaxiAdapter(ToyTextAdapter):
    id = GameId.TAXI.value


TOY_TEXT_ADAPTERS: dict[GameId, Type[ToyTextAdapter]] = {
    GameId.FROZEN_LAKE: FrozenLakeAdapter,
    GameId.CLIFF_WALKING: CliffWalkingAdapter,
    GameId.TAXI: TaxiAdapter,
}

__all__ = [
    "ToyTextAdapter",
    "FrozenLakeAdapter",
    "CliffWalkingAdapter",
    "TaxiAdapter",
    "TOY_TEXT_ADAPTERS",
]
