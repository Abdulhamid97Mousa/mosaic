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
_AGENT_TOKEN_HINTS = {"x"}


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
            if char == " ":
                # Preserve spacing so the grid matches the rendered layout and can be highlighted.
                row_chars.append(" ")
                if current_bg is not None:
                    agent_pos = (len(rows), len(row_chars) - 1)
            else:
                row_chars.append(char)
                if current_bg is not None:
                    agent_pos = (len(rows), len(row_chars) - 1)
            i += 1
        if row_chars:
            rows.append(row_chars)

    if agent_pos is None:
        hint = _find_agent_token(rows)
        if hint is not None:
            agent_pos = hint

    return rows, agent_pos


def _find_agent_token(grid: List[List[str]]) -> Tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if not value:
                continue
            if value.strip() == "":
                continue
            if value.lower() in _AGENT_TOKEN_HINTS:
                return r, c
    return None


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
        env_position = self._agent_position_from_state(grid)
        if env_position is not None:
            agent_pos = env_position
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

    def _agent_position_from_state(self, grid: List[List[str]]) -> Tuple[int, int] | None:
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        try:
            state = getattr(unwrapped, "s", None)
            width: int | None = None
            height: int | None = None
            row: int | None = None
            col: int | None = None

            if self.id == GameId.TAXI.value:
                decode = getattr(unwrapped, "decode", None)
                if state is None or decode is None:
                    return None
                taxi_row, taxi_col, *_ = decode(int(state))
                row = int(taxi_row)
                col = int(taxi_col)
                # Taxi map is always 5x5.
                height = 5
                width = 5
            else:
                if state is None:
                    return None
                if hasattr(unwrapped, "ncol"):
                    width = int(getattr(unwrapped, "ncol"))
                if hasattr(unwrapped, "nrow"):
                    height = int(getattr(unwrapped, "nrow"))
                if (width is None or height is None) and hasattr(unwrapped, "desc"):
                    desc = getattr(unwrapped, "desc")
                    try:
                        height = height or len(desc)
                        width = width or (len(desc[0]) if height else None)
                    except Exception:
                        pass
                if (width is None or height is None) and hasattr(unwrapped, "shape"):
                    shape = getattr(unwrapped, "shape")
                    if shape and len(shape) >= 2:
                        if height is None:
                            height = int(shape[0])
                        if width is None:
                            width = int(shape[1])
                if width is None or height is None:
                    return None
                row = int(state) // width
                col = int(state) % width

            if row is None or col is None or not grid:
                return None
            grid_row = min(max(row, 0), len(grid) - 1)
            row_chars = grid[grid_row]
            if not row_chars:
                return None
            if width <= 1:
                grid_col = 0
            else:
                span = len(row_chars) - 1
                base = width - 1
                if base <= 0:
                    grid_col = 0
                else:
                    ratio = col / base
                    grid_col = int(round(ratio * span))
            grid_col = min(max(grid_col, 0), len(row_chars) - 1)
            return grid_row, grid_col
        except Exception:
            return None


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
