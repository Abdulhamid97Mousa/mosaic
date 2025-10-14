"""Toy-text environment adapters for the Gym GUI."""

from __future__ import annotations

from pathlib import Path
import re
from collections.abc import Iterable
from typing import Any, List, Sequence, Tuple, Type

import gymnasium as gym

from gym_gui.config.game_configs import (
    FrozenLakeConfig,
    TaxiConfig,
    CliffWalkingConfig,
    DEFAULT_FROZEN_LAKE_CONFIG,
    DEFAULT_FROZEN_LAKE_V2_CONFIG,
    DEFAULT_TAXI_CONFIG,
    DEFAULT_CLIFF_WALKING_CONFIG,
)
from gym_gui.core.adapters.base import AdapterContext, EnvironmentAdapter
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_TOY_TEXT_DATA_DIR = Path(__file__).resolve().parents[2] / "runtime" / "data" / "toy_text"
_TOY_TEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[(?P<codes>[0-9;]*)m")
_AGENT_TOKEN_HINTS = {"x"}


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence):
        return value
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


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
    supported_render_modes = (RenderMode.GRID,)
    _gym_render_mode = "ansi"

    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    def __init__(self, context: AdapterContext | None = None) -> None:
        """Initialize toy-text adapter with step tracking."""
        super().__init__(context)
        self._last_terminated: bool = False
        self._last_truncated: bool = False

    def load(self) -> None:
        kwargs = self.gym_kwargs()
        env = gym.make(self.id, render_mode=self._gym_render_mode, **kwargs)
        self.logger.debug("Loaded toy-text env '%s' with kwargs=%s", self.id, kwargs)
        self._set_env(env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset and clear step tracking."""
        self._last_terminated = False
        self._last_truncated = False
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        """Step and track terminated/truncated states."""
        result = super().step(action)
        self._last_terminated = bool(result.terminated)
        self._last_truncated = bool(result.truncated)

        payload = result.render_payload
        if isinstance(payload, dict):
            enriched = dict(payload)
            enriched["terminated"] = self._last_terminated
            enriched["truncated"] = self._last_truncated
            enriched.setdefault("game_id", self.id)
            result.render_payload = enriched

        return result

    def render(self) -> dict[str, Any]:
        env = self._require_env()
        ansi_raw = env.render()
        ansi = str(ansi_raw)
        grid, agent_pos = _ansi_to_grid(ansi)
        env_position = self._agent_position_from_state(grid)
        if env_position is not None:
            agent_pos = env_position
        snapshot_path = _TOY_TEXT_DATA_DIR / f"{self.id}.txt"
        snapshot_path.write_text(ansi, encoding="utf-8")
        
        payload = {
            "mode": RenderMode.GRID.value,
            "grid": grid,
            "ansi": ansi,
            "snapshot_path": str(snapshot_path),
            "agent_position": agent_pos,
            "game_id": self.id,
        }
        
        # Add Taxi-specific state info
        if self.id == GameId.TAXI.value:
            unwrapped = getattr(env, "unwrapped", env)
            state = getattr(unwrapped, "s", None)
            decode = getattr(unwrapped, "decode", None)
            if state is not None and callable(decode):
                raw_decoded = decode(int(state))
                decoded = _ensure_sequence(raw_decoded)
                if len(decoded) >= 4:
                    taxi_row = decoded[0]
                    taxi_col = decoded[1]
                    pass_idx = decoded[2]
                    dest_idx = decoded[3]
                    payload["taxi_state"] = {
                        "taxi_position": (int(taxi_row), int(taxi_col)),
                        "passenger_index": int(pass_idx),  # 0-3 = at depot R/G/Y/B, 4 = in taxi
                        "destination_index": int(dest_idx),  # 0-3 = depot R/G/Y/B
                    }
        
        return payload

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
                if state is None or not callable(decode):
                    return None
                raw_decoded = decode(int(state))
                decoded = _ensure_sequence(raw_decoded)
                if len(decoded) < 2:
                    return None
                taxi_row = decoded[0]
                taxi_col = decoded[1]
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
    """Adapter for FrozenLake environment with game-specific configuration."""
    
    id = GameId.FROZEN_LAKE.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: FrozenLakeConfig | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context)
        self._game_config = game_config or DEFAULT_FROZEN_LAKE_CONFIG
        self._last_action: int | None = None

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset state and clear the last action tracker."""
        self._last_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        """Record the most recent action for orientation-aware rendering."""
        self._last_action = int(action)
        return super().step(action)

    def render(self) -> dict[str, Any]:
        """Render with FrozenLake-specific terminated state."""
        payload = super().render()
        
        # Add terminated state for cracked_hole visualization
        payload["terminated"] = self._last_terminated
        payload["truncated"] = self._last_truncated
        payload["last_action"] = self._last_action
        
        return payload


class FrozenLakeV2Adapter(ToyTextAdapter):
    """Adapter for FrozenLake-v2 with configurable grid, start, goal, and hole count."""
    
    id = GameId.FROZEN_LAKE_V2.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: FrozenLakeConfig | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context)
        self._game_config = game_config or DEFAULT_FROZEN_LAKE_V2_CONFIG
        self._last_action: int | None = None
        self._custom_desc: list[str] | None = None

    def _generate_map_descriptor(self) -> list[str]:
        """Generate custom map descriptor based on configuration.
        
        If random_holes=False and using standard 4×4 or 8×8 grid with default positions,
        returns the official Gymnasium default map. Otherwise generates a custom map.
        """
        import random
        
        height = self._game_config.grid_height
        width = self._game_config.grid_width
        start_pos = self._game_config.start_position or (0, 0)
        goal_pos = self._game_config.goal_position or (height - 1, width - 1)
        hole_count = self._game_config.hole_count
        random_holes = self._game_config.random_holes
        
        # Use official Gymnasium maps if conditions match
        if not random_holes and start_pos == (0, 0):
            if height == 4 and width == 4 and goal_pos == (3, 3):
                # Official 4×4 map from Gymnasium
                return [
                    "SFFF",
                    "FHFH",
                    "FFFH",
                    "HFFG"
                ]
            elif height == 8 and width == 8 and goal_pos == (7, 7):
                # Official 8×8 map from Gymnasium
                return [
                    "SFFFFFFF",
                    "FFFFFFFF",
                    "FFFHFFFF",
                    "FFFFFHFF",
                    "FFFHFFFF",
                    "FHHFFFHF",
                    "FHFFHFHF",
                    "FFFHFFFG",
                ]
        
        # Generate random/custom map
        # Default hole count if not specified
        if hole_count is None:
            total_tiles = height * width
            if height == 4 and width == 4:
                hole_count = 4  # Gymnasium default 4×4 map
            elif height == 8 and width == 8:
                hole_count = 10  # Gymnasium default 8×8 map
            else:
                # Scale holes proportionally
                hole_count = max(1, int(total_tiles * 0.15))  # ~15% holes
        
        # Initialize grid with frozen tiles
        grid = [['F' for _ in range(width)] for _ in range(height)]
        
        # Place start and goal
        grid[start_pos[0]][start_pos[1]] = 'S'
        grid[goal_pos[0]][goal_pos[1]] = 'G'
        
        # Collect available positions for holes (exclude start and goal)
        available_positions = [
            (r, c) for r in range(height) for c in range(width)
            if (r, c) != start_pos and (r, c) != goal_pos
        ]
        
        # Randomly place holes
        hole_count = min(hole_count, len(available_positions))
        hole_positions = random.sample(available_positions, hole_count)
        
        for r, c in hole_positions:
            grid[r][c] = 'H'
        
        # Convert to list of strings
        return [''.join(row) for row in grid]

    def load(self) -> None:
        """Load with custom map descriptor."""
        self._custom_desc = self._generate_map_descriptor()
        kwargs = {
            "is_slippery": self._game_config.is_slippery,
            "success_rate": self._game_config.success_rate,
            "reward_schedule": self._game_config.reward_schedule,
            "desc": self._custom_desc,
        }
        env = gym.make("FrozenLake-v1", render_mode=self._gym_render_mode, **kwargs)
        self.logger.debug("Loaded FrozenLake-v2 env with custom map size=%dx%d", 
                         self._game_config.grid_height, self._game_config.grid_width)
        self._set_env(env)

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs (handled in load)."""
        return {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset state and clear the last action tracker."""
        self._last_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        """Record the most recent action for orientation-aware rendering."""
        self._last_action = int(action)
        return super().step(action)

    def render(self) -> dict[str, Any]:
        """Render with FrozenLake-specific terminated state."""
        payload = super().render()
        
        # Add terminated state for cracked_hole visualization
        payload["terminated"] = self._last_terminated
        payload["truncated"] = self._last_truncated
        payload["last_action"] = self._last_action
        
        return payload

    @staticmethod
    def get_available_positions(grid_height: int, grid_width: int, start_position: tuple[int, int], exclude_holes: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
        """Get list of valid positions for goal selection (excludes start and existing holes)."""
        exclude_holes = exclude_holes or []
        excluded_set = {start_position} | set(exclude_holes)
        
        return [
            (r, c) for r in range(grid_height) for c in range(grid_width)
            if (r, c) not in excluded_set
        ]


class CliffWalkingAdapter(ToyTextAdapter):
    """Adapter for CliffWalking environment with game-specific configuration."""
    
    id = GameId.CLIFF_WALKING.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: CliffWalkingConfig | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context)
        self._game_config = game_config or DEFAULT_CLIFF_WALKING_CONFIG
        self._last_action: int | None = None

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset and clear last action."""
        self._last_action = None
        return super().reset(seed=seed, options=options)
    
    def step(self, action: int):
        """Step and track the action for rendering."""
        self._last_action = int(action)
        return super().step(action)
    
    def _agent_position_from_state(self, grid: List[List[str]]) -> Tuple[int, int] | None:
        """Override to correctly calculate position for CliffWalking's 4x12 grid."""
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        try:
            state = getattr(unwrapped, "s", None)
            if state is None:
                return None
            
            # CliffWalking is 4 rows × 12 columns
            width = 12
            
            # State is a single integer from 0-47
            row = int(state) // width
            col = int(state) % width
            
            return (row, col)
        except Exception:
            return None
    
    def render(self) -> dict[str, Any]:
        """Override render to strip spacing columns from CliffWalking ANSI grid.
        
        CliffWalking ANSI format includes spacing between cells: 'o  o  o...'
        This creates a 34-column grid (['o', ' ', ' ', 'o', ' ', ' ', ...])
        We need to remove the spacing columns to get a clean 12-column grid.
        """
        # Get the base render payload with the raw grid
        payload = super().render()
        
        # Extract the grid with spacing
        grid_with_spacing = payload.get("grid", [])
        if not grid_with_spacing:
            return payload
        
        # Strip spacing columns: keep only columns 0, 3, 6, 9, ... (every 3rd)
        clean_grid = []
        for row in grid_with_spacing:
            clean_row = [row[col] for col in range(len(row)) if col % 3 == 0]
            clean_grid.append(clean_row)
        
        # Update payload with clean grid
        payload["grid"] = clean_grid
        
        # Agent position was already calculated correctly by our override above
        # (using 4x12 dimensions), so no adjustment needed
        
        # Add last action for directional elf rendering
        payload["last_action"] = self._last_action
        
        return payload


class TaxiAdapter(ToyTextAdapter):
    """Adapter for Taxi-v3 environment with game-specific configuration."""
    
    id = GameId.TAXI.value
    
    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: TaxiConfig | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context)
        self._game_config = game_config or DEFAULT_TAXI_CONFIG
        self._last_action: int | None = None

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset and clear last action."""
        self._last_action = None
        return super().reset(seed=seed, options=options)
    
    def step(self, action: int):
        """Step and track the action for rendering."""
        self._last_action = int(action)
        return super().step(action)
    
    def render(self) -> dict[str, Any]:
        """Override render to convert taxi position from 5×5 logical to 11×11 grid coordinates."""
        # Get the base render payload with full 11×11 ANSI grid
        payload = super().render()
        
        # Taxi uses an 11×11 character grid with borders
        # The taxi_state has taxi_position in 5×5 logical coordinates
        # We need to convert it to 11×11 grid coordinates
        
        taxi_state = payload.get("taxi_state")
        if taxi_state and "taxi_position" in taxi_state:
            logical_row, logical_col = taxi_state["taxi_position"]
            
            # Convert 5×5 logical coordinates to 11×11 grid coordinates
            # Logical row N → grid row (N + 1)
            # Logical col N → grid col (N * 2 + 1)
            # This accounts for borders and spacing in the ANSI grid
            grid_row = logical_row + 1  # Skip top border row
            grid_col = logical_col * 2 + 1  # Account for spacing (each cell takes 2 chars)
            
            payload["agent_position"] = (grid_row, grid_col)
            # Add last action for directional cab rendering
            taxi_state["last_action"] = self._last_action
        
        return payload


TOY_TEXT_ADAPTERS: dict[GameId, Type[ToyTextAdapter]] = {
    GameId.FROZEN_LAKE: FrozenLakeAdapter,
    GameId.FROZEN_LAKE_V2: FrozenLakeV2Adapter,
    GameId.CLIFF_WALKING: CliffWalkingAdapter,
    GameId.TAXI: TaxiAdapter,
}

__all__ = [
    "ToyTextAdapter",
    "FrozenLakeAdapter",
    "FrozenLakeV2Adapter",
    "CliffWalkingAdapter",
    "TaxiAdapter",
    "TOY_TEXT_ADAPTERS",
]